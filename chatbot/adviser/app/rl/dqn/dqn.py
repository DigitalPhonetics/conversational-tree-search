###############################################################################
#
# Copyright 2020, University of Stuttgart: Institute for Natural Language Processing (IMS)
#
# This file is part of Adviser.
# Adviser is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3.
#
# Adviser is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Adviser.  If not, see <https://www.gnu.org/licenses/>.
#
###############################################################################

from enum import Enum
from typing import List, Tuple, Union
import random

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence
from chatbot.adviser.app.encoding.text import RNN_OUTPUT_SIZE
from chatbot.adviser.app.rl.layers.attention.attention_factory import get_attention_module_instance
from chatbot.adviser.app.rl.spaceAdapter import ActionConfig, SpaceAdapter


class NetArchitecture(Enum):
    """ Network architecture for DQN 
    
        vanilla: normal MLP
        dueling: splits network into value- and advantage stream, recombined in final layer
    """
    VANILLA = 'vanilla'
    DUELING = 'dueling'



class DQNNetwork(nn.Module):
    def __init__(self, adapter: SpaceAdapter) -> None:
        """
        Args:
            attention_fns: mapping from input (str) to: {attention_meachanism: str, active: bool, activation: str}
        """
        super().__init__()
        self.adapter = adapter
        attention_layers = {}
        for attn in adapter.attentioninput:
            if attn.active:
                attention_layers[attn.name] = get_attention_module_instance(attention_mechanism=attn.attention_mechanism,
                                                                                    vector_dim=attn.get_vector_dim(),
                                                                                    matrix_dim=attn.get_vector_dim(),
                                                                                    activation=attn.activation)
        self.attention_layers = nn.ModuleDict(attention_layers)

        rnn_layers = {}
        for text_key in adapter.stateinput.rnn_encoders:
                text_emb = adapter.stateinput.rnn_encoders[text_key]
                if text_key == 'dialog_history':
                    rnn_layers[text_key] = nn.GRU(text_emb.sys_text_enc_wrapper.text_embedding.get_encoding_dim() + 1, RNN_OUTPUT_SIZE, batch_first=True)  # + 1 because of of speaker embedding
                else:
                    rnn_layers[text_key] = nn.GRU(text_emb.text_embedding.get_encoding_dim(), RNN_OUTPUT_SIZE, batch_first=True)
        self.rnn_layers = nn.ModuleDict(rnn_layers)
    
    def process_attention(self, name: str, query: torch.FloatTensor, matrix: torch.FloatTensor) -> torch.FloatTensor: 
        # calculate attention weights
        attn_weights = self.attention_layers[name](query, matrix).transpose(1,0) # (1 x matrix_tokens) -> (matrix_tokens x 1) 
        # weight each matrix row (each token embedding) by its attention weight
        weighted_matrix = attn_weights * matrix # (matrix_tokens x 1)  * (1 x matrix_tokens x embedding_dim) -> (1 x matrix_tokens x embedding_dim) via broadcasting
        # summ all weighted token embeddings to context vector
        return torch.sum(weighted_matrix, dim=1) # 1 x embedding_dim

    def process_rnn(self, encodings: torch.FloatTensor, rnn_key: str):
        _, h = self.rnn_layers[rnn_key].forward(encodings) # discard output, use only last hidden state
        return h # 1 x 512
        

    @torch.no_grad()
    def select_actions_eps_greedy(self, node_keys: List[int], state_vectors: torch.FloatTensor, epsilon: float) -> Tuple[List[int], Union[List[int], None]]:
        """ Epsilon-greedy policy.

        Args:
            node_keys: current node keys
            state_vectors: current state (dimension batch x state_dim    or num_actions x state_dim)
            epsilon: current scheduled exploration rate

        Returns:
            List of action indices for action selected by the agent for the current states
            List of predicted intent classes if supported by model, else None
        """
        # TODO mask STOP action in first turn?

        # epsilon greedy exploration
        intent_classes = None
        if self.training and random.random() < epsilon:
            # exploration
            if self.adapter.configuration.action_masking:
                # only choose between allowed actions after masking
                allowed_action_indices = (~self.adapter.get_action_masks(node_keys=node_keys)).float() # batch x num_max_actions
                next_action_indices = [random.choice(allowed_action_indices[batch_idx].nonzero().view(-1).tolist()) for batch_idx in range(allowed_action_indices.size(0))] # 1-dim list with one valid action index per batch item
            else:
                # no masking: random choice
                next_action_indices = [random.randint(0, self.adapter.num_actions - 1) for _ in range(len(node_keys))]
        else:
            # exploitation
            train_state = self.training # save current training mode
            self.eval() # disable influence of dropout etc.
            q_values, intent_logits = self(state_vectors) # batch x num_max_actions
            if self.adapter.configuration.action_masking:
                q_values = torch.masked_fill(q_values, self.adapter.get_action_masks(node_keys=node_keys)[:,:q_values.size(-1)], float('-inf'))
            next_action_indices = q_values.argmax(-1).tolist()
            intent_classes = None if isinstance(intent_logits, type(None)) else (torch.sigmoid(intent_logits).view(-1) > 0.5).long()

            if train_state:
                # restore training mode
                self.train()

        return next_action_indices, intent_classes


    def pad_and_mask(self, state_batch: Union[torch.FloatTensor, torch.nn.utils.rnn.PackedSequence]) -> Tuple[torch.FloatTensor, torch.LongTensor, torch.FloatTensor]:
        """
        Returns:
            padded_sequence: padded state_batch tensor (batch x max_actions x state_dim)
            num_actions: number of actions per batch item (batch)
            mask: mask for outputs where num_actions for batch item was smaller than max_actions (batch x max_actions)
        """
        if isinstance(state_batch, torch.nn.utils.rnn.PackedSequence):
            padded_sequence, num_actions = pad_packed_sequence(state_batch, batch_first=True) # batch x max_actions x state_dim,  batch
        else:
            assert state_batch.ndim == 2, "can't forward batched state in non-packed form"
            padded_sequence = state_batch.unsqueeze(0) # 1 x actions x embedding_dim
            num_actions = torch.tensor([padded_sequence.size(1)], dtype=torch.long)
        num_actions = num_actions.to(padded_sequence.device)
        max_actions = padded_sequence.size(1)
        mask = ~(torch.arange(max_actions, device=padded_sequence.device)[None, :] < num_actions[:, None]) # batch x max_actions (negate mask!)

        return padded_sequence, num_actions, mask
    

def _add_weight_normalization(layers: nn.ModuleList, input_dim: int, normalize: bool):
    if normalize:
        layers.append(nn.LayerNorm(normalized_shape=input_dim))


class DQN(DQNNetwork):
    """ Simple Deep Q-Network """

    def __init__(self, adapter: SpaceAdapter, hidden_layer_sizes: List[int] = [1024, 1024],
                 dropout_rate: float = 0.0, activation_fn = nn.ReLU, normalization_layers: bool = False,
                 q_value_clipping: float = 0.0):
        """ Initialize a DQN Network with an arbitrary amount of linear hidden
            layers """

        super(DQN, self).__init__(adapter=adapter)
        print("Architecture: DQN")

        self.dropout_rate = dropout_rate
        self.q_value_clipping = q_value_clipping

        # create layers
        self.layers = nn.ModuleList()
        current_input_dim = adapter.get_state_dim()
        for layer_size in hidden_layer_sizes:
            self.layers.append(nn.Linear(current_input_dim, layer_size))
            _add_weight_normalization(self.layers, layer_size, normalization_layers)
            self.layers.append(activation_fn())
            if dropout_rate > 0.0:
                self.layers.append(nn.Dropout(p=dropout_rate))
            current_input_dim = layer_size
        # output layer
        self.layers.append(nn.Linear(current_input_dim, adapter.get_action_dim()))

    
    def _forward(self, x: torch.FloatTensor, mask: Union[torch.FloatTensor, None] = None):
        output = x
        for layer in self.layers:
            output = layer(output)
            if not isinstance(mask, type(None)):
                output = torch.masked_fill(output, mask, 0.0)
        # q-value clipping
        if self.q_value_clipping > 0.0:
            return output.clip(min=None, max=self.q_value_clipping) # don't mask min because of "inf" masking
        return output


    def forward(self, state_batch: Union[torch.FloatTensor, torch.nn.utils.rnn.PackedSequence]) -> Tuple[torch.FloatTensor, None]:
        """ Forward pass: calculate Q(state) for all actions

        Args:
            state_batch (torch.FloatTensor): tensor of size batch_size x state_dim if adapter.action_config = ACTIONS_IN_ACTION_SPACE
                        (torch.PackedSequence): of length batch_size, each entry actions x state_dim        = ACTIONS_IN_STATE_SPACE

        Returns:
            output: tensor of size batch_size x action_dim,  if adapter.action_config = ACTIONS_IN_ACTION_SPACE
                                   batch_sixe x max_actions,                          = ACTIONS_IN_STATE_SPACE
        """
        if self.adapter.configuration.action_config == ActionConfig.ACTIONS_IN_STATE_SPACE:
            # forward 1 time per action
            padded_sequence, num_actions, mask = self.pad_and_mask(state_batch)
            y = self._forward(padded_sequence, mask.unsqueeze(2)) # batch x max_actions x action_dim = 1
            y = y.squeeze(2) # batch x max_actions
            # mask off outputs for inputs with less actions than max_actions
            y = torch.masked_fill(y, mask, float('-inf')) # batch x max_actions  (mask Q values)
            return y, None
        else:
            # forward once for all actions (normal behaviour)
            return self._forward(state_batch), None



class DuelingDQN(DQNNetwork):
    """ Dueling DQN network architecture

    Splits network into value- and advantage stream (V(s) and A(s,a)), 
    recombined in final layer to form Q-value again:
    Q(s,a) = V(s) + A(s,a).

    """

    def __init__(self, adapter: SpaceAdapter,
                 shared_layer_sizes: List[int] = [128], value_layer_sizes: List[int] = [128],
                 advantage_layer_sizes: List[int] = [128], dropout_rate: float = 0.0,
                 activation_fn = nn.ReLU, normalization_layers: bool = False,
                 q_value_clipping: float = 0.0):
        super(DuelingDQN, self).__init__(adapter=adapter)
        print("ARCHITECTURE: Dueling")

        self.dropout_rate = dropout_rate
        self.q_value_clipping = q_value_clipping

        # configure layers
        self.shared_layers = nn.ModuleList()
        self.value_layers = nn.ModuleList()
        self.advantage_layers = nn.ModuleList()

        # shared layer: state_dim -> shared_layer_sizes[-1]
        shared_layer_dim = adapter.get_state_dim()
        for layer_size in shared_layer_sizes:
            self.shared_layers.append(nn.Linear(shared_layer_dim, layer_size))
            _add_weight_normalization(self.shared_layers, layer_size, normalization_layers)
            self.shared_layers.append(activation_fn())
            if dropout_rate > 0.0:
                self.shared_layers.append(nn.Dropout(p=dropout_rate))
            shared_layer_dim = layer_size
        # value layer: shared_layer_sizes[-1] -> 1
        value_layer_dim = shared_layer_dim
        for layer_size in value_layer_sizes:
            self.value_layers.append(nn.Linear(value_layer_dim, layer_size))
            _add_weight_normalization(self.value_layers, layer_size, normalization_layers)
            self.value_layers.append(activation_fn())
            if dropout_rate > 0.0:
                self.value_layers.append(nn.Dropout(p=dropout_rate))
            value_layer_dim = layer_size
        self.value_layers.append(nn.Linear(value_layer_dim, 1))
        # advantage layer: shared_layer_sizes[-1] -> actions
        advantage_layer_dim = shared_layer_dim
        for layer_size in advantage_layer_sizes:
            self.advantage_layers.append(nn.Linear(advantage_layer_dim, layer_size))
            _add_weight_normalization(self.advantage_layers, layer_size, normalization_layers)
            self.advantage_layers.append(activation_fn())
            if dropout_rate > 0.0:
                self.advantage_layers.append(nn.Dropout(p=dropout_rate))
            advantage_layer_dim = layer_size
        self.advantage_layers.append(nn.Linear(advantage_layer_dim, adapter.get_action_dim()))


    def forward(self, state_batch: Union[torch.FloatTensor, torch.nn.utils.rnn.PackedSequence]) -> Tuple[torch.LongTensor, None]:
        """ Forward pass: calculate Q(state) for all actions

        Args:
            state_batch (torch.FloatTensor): tensor of size batch_size x state_dim if adapter.action_config = ACTIONS_IN_ACTION_SPACE
                        (torch.PackedSequence): of length batch_size, each entry actions x state_dim        = ACTIONS_IN_STATE_SPACE

        Returns:
            output: tensor of size batch_size x action_dim,  if adapter.action_config = ACTIONS_IN_ACTION_SPACE
                                   batch_sixe x max_actions,                          = ACTIONS_IN_STATE_SPACE
        """
      
        if self.adapter.configuration.action_config == ActionConfig.ACTIONS_IN_STATE_SPACE:
            # forward 1 time per action
            shared_output, num_actions, mask = self.pad_and_mask(state_batch)

            for layer in self.shared_layers:
                shared_output = layer(shared_output)
                shared_output = torch.masked_fill(shared_output, mask.unsqueeze(2), 0.0)
            value_stream = shared_output
            for layer in self.value_layers:
                value_stream = layer(value_stream)
                value_stream = torch.masked_fill(value_stream, mask.unsqueeze(2), 0.0)
            # advantage stream
            advantage_stream = shared_output
            for layer in self.advantage_layers:
                advantage_stream = layer(advantage_stream)
                advantage_stream = torch.masked_fill(advantage_stream, mask.unsqueeze(2), 0.0)
            # exclude padded advantages
            advantage_stream = advantage_stream.squeeze(2) # batch x max_actions x 1 -> batch x mask_actions  (mask Q values)
            advantage_stream_mean = advantage_stream.sum(1) / num_actions.float() # batch 
            # combine value and advantage streams into Q values
            result = value_stream.squeeze(2) + advantage_stream - advantage_stream_mean.unsqueeze(1) # batch x max_actions
            result = torch.masked_fill(result, mask, float('-inf')) # batch x max_actions (mask Q values)
        else:
            shared_output = state_batch

            # shared layer representation
            for layer in self.shared_layers:
                shared_output = layer(shared_output)
            # value stream
            value_stream = shared_output
            for layer in self.value_layers:
                value_stream = layer(value_stream)
            # advantage stream
            advantage_stream = shared_output
            for layer in self.advantage_layers:
                advantage_stream = layer(advantage_stream)
            # combine value and advantage streams into Q values
            result = value_stream + advantage_stream - advantage_stream.mean(-1).unsqueeze(1)

        if self.q_value_clipping > 0.0:
            return result.clip(min=None, max=self.q_value_clipping), None # don't mask min because of "inf" masking
        return result, None


class DuelingDQNWithIntentPredictionHead(DuelingDQN):
    def __init__(self, adapter: SpaceAdapter,
                 shared_layer_sizes: List[int] = [128], value_layer_sizes: List[int] = [128],
                 advantage_layer_sizes: List[int] = [128], dropout_rate: float = 0.0,
                 activation_fn = nn.ReLU, normalization_layers: bool = False,
                 q_value_clipping: float = 0.0):

        super(DuelingDQNWithIntentPredictionHead, self).__init__(adapter=adapter, shared_layer_sizes=shared_layer_sizes, value_layer_sizes=value_layer_sizes,
                                        advantage_layer_sizes=advantage_layer_sizes, dropout_rate=dropout_rate, activation_fn=activation_fn, normalization_layers=normalization_layers,
                                        q_value_clipping=q_value_clipping)
        print("ARCHITECTURE: Dueling with Intent Prediction Head")
        self.shared_layer_sizes = shared_layer_sizes

        shared_hidden_dim = self.shared_layer_sizes[-1]  # last linear layer ... -1 would be activation function
        self.intent_head = nn.Sequential(
            nn.Linear(shared_hidden_dim, 256),
            nn.Dropout(p=dropout_rate),
            nn.Linear(256, 1)
        )

    def forward(self, state_batch: Union[torch.FloatTensor, torch.nn.utils.rnn.PackedSequence]) -> Tuple[torch.LongTensor, torch.LongTensor]:
        """ Forward pass: calculate Q(state) for all actions

        Args:
            state_batch (torch.FloatTensor): tensor of size batch_size x state_dim if adapter.action_config = ACTIONS_IN_ACTION_SPACE
                        (torch.PackedSequence): of length batch_size, each entry actions x state_dim        = ACTIONS_IN_STATE_SPACE

        Returns:
            output: tensor of size batch_size x action_dim,  if adapter.action_config = ACTIONS_IN_ACTION_SPACE
                                   batch_sixe x max_actions,                          = ACTIONS_IN_STATE_SPACE
            logits: intent prediction head logits: batch x 2 (False: dialog, True: faq)
        """
      
        if self.adapter.configuration.action_config == ActionConfig.ACTIONS_IN_STATE_SPACE:
            # forward 1 time per action
            shared_output, num_actions, mask = self.pad_and_mask(state_batch)

            for layer in self.shared_layers:
                shared_output = layer(shared_output)
                shared_output = torch.masked_fill(shared_output, mask.unsqueeze(2), 0.0)
            value_stream = shared_output
            for layer in self.value_layers:
                value_stream = layer(value_stream)
                value_stream = torch.masked_fill(value_stream, mask.unsqueeze(2), 0.0)
            # advantage stream
            advantage_stream = shared_output
            for layer in self.advantage_layers:
                advantage_stream = layer(advantage_stream)
                advantage_stream = torch.masked_fill(advantage_stream, mask.unsqueeze(2), 0.0)
            # exclude padded advantages
            advantage_stream = advantage_stream.squeeze(2) # batch x max_actions x 1 -> batch x mask_actions  (mask Q values)
            advantage_stream_mean = advantage_stream.sum(1) / num_actions.float() # batch 
            # combine value and advantage streams into Q values
            result = value_stream.squeeze(2) + advantage_stream - advantage_stream_mean.unsqueeze(1) # batch x max_actions
            result = torch.masked_fill(result, mask, float('-inf')) # batch x max_actions (mask Q values)

            # predict user intent
            intent_logits = shared_output.max(dim=1)[0] # batch x shared_dim
            intent_logits = self.intent_head(intent_logits) # batch x 1
        else:
            shared_output = state_batch

            # shared layer representation
            for layer in self.shared_layers:
                shared_output = layer(shared_output)
            # value stream
            value_stream = shared_output
            for layer in self.value_layers:
                value_stream = layer(value_stream)
            # advantage stream
            advantage_stream = shared_output
            for layer in self.advantage_layers:
                advantage_stream = layer(advantage_stream)
            # combine value and advantage streams into Q values
            result = value_stream + advantage_stream - advantage_stream.mean(-1).unsqueeze(1)
    
            # predict user intent
            intent_logits = self.intent_head(shared_output) # batch x 1

        if self.q_value_clipping > 0.0:
            result = result.clip(min=None, max=self.q_value_clipping) # don't mask min because of "inf" masking
        return result, intent_logits



class NewDuelingDQNWithIntentPredictionHead(DQNNetwork):
    def __init__(self, adapter: SpaceAdapter,
                 shared_layer_sizes: List[int] = [128], value_layer_sizes: List[int] = [128],
                 advantage_layer_sizes: List[int] = [128], dropout_rate: float = 0.0,
                 activation_fn = nn.ReLU, normalization_layers: bool = False,
                 q_value_clipping: float = 0.0):

        super(NewDuelingDQNWithIntentPredictionHead, self).__init__(adapter=adapter)
        print("ARCHITECTURE: New Dueling with Intent Prediction Head")
        assert self.adapter.configuration.action_config == ActionConfig.ACTIONS_IN_STATE_SPACE

        self.dropout_rate = dropout_rate
        self.q_value_clipping = q_value_clipping

        # configure layers
        self.shared_layers = nn.ModuleList()
        self.action_inputs = nn.ModuleList()
        self.value_layers = nn.ModuleList()
        self.advantage_layers = nn.ModuleList()

        self.state_input_size = adapter.get_state_dim() - adapter.get_actionstatesubvector_dim() # shared layer without action inputs

        # shared layer: state_dim -> shared_layer_sizes[-1]
        shared_layer_dim = adapter.get_state_dim() - adapter.get_actionstatesubvector_dim() # shared layer without action inputs
        action_layer_dim = adapter.get_actionstatesubvector_dim()
        for layer_size in shared_layer_sizes:
            self.shared_layers.append(nn.Linear(shared_layer_dim, layer_size))
            _add_weight_normalization(self.shared_layers, layer_size, normalization_layers)
            self.shared_layers.append(activation_fn())
            self.action_inputs.append(nn.Linear(action_layer_dim, layer_size))
            _add_weight_normalization(self.action_inputs, layer_size, normalization_layers)
            self.action_inputs.append(activation_fn())
            if dropout_rate > 0.0:
                self.shared_layers.append(nn.Dropout(p=dropout_rate))
                self.action_inputs.append(nn.Dropout(p=dropout_rate))
            shared_layer_dim = layer_size
            action_layer_dim = layer_size
        # value layer: shared_layer_sizes[-1] -> 1
        value_layer_dim = shared_layer_dim
        for layer_size in value_layer_sizes:
            self.value_layers.append(nn.Linear(value_layer_dim, layer_size))
            _add_weight_normalization(self.value_layers, layer_size, normalization_layers)
            self.value_layers.append(activation_fn())
            if dropout_rate > 0.0:
                self.value_layers.append(nn.Dropout(p=dropout_rate))
            value_layer_dim = layer_size
        self.value_layers.append(nn.Linear(value_layer_dim, 1))
        # advantage layer: shared_layer_sizes[-1] -> actions
        advantage_layer_dim = 2 * shared_layer_dim # shared layer output + action inputs
        for layer_size in advantage_layer_sizes:
            self.advantage_layers.append(nn.Linear(advantage_layer_dim, layer_size))
            _add_weight_normalization(self.advantage_layers, layer_size, normalization_layers)
            self.advantage_layers.append(activation_fn())
            if dropout_rate > 0.0:
                self.advantage_layers.append(nn.Dropout(p=dropout_rate))
            advantage_layer_dim = layer_size
        self.advantage_layers.append(nn.Linear(advantage_layer_dim, adapter.get_action_dim()))

        self.shared_layer_sizes = shared_layer_sizes

        shared_hidden_dim = self.shared_layer_sizes[-1]  # last linear layer ... -1 would be activation function
        self.intent_head = nn.Sequential(
            nn.Linear(shared_hidden_dim, 256),
            nn.Dropout(p=dropout_rate),
            nn.Linear(256, 1)
        )

    def forward(self, state_batch: Union[torch.FloatTensor, torch.nn.utils.rnn.PackedSequence]) -> Tuple[torch.LongTensor, torch.LongTensor]:
        """ Forward pass: calculate Q(state) for all actions

        Args:
            state_batch (torch.FloatTensor): tensor of size batch_size x state_dim if adapter.action_config = ACTIONS_IN_ACTION_SPACE
                        (torch.PackedSequence): of length batch_size, each entry actions x state_dim        = ACTIONS_IN_STATE_SPACE

        Returns:
            output: tensor of size batch_size x action_dim,  if adapter.action_config = ACTIONS_IN_ACTION_SPACE
                                   batch_sixe x max_actions,                          = ACTIONS_IN_STATE_SPACE
            logits: intent prediction head logits: batch x 2 (False: dialog, True: faq)
        """
      
        # forward 1 time per action
        padded_inputs, num_actions, mask = self.pad_and_mask(state_batch)
        shared_output = padded_inputs[:, :, :self.state_input_size] # state portion of inputs
        action_input = padded_inputs[:, :, self.state_input_size:] # action portion of inputs

        for layer in self.shared_layers:
            shared_output = layer(shared_output)
            shared_output = torch.masked_fill(shared_output, mask.unsqueeze(2), 0.0)
        for layer in self.action_inputs:
            action_input = layer(action_input)
            action_input = torch.masked_fill(action_input, mask.unsqueeze(2), 0.0)
        value_stream = shared_output
        for layer in self.value_layers:
            value_stream = layer(value_stream)
            value_stream = torch.masked_fill(value_stream, mask.unsqueeze(2), 0.0)
        # advantage stream
        advantage_stream = torch.cat([shared_output, action_input], -1)
        for layer in self.advantage_layers:
            advantage_stream = layer(advantage_stream)
            advantage_stream = torch.masked_fill(advantage_stream, mask.unsqueeze(2), 0.0)
        # exclude padded advantages
        advantage_stream = advantage_stream.squeeze(2) # batch x max_actions x 1 -> batch x mask_actions  (mask Q values)
        advantage_stream_mean = advantage_stream.sum(1) / num_actions.float() # batch 
        # combine value and advantage streams into Q values
        result = value_stream.squeeze(2) + advantage_stream - advantage_stream_mean.unsqueeze(1) # batch x max_actions
        result = torch.masked_fill(result, mask, float('-inf')) # batch x max_actions (mask Q values)

        # predict user intent
        intent_logits = shared_output.max(dim=1)[0] # batch x shared_dim
        intent_logits = self.intent_head(intent_logits) # batch x 1
        
        if self.q_value_clipping > 0.0:
            result = result.clip(min=None, max=self.q_value_clipping) # don't mask min because of "inf" masking
        return result, intent_logits



# class DuelingTransformerDQN(DQNNetwork):
#     """ Dueling DQN network architecture

#     Splits network into value- and advantage stream (V(s) and A(s,a)), 
#     recombined in final layer to form Q-value again:
#     Q(s,a) = V(s) + A(s,a).

#     """

#     def __init__(self, adapter: SpaceAdapter,
#                  shared_layer_sizes: List[int] = [128], value_layer_sizes: List[int] = [128],
#                  advantage_layer_sizes: List[int] = [128], dropout_rate: float = 0.0,
#                  activation_fn = nn.ReLU, normalization_layers: bool = False):
#         super(DuelingDQN, self).__init__(adapter=adapter)
#         print("ARCHITECTURE: Dueling")

#         self.dropout_rate = dropout_rate

#         # configure layers
#         self.shared_layers = nn.Transformer(d_model=shared_layer_sizes[0], nhead=8, num_encoder_layers=len(shared_layer_sizes), num_decoder_layers=0,
#                                             batch_first=True)
#         self.value_layers = nn.ModuleList()
#         self.advantage_layers = nn.ModuleList()

#         # value layer: shared_layer_sizes[-1] -> 1
#         value_layer_dim = shared_layer_sizes[0]
#         for layer_size in value_layer_sizes:
#             self.value_layers.append(_add_weight_normalization(nn.Linear(value_layer_dim, layer_size), normalization_layers))
#             self.value_layers.append(activation_fn())
#             if dropout_rate > 0.0:
#                 self.value_layers.append(nn.Dropout(p=dropout_rate))
#             value_layer_dim = layer_size
#         self.value_layers.append(nn.Linear(value_layer_dim, 1))
#         # advantage layer: shared_layer_sizes[-1] -> actions
#         advantage_layer_dim = shared_layer_sizes[0]
#         for layer_size in advantage_layer_sizes:
#             self.advantage_layers.append(_add_weight_normalization(nn.Linear(advantage_layer_dim, layer_size), normalization_layers))
#             self.advantage_layers.append(activation_fn())
#             if dropout_rate > 0.0:
#                 self.advantage_layers.append(nn.Dropout(p=dropout_rate))
#             advantage_layer_dim = layer_size
#         self.advantage_layers.append(nn.Linear(advantage_layer_dim, adapter.get_action_dim()))


#     def forward(self, state_batch: Union[torch.FloatTensor, torch.nn.utils.rnn.PackedSequence]):
#         """ Forward pass: calculate Q(state) for all actions

#         Args:
#             state_batch (torch.FloatTensor): tensor of size batch_size x state_dim if adapter.action_config = ACTIONS_IN_ACTION_SPACE
#                         (torch.PackedSequence): of length batch_size, each entry actions x state_dim        = ACTIONS_IN_STATE_SPACE

#         Returns:
#             output: tensor of size batch_size x action_dim,  if adapter.action_config = ACTIONS_IN_ACTION_SPACE
#                                    batch_sixe x max_actions,                          = ACTIONS_IN_STATE_SPACE
#         """
      
#         # TODO src masking -> forward -> pooling -> advantage + value layers
#         pass