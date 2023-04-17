from typing import List, Tuple, Union

import torch
import torch.nn as nn
from models.layers.utils import add_weight_normalization_layer

from models.dqn.base import DQNBase, NetworkOutput

class DQN(DQNBase):
    def __init__(self, intent_prediction: bool,
                    hidden_layer_sizes: List[int],
                    dropout_rate: float,
                    normalization_layers: bool,
                    activation_fn: torch.nn.Module) -> None:
        print("DQN")
        super().__init__()

        self.intent_prediction = intent_prediction
        self.hidden_layer_sizes = hidden_layer_sizes
        self.dropout_rate = dropout_rate
        self.normalization_layers = normalization_layers
        self.activation_fn = activation_fn

    def build(self, state_dim: int, action_dim: int, actions_in_state_space: bool):
        self.actions_in_state_space = actions_in_state_space

        # setup network layers
        self.layers = nn.ModuleList()
        current_input_dim = state_dim
        for layer_size in self.hidden_layer_sizes:
            self.layers.append(add_weight_normalization_layer(nn.Linear(current_input_dim, layer_size), self.normalization_layers))
            self.layers.append(self.activation_fn())
            if self.dropout_rate > 0.0:
                self.layers.append(nn.Dropout(p=self.dropout_rate))
            current_input_dim = layer_size
        # output layer
        self.layers.append(nn.Linear(current_input_dim, action_dim))

    def _forward(self, x: torch.FloatTensor, mask: Union[torch.FloatTensor, None] = None) -> torch.FloatTensor:
        output = x
        for layer in self.layers:
            output = layer(output)
            if not isinstance(mask, type(None)):
                output = torch.masked_fill(output, mask, 0.0)
        return output

    def forward(self, state_batch: Union[torch.FloatTensor, torch.nn.utils.rnn.PackedSequence]) -> NetworkOutput:
        """ Forward pass: calculate Q(state) for all actions

        Args:
            state_batch (torch.FloatTensor): tensor of size batch_size x state_dim if adapter.action_config = ACTIONS_IN_ACTION_SPACE
                        (torch.PackedSequence): of length batch_size, each entry actions x state_dim        = ACTIONS_IN_STATE_SPACE

        Returns:
            output: tensor of size batch_size x action_dim,  if adapter.action_config = ACTIONS_IN_ACTION_SPACE
                                   batch_sixe x max_actions,                          = ACTIONS_IN_STATE_SPACE
        """


        if self.actions_in_state_space:
            # forward 1 time per action
            padded_sequence, num_actions, mask = self.pad_and_mask(state_batch)
            y = self._forward(padded_sequence, mask.unsqueeze(2)) # batch x max_actions x action_dim = 1
            y = y.squeeze(2) # batch x max_actions
            # mask off outputs for inputs with less actions than max_actions
            y = torch.masked_fill(y, mask, float('-inf')) # batch x max_actions  (mask Q values)
            return NetworkOutput(logits=y, intent_logits=None)
        else:
            # forward once for all actions (normal behaviour)
            return NetworkOutput(logits=self._forward(state_batch), intent_logits=None)
