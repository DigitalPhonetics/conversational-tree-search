
from typing import List, Type, Union

import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

from gymnasium import Space, spaces
from stable_baselines3.dqn.policies import BasePolicy

from algorithm.dqn.layers import NoisyLinear
from encoding.state import StateDims



def add_weight_normalization(layers: List[nn.Module], input_dim: int, normalize: bool):
    if normalize:
        layers.append(nn.LayerNorm(normalized_shape=input_dim))


def mask_output(input: th.Tensor, mask: th.Tensor):
    if th.is_tensor(mask):
        return input * mask
    return input



class CustomQNetwork(BasePolicy):
    """
    Action-Value (Q-Value) network for DQN

    :param observation_space: Observation space
    :param action_space: Action space
    :param activation_fn: Activation function
    """

    action_space: spaces.Discrete

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Discrete,
        hidden_layer_sizes: List[int],
        state_dims: StateDims,
        normalization_layers: bool = False,
        dropout_rate: float = 0.0,
        activation_fn: Type[nn.Module] = nn.ReLU,
        q_value_clipping: int = 0.0,
        **kwargs
    ) -> None:
        super().__init__(
            observation_space,
            action_space,
            features_extractor=None,
            normalize_images=False,
        )

        self.q_value_clipping = q_value_clipping
        self.intent_prediction = False
        self.activation_fn = activation_fn
        self.dropout_rate = dropout_rate

        # network architecture

        # create layers
        layers = []
        current_input_dim = observation_space.shape[-1]
        self.actions_in_state_space = len(observation_space.shape) > 1
        self.output_dim = 1 if self.actions_in_state_space else action_space.n

        for layer_size in hidden_layer_sizes:
            layer = nn.Linear(current_input_dim, layer_size)
            layers.append(layer)
            add_weight_normalization(layers, layer_size, normalization_layers)
            activation = activation_fn()
            layers.append(activation)
            current_input_dim = layer_size
        # output layer
        output_layer = nn.Linear(current_input_dim, self.output_dim)
        layers.append(output_layer)

        self.q_net = nn.ModuleList(layers)
        print("NETWORK", self.q_net)

    def get_shared_module_names(self) -> List[str]:
        return ["q_net"]

    def forward(self, obs: th.Tensor, deterministic: bool = False) -> th.Tensor:
        """
        Predict the q-values.

        :param obs: Observation
        :return: The estimated Q-Value for each action.
        """
        mask = None # mask will be 0.0 for rows that should be masked, so we can multiply with them and eliminate gradient propagation for these
        q = obs
        if self.actions_in_state_space:
            # obs: batch x actions x state
            mask = (~(obs.abs().sum(-1) == 0.0)).float().unsqueeze(-1) # batch x actions x 1
        
        for layer_idx, layer in enumerate(self.q_net):
            q = layer(q)
            if (not deterministic) and layer_idx < len(self.q_net) - 1 and self.dropout_rate > 0.0:
                # no dropout in last layer, or if deterministic output is required
                q = F.dropout(q, p=self.dropout_rate)
            q = mask_output(q, mask)

        if self.actions_in_state_space:
            # reshape outputs
            q = q.view(-1, self.action_space.n)

        if self.q_value_clipping > 0.0:
            q = q.clip(min=None, max=self.q_value_clipping) # don't mask min because of "inf" masking
        return q

    def _predict(self, observation: th.Tensor, deterministic: bool = True, **kwargs) -> th.Tensor:
        q_values = self(observation, deterministic=deterministic)
        if self.actions_in_state_space:
            # obs: batch x actions x state
            mask = (observation.abs().sum(-1) == 0.0) # batch x actions
            q_values = th.masked_fill(q_values, mask, float('-inf'))

        # Greedy action
        action = q_values.argmax(dim=1).reshape(-1)
        return action


class CustomDuelingQNetwork(BasePolicy):
    action_space: spaces.Discrete

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Discrete,
        shared_layer_sizes: List[int],
        value_layer_sizes: List[int],
        advantage_layer_sizes: List[int],
        state_dims: StateDims,
        normalization_layers: bool = False,
        dropout_rate: float = 0.0,
        activation_fn: Type[nn.Module] = nn.ReLU,
        q_value_clipping: int = 0.0,
        **kwargs
    ) -> None:
        super().__init__(
            observation_space,
            action_space,
            features_extractor=None,
            normalize_images=False,
        )

        self.q_value_clipping = q_value_clipping
        self.intent_prediction = False
        self.activation_fn = activation_fn
        self.dropout_rate = dropout_rate
        self.state_dims = state_dims
        self.actions_in_state_space = len(observation_space.shape) > 1
        self.output_dim = 1 if self.actions_in_state_space else action_space.n
        self.state_input_size = state_dims.state_vector - state_dims.state_action_subvector

        # network architecture
        shared_layers = []
        action_inputs = []
        value_layers = []
        advantage_layers = []

        # create shared (and optionally, action input) layers
        shared_layer_dim = self.state_input_size
        action_layer_dim = state_dims.state_action_subvector if self.actions_in_state_space else 0
        for layer_size in shared_layer_sizes:
            shared_layer = nn.Linear(shared_layer_dim, layer_size)
            shared_layers.append(shared_layer)
            add_weight_normalization(shared_layers, layer_size, normalization_layers)
            activation = activation_fn()
            shared_layers.append(activation)
            shared_layer_dim = layer_size

            if self.actions_in_state_space:
                action_layer = nn.Linear(action_layer_dim, layer_size)
                action_inputs.append(action_layer)
                add_weight_normalization(action_inputs, layer_size, normalization_layers)
                activation = activation_fn()
                action_inputs.append(activation)
                action_layer_dim = layer_size

        # create value layers: (state_input_dim - action_input_dim) -> 1
        value_layer_dim = shared_layer_dim
        for layer_size in value_layer_sizes:
            value_layer = nn.Linear(value_layer_dim, layer_size)
            value_layers.append(value_layer)
            add_weight_normalization(value_layers, layer_size, normalization_layers)
            activation = activation_fn()
            value_layers.append(activation)
            value_layer_dim = layer_size
        value_output_layer = nn.Linear(value_layer_dim, 1) # value output layer
        value_layers.append(value_output_layer)

        # advantage layer: shared_layer_sizes[-1] -> actions
        advantage_layer_dim = shared_layer_dim + action_layer_dim
        for layer_size in advantage_layer_sizes:
            adv_layer = nn.Linear(advantage_layer_dim, layer_size)
            advantage_layers.append(adv_layer)
            add_weight_normalization(advantage_layers, layer_size, normalization_layers)
            activation = activation_fn()
            advantage_layers.append(activation)
            advantage_layer_dim = layer_size

        # advantage output layer: advantage_layer_dim -> actions
        output_layer = nn.Linear(advantage_layer_dim, self.output_dim)
        advantage_layers.append(output_layer)

        # connect layers to module
        self.shared_net = nn.ModuleList(shared_layers)
        self.action_input_net = nn.ModuleList(action_inputs)
        self.value_net = nn.ModuleList(value_layers)
        self.advantage_net = nn.ModuleList(advantage_layers)

    def get_shared_module_names(self) -> List[str]:
        return ["shared_net", "action_input_net"]

    def calculate_shared_stream(self, obs: th.Tensor, mask: Union[th.Tensor, None], deterministic: bool) -> th.Tensor:
        shared_stream = obs
        for layer in self.shared_net:
            shared_stream = layer(shared_stream)
            if (not deterministic) and self.dropout_rate > 0.0:
                shared_stream = F.dropout(shared_stream, p=self.dropout_rate)
            shared_stream = mask_output(shared_stream, mask)
        return shared_stream
    
    def calculate_value_stream(self, shared_stream: th.Tensor, mask: Union[th.Tensor, None], deterministic: bool) -> th.Tensor:
        value_stream = shared_stream
        for layer_idx, layer in enumerate(self.value_net):
            value_stream = layer(value_stream)
            if (not deterministic) and layer_idx < len(self.value_net) - 1 and self.dropout_rate > 0.0:
                # no dropout in last layer, or if deterministic output is required
                value_stream = F.dropout(value_stream, p=self.dropout_rate)
            value_stream = mask_output(value_stream, mask)
        return value_stream
    
    def calculate_action_stream(self, action_input: th.Tensor, mask: Union[th.Tensor, None], deterministic: bool):
        action_stream = action_input
        for layer in self.action_input_net:
            action_stream = layer(action_stream)
            if (not deterministic) and self.dropout_rate > 0.0:
                action_stream = F.dropout(action_stream, p=self.dropout_rate)
            action_stream = mask_output(action_stream, mask)
        return action_stream

    def calculate_advantage_stream(self, shared_stream: th.Tensor, action_stream: Union[th.Tensor, None], mask: Union[th.Tensor, None], deterministic: bool) -> th.Tensor:
        advantage_stream = shared_stream
        if self.actions_in_state_space:
            advantage_stream = th.cat((advantage_stream, action_stream), -1)
        for layer_idx, layer in enumerate(self.advantage_net):
            advantage_stream = layer(advantage_stream)
            if (not deterministic) and layer_idx < len(self.advantage_net) - 1 and self.dropout_rate > 0.0:
                # no dropout in last layer, or if deterministic output is required 
                advantage_stream = F.dropout(advantage_stream, p=self.dropout_rate)
            advantage_stream = mask_output(advantage_stream, mask)
        return advantage_stream

    def fuse_streams(self, value_stream: th.Tensor, advantage_stream: th.Tensor, num_actions: int) -> th.Tensor:
        if self.actions_in_state_space:
            # adapt dimensions
            value_stream = value_stream.squeeze(-1)
            advantage_stream = advantage_stream.squeeze(-1) # batch x actions x 1 -> batch x actions (because we only have 1 output here)
            
        # calculate advantage mean
        advantage_stream_mean = advantage_stream.sum(-1) / num_actions
        # combine value and advantage streams into Q values
        q = value_stream + advantage_stream - advantage_stream_mean.unsqueeze(-1) # batch x actions
        return q


    def forward(self, obs: th.Tensor, deterministic: bool = False) -> th.Tensor:
        """
        Predict the q-values.

        :param obs: Observation
        :return: The estimated Q-Value for each action.
        """
        # preprocess inputs
        mask = None # mask will be 0.0 for rows that should be masked, so we can multiply with them and eliminate gradient propagation for these
        num_actions = self.state_dims.num_actions
        action_stream = None
        state_input = obs
        if self.actions_in_state_space:
            # obs: batch x actions x state
            mask = (~(obs.abs().sum(-1) == 0.0)).float() # batch x actions
            num_actions = mask.sum(-1) # batch (contains number of actions per batch)
            mask = mask.unsqueeze(-1) # batch x actions x 1

            state_input = obs[:, :, :self.state_input_size] # state input portion of input
            action_input = obs[:, :, self.state_input_size:] # action input portion of input

            action_stream = self.calculate_action_stream(action_input, mask, deterministic)
            
        # calculate streams
        shared_stream = self.calculate_shared_stream(state_input, mask, deterministic)
        value_stream = self.calculate_value_stream(shared_stream, mask, deterministic)
        advantage_stream = self.calculate_advantage_stream(shared_stream, action_stream, mask, deterministic)

        # Fuse streams into q-values
        q = self.fuse_streams(value_stream, advantage_stream, num_actions)

        if self.q_value_clipping > 0.0:
            q = q.clip(min=None, max=self.q_value_clipping) # don't mask min because of "inf" masking
        return q
       
    def _predict(self, observation: th.Tensor, deterministic: bool = True, **kwargs) -> th.Tensor:
        q_values = self(observation, deterministic=deterministic)
        if self.actions_in_state_space:
            # obs: batch x actions x state
            mask = (observation.abs().sum(-1) == 0.0) # batch x actions
            q_values = th.masked_fill(q_values, mask, float('-inf'))

        # Greedy action
        action = q_values.argmax(dim=-1).reshape(-1)
        return action
    
 
class CustomDuelingQNetworkWithIntentPrediction(CustomDuelingQNetwork):
    action_space: spaces.Discrete
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Discrete,
        shared_layer_sizes: List[int],
        value_layer_sizes: List[int],
        advantage_layer_sizes: List[int],
        state_dims: StateDims,
        normalization_layers: bool = False,
        dropout_rate: float = 0.0,
        activation_fn: Type[nn.Module] = nn.ReLU,
        intent_loss_weight: float = 0.0,
        q_value_clipping: int = 0.0,
        **kwargs
    ) -> None:
        assert intent_loss_weight > 0.0
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            shared_layer_sizes=shared_layer_sizes,
            value_layer_sizes=value_layer_sizes,
            advantage_layer_sizes=advantage_layer_sizes,
            state_dims=state_dims,
            normalization_layers=normalization_layers,
            dropout_rate=dropout_rate,
            activation_fn=activation_fn,
            q_value_clipping=q_value_clipping
        )

        # intent prediction head
        self.intent_loss_weight = intent_loss_weight
        self.intent_prediction = True
        self.intent_head = nn.Sequential(
            nn.Linear(shared_layer_sizes[-1], 256),
            nn.SELU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(256, 1)
        )

    def forward(self, obs: th.Tensor, deterministic: bool = False) -> th.Tensor:
        """
        Predict the q-values.

        :param obs: Observation
        :return: The estimated Q-Value for each action.
        """
        # preprocess inputs
        mask = None # mask will be 0.0 for rows that should be masked, so we can multiply with them and eliminate gradient propagation for these
        num_actions = self.state_dims.num_actions
        action_stream = None
        state_input = obs
        if self.actions_in_state_space:
            # obs: batch x actions x state
            mask = (~(obs.abs().sum(-1) == 0.0)).float() # batch x actions
            num_actions = mask.sum(-1) # batch (contains number of actions per batch)
            mask = mask.unsqueeze(-1) # batch x actions x 1

            state_input = obs[:, :, :self.state_input_size] # state input portion of input
            action_input = obs[:, :, self.state_input_size:] # action input portion of input

            action_stream = self.calculate_action_stream(action_input, mask, deterministic)
            
        # calculate streams
        shared_stream = self.calculate_shared_stream(state_input, mask, deterministic)
        value_stream = self.calculate_value_stream(shared_stream, mask, deterministic)
        advantage_stream = self.calculate_advantage_stream(shared_stream, action_stream, mask, deterministic)

        # intent prediction
        intent_logits = shared_stream.max(dim=1)[0] if self.actions_in_state_space else shared_stream  # batch x shared_dim
        intent_logits = self.intent_head(intent_logits).squeeze(-1) # batch x 1 -> batch
        
        # Fuse streams into q-values
        q = self.fuse_streams(value_stream, advantage_stream, num_actions)

        if self.q_value_clipping > 0.0:
            q = q.clip(min=None, max=self.q_value_clipping) # don't mask min because of "inf" masking

        return q, intent_logits
       
    def _predict(self, observation: th.Tensor, deterministic: bool = True, **kwargs) -> th.Tensor:
        q_values, intent_logits = self(observation, deterministic)
        if self.actions_in_state_space:
            # obs: batch x actions x state
            mask = (observation.abs().sum(-1) == 0.0) # batch x actions
            q_values = th.masked_fill(q_values, mask, float('-inf'))

        # Greedy action
        action = q_values.argmax(dim=1).reshape(-1)
        return action, (th.sigmoid(intent_logits) >= 0.5)




 
class CustomDuelingMultiheadQNetwork(BasePolicy):
    """ 
    Implementation with multi-head outputs (K heads for K differently parameterized Q-value functions)
    Based on: Deep Exploration via Bootstrapped DQN (Osband et. al.)

    This can be seen as a generalization of CustomDuelingQNetwork (for num_prediction_heads = 1)
    """
    action_space: spaces.Discrete

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Discrete,
        shared_layer_sizes: List[int],
        value_layer_sizes: List[int],
        advantage_layer_sizes: List[int],
        state_dims: StateDims,
        normalization_layers: bool = False,
        dropout_rate: float = 0.0,
        activation_fn: Type[nn.Module] = nn.ReLU,
        q_value_clipping: int = 0.0,
        num_prediction_heads: int = 10,
        bernoulli_p_train_heads: float = 1.0,
        **kwargs
    ) -> None:
        
        super().__init__(
            observation_space,
            action_space,
            features_extractor=None,
            normalize_images=False,
        )

        assert num_prediction_heads > 0, "need at least 1 prediction head"
        self.q_value_clipping = q_value_clipping
        self.intent_prediction = False
        self.activation_fn = activation_fn
        self.dropout_rate = dropout_rate
        self.state_dims = state_dims
        self.actions_in_state_space = len(observation_space.shape) > 1
        self.output_dim = 1 if self.actions_in_state_space else action_space.n
        self.state_input_size = state_dims.state_vector - state_dims.state_action_subvector
        self.num_prediction_heads = num_prediction_heads

        # network architecture
        shared_layers = []
        action_inputs = []
        value_layers = []
        value_output_heads = []
        advantage_layers = []
        advantage_output_heads = [] # num_prediction_heads many output functions

        # create shared (and optionally, action input) layers
        shared_layer_dim = self.state_input_size
        action_layer_dim = state_dims.state_action_subvector if self.actions_in_state_space else 0
        for layer_size in shared_layer_sizes:
            shared_layer = nn.Linear(shared_layer_dim, layer_size)
            shared_layers.append(shared_layer)
            add_weight_normalization(shared_layers, layer_size, normalization_layers)
            activation = activation_fn()
            shared_layers.append(activation)
            shared_layer_dim = layer_size

            if self.actions_in_state_space:
                action_layer = nn.Linear(action_layer_dim, layer_size)
                action_inputs.append(action_layer)
                add_weight_normalization(action_inputs, layer_size, normalization_layers)
                activation = activation_fn()
                action_inputs.append(activation)
                action_layer_dim = layer_size

        # create value layers: (state_input_dim - action_input_dim) -> 1
        value_layer_dim = shared_layer_dim
        for layer_size in value_layer_sizes:
            value_layer = nn.Linear(value_layer_dim, layer_size)
            value_layers.append(value_layer)
            add_weight_normalization(value_layers, layer_size, normalization_layers)
            activation = activation_fn()
            value_layers.append(activation)
            value_layer_dim = layer_size

        # value output layer: value_layer_sizes[-1] -> 1
        for head_idx in range(num_prediction_heads):
            value_output_heads.append(nn.Linear(value_layer_dim, 1))

        # advantage layer: shared_layer_sizes[-1] -> actions
        advantage_layer_dim = shared_layer_dim + action_layer_dim
        for layer_size in advantage_layer_sizes:
            adv_layer = nn.Linear(advantage_layer_dim, layer_size)
            advantage_layers.append(adv_layer)
            add_weight_normalization(advantage_layers, layer_size, normalization_layers)
            activation = activation_fn()
            advantage_layers.append(activation)
            advantage_layer_dim = layer_size

        # advantage output layer: advantage_layer_dim -> actions
        for head_idx in range(num_prediction_heads):
            advantage_output_heads.append(nn.Linear(advantage_layer_dim, self.output_dim))

        # connect layers to module
        self.shared_net = nn.ModuleList(shared_layers)
        self.action_input_net = nn.ModuleList(action_inputs)
        self.value_net = nn.ModuleList(value_layers)
        self.value_heads_net = nn.ModuleList(value_output_heads)
        self.advantage_net = nn.ModuleList(advantage_layers)
        self.advantage_heads_net = nn.ModuleList(advantage_output_heads)

    def get_shared_module_names(self) -> List[str]:
        return ["shared_net", "action_input_net", "value_net", "advantage_net"]

    def calculate_shared_stream(self, obs: th.Tensor, mask: Union[th.Tensor, None], deterministic: bool) -> th.Tensor:
        shared_stream = obs
        for layer in self.shared_net:
            shared_stream = layer(shared_stream)
            if (not deterministic) and self.dropout_rate > 0.0:
                shared_stream = F.dropout(shared_stream, p=self.dropout_rate)
            shared_stream = mask_output(shared_stream, mask)
        return shared_stream
    
    def calculate_value_stream(self, shared_stream: th.Tensor, mask: Union[th.Tensor, None], deterministic: bool) -> th.Tensor:
        value_stream = shared_stream
        for layer_idx, layer in enumerate(self.value_net):
            value_stream = layer(value_stream)
            if (not deterministic) and layer_idx < len(self.value_net) - 1 and self.dropout_rate > 0.0:
                # no dropout in last layer, or if deterministic output is required
                value_stream = F.dropout(value_stream, p=self.dropout_rate)
            value_stream = mask_output(value_stream, mask)
        return value_stream

    def calculate_value_heads_stream(self, value_stream: th.Tensor) -> th.Tensor:
        output = []
        # if self.actions_in_state_space:
        #     value_stream = value_stream.squeeze(-1) # batch x value_dim x 1 -> batch x value_dim
        for output_head in self.value_heads_net:
            output.append(output_head(value_stream).squeeze(-1).unsqueeze(1))  # batch x value_dim -> batch x num_prediction_heads x 1
        return th.cat(output, dim=1) # batch x num_prediction_heads

    def calculate_action_stream(self, action_input: th.Tensor, mask: Union[th.Tensor, None], deterministic: bool):
        action_stream = action_input
        for layer in self.action_input_net:
            action_stream = layer(action_stream)
            if (not deterministic) and self.dropout_rate > 0.0:
                action_stream = F.dropout(action_stream, p=self.dropout_rate)
            action_stream = mask_output(action_stream, mask)
        return action_stream

    def calculate_advantage_stream(self, shared_stream: th.Tensor, action_stream: Union[th.Tensor, None], mask: Union[th.Tensor, None], deterministic: bool) -> th.Tensor:
        advantage_stream = shared_stream
        if self.actions_in_state_space:
            advantage_stream = th.cat((advantage_stream, action_stream), -1)
        for layer_idx, layer in enumerate(self.advantage_net):
            advantage_stream = layer(advantage_stream)
            if (not deterministic) and self.dropout_rate > 0.0:
                # no dropout in last layer, or if deterministic output is required 
                advantage_stream = F.dropout(advantage_stream, p=self.dropout_rate)
            advantage_stream = mask_output(advantage_stream, mask)
        return advantage_stream
    
    def calculate_advantage_heads_stream(self, advantage_stream: th.Tensor) -> th.Tensor:
        # calculate output for each advantage head
        # no dropout here - this is the last layer
        output = []
        if self.actions_in_state_space:
            advantage_stream = advantage_stream.squeeze(-1) # batch x advantage_dim x 1 -> batch x actions (because we only have 1 output here)
        for output_head in self.advantage_heads_net:
            output.append(output_head(advantage_stream).squeeze(-1).unsqueeze(1)) # batch x advantage_dim -> batch x num_prediction_heads x action_dim
        return th.cat(output, dim=1) # batch x num_prediction_heads x advantage_dim 


    def fuse_streams(self, value_heads_stream: th.Tensor, advantage_heads_stream: th.Tensor, num_actions: int) -> th.Tensor:
        # calculate advantage mean
        advantage_stream_mean = advantage_heads_stream.sum(-1) / num_actions.unsqueeze(-1) # batch x num_prediction_heads

        # combine value and advantage streams into Q values
        #   batch x num_prediction_heads  + batch x num_prediction_heads x actions - batch x num_prediction_heads
        q = value_heads_stream + advantage_heads_stream - advantage_stream_mean.unsqueeze(-1) # batch x num_prediction_heads x actions
        return q 


    def forward(self, obs: th.Tensor, deterministic: bool = False) -> th.Tensor:
        """
        Predict the q-values.

        :param obs: Observation
        :return: The estimated Q-Value for each action.
        """
        # preprocess inputs
        mask = None # mask will be 0.0 for rows that should be masked, so we can multiply with them and eliminate gradient propagation for these
        num_actions = self.state_dims.num_actions
        action_stream = None
        state_input = obs
        if self.actions_in_state_space:
            # obs: batch x actions x state
            mask = (~(obs.abs().sum(-1) == 0.0)).float() # batch x actions
            num_actions = mask.sum(-1) # batch (contains number of actions per batch)
            mask = mask.unsqueeze(-1) # batch x actions x 1

            state_input = obs[:, :, :self.state_input_size] # state input portion of input
            action_input = obs[:, :, self.state_input_size:] # action input portion of input

            action_stream = self.calculate_action_stream(action_input, mask, deterministic)
            
        # calculate streams
        shared_stream = self.calculate_shared_stream(state_input, mask, deterministic)
        value_stream = self.calculate_value_stream(shared_stream, mask, deterministic) # batch x value_dim (x 1, if actions_in_state_space)
        value_heads_stream = self.calculate_value_heads_stream(value_stream) # batch x num_prediction_heads
        advantage_stream = self.calculate_advantage_stream(shared_stream, action_stream, mask, deterministic) # batch x advantage_dim (x 1, if actions_in_state_space)
        advantage_heads_stream = self.calculate_advantage_heads_stream(advantage_stream) # batch x num_prediction_heads x actions

        # Fuse streams into q-values
        q = self.fuse_streams(value_heads_stream, advantage_heads_stream, num_actions) # batch x num_prediction_heads x actions

        if self.q_value_clipping > 0.0:
            q = q.clip(min=None, max=self.q_value_clipping) # don't mask min because of "inf" masking
        return q # batch x num_prediction_heads x actions
       
    def _predict(self, observation: th.Tensor, deterministic: bool = True, prediction_head_mask: th.Tensor = None) -> th.Tensor:
        # if deterministic, do majority vote from all heads
        # if non-deterministric, follow prediction head mask to sample greedily from exploration heads

        q_values = self(observation, deterministic) # batch x num_prediction_heads x actions

        if self.actions_in_state_space:
            # obs: batch x actions x state
            mask = (observation.abs().sum(-1) == 0.0).unsqueeze(1) # batch x 1 x actions (unsqueeze prediction head dimension: mask same outputs for each prediction head)
            q_values = th.masked_fill(q_values, mask, float('-inf'))
        
        if deterministic:
            # majority vote across all prediction heads
            max_q_values, _ = q_values.max(dim=2) # batch x num_prediction_heads
            q_values = (q_values == max_q_values.view(-1, self.num_prediction_heads, 1)) # batch x num_prediction_heads x actions
            action = q_values.sum(1).argmax(-1).reshape(-1) # majority vote per prediction heads for each batch item: batch x num_prediction_heads x actions -> batch x actions -> batch
        else:
            # greedy action from current exploration head
            action = q_values[range(observation.size(0)),prediction_head_mask,:].argmax(dim=-1).reshape(-1) # batch x action -> batch 
        
        return action
    
    @th.no_grad()
    def uncertainty(self, obs: th.Tensor) -> th.Tensor:
        # softmax to get policy distribution
        q_values = self.forward(obs.to(self.device), deterministic=False) # batch x num_prediction_heads x actions
        if self.actions_in_state_space:
            # obs: batch x actions x state
            mask = (obs.abs().sum(-1) == 0.0).unsqueeze(1).to(self.device) # batch x actions
            q_values = th.masked_fill(q_values, mask, float('-inf'))
        q_probs = q_values.softmax(dim=-1) # softmax across actions (per head)

        head_normalization = 1. / self.num_prediction_heads

        h1 = Categorical(head_normalization * q_probs.sum(1)).entropy() # batch x actions -> batch
        h2 = th.sum(Categorical(q_probs).entropy(), dim=-1) # batch x num_heads x actions -> batch x num_heads -> batch
        
        div = h1 - head_normalization * h2 # batch
        return th.clamp(div, min=0)


def shannon_entropy(probs: th.Tensor) -> th.Tensor:
    """ Taken across last dimension.
    Args:
        probs: probability distribution (has to sum to 1 across last dimension)
    """
    entropy = - probs * th.log(probs) # batch x ... x probs
    entropy = th.nansum(entropy, -1)     # batch x ...
    return entropy



class CustomDuelingMultiheadQNetworkWithIntentPrediction(CustomDuelingMultiheadQNetwork):
    action_space: spaces.Discrete

    def __init__(self, observation_space: spaces.Space, action_space: spaces.Discrete, 
                 shared_layer_sizes: List[int], value_layer_sizes: List[int], advantage_layer_sizes: List[int], 
                 state_dims: StateDims,
                 normalization_layers: bool = False, dropout_rate: float = 0, activation_fn: nn.Module = nn.ReLU, 
                 intent_loss_weight: float = 0.0, q_value_clipping: int = 0,
                 num_prediction_heads: int = 10, bernoulli_p_train_heads: float = 1.0,
                 **kwargs) -> None:
        
        assert intent_loss_weight > 0.0

        super().__init__(observation_space, action_space, shared_layer_sizes, value_layer_sizes, advantage_layer_sizes, 
                         state_dims, normalization_layers, dropout_rate, activation_fn, q_value_clipping, num_prediction_heads,
                         **kwargs)
        
        # intent prediction head
        self.intent_loss_weight = intent_loss_weight
        self.intent_prediction = True
        self.intent_head = nn.Sequential(
            nn.Linear(shared_layer_sizes[-1], 256),
            nn.SELU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(256, 1)
        )

    def forward(self, obs: th.Tensor, deterministic: bool = False) -> th.Tensor:
        """
        Predict the q-values.

        :param obs: Observation
        :return: The estimated Q-Value for each action.
        """
        # preprocess inputs
        mask = None # mask will be 0.0 for rows that should be masked, so we can multiply with them and eliminate gradient propagation for these
        num_actions = self.state_dims.num_actions
        action_stream = None
        state_input = obs
        if self.actions_in_state_space:
            # obs: batch x actions x state
            mask = (~(obs.abs().sum(-1) == 0.0)).float() # batch x actions
            num_actions = mask.sum(-1) # batch (contains number of actions per batch)
            mask = mask.unsqueeze(-1) # batch x actions x 1

            state_input = obs[:, :, :self.state_input_size] # state input portion of input
            action_input = obs[:, :, self.state_input_size:] # action input portion of input

            action_stream = self.calculate_action_stream(action_input, mask, deterministic)
            
        # calculate streams
        shared_stream = self.calculate_shared_stream(state_input, mask, deterministic)
        value_stream = self.calculate_value_stream(shared_stream, mask, deterministic) # batch x value_dim (x 1, if actions_in_state_space)
        value_heads_stream = self.calculate_value_heads_stream(value_stream) # batch x num_prediction_heads
        advantage_stream = self.calculate_advantage_stream(shared_stream, action_stream, mask, deterministic) # batch x advantage_dim (x 1, if actions_in_state_space)
        advantage_heads_stream = self.calculate_advantage_heads_stream(advantage_stream) # batch x num_prediction_heads x actions

         # intent prediction
        intent_logits = shared_stream.max(dim=1)[0] if self.actions_in_state_space else shared_stream  # batch x shared_dim
        intent_logits = self.intent_head(intent_logits).squeeze(-1) # batch x 1 -> batch

        # Fuse streams into q-values
        q = self.fuse_streams(value_heads_stream, advantage_heads_stream, num_actions) # batch x num_prediction_heads x actions

        if self.q_value_clipping > 0.0:
            q = q.clip(min=None, max=self.q_value_clipping) # don't mask min because of "inf" masking
        return q, intent_logits # batch x num_prediction_heads x actions
       

    def _predict(self, observation: th.Tensor, deterministic: bool = True, prediction_head_mask: th.Tensor = None) -> th.Tensor:
        # if deterministic, do majority vote from all heads
        # if non-deterministric, follow prediction head mask to sample greedily from exploration heads

        q_values, intent_logits = self(observation, deterministic) # batch x num_prediction_heads x actions
        if self.actions_in_state_space:
            # obs: batch x actions x state
            mask = (observation.abs().sum(-1) == 0.0).unsqueeze(1) # batch x 1 x actions (unsqueeze prediction head dimension: mask same outputs for each prediction head)
            q_values = th.masked_fill(q_values, mask, float('-inf'))
        
        if deterministic:
            # majority vote across all prediction heads
            max_q_values, _ = q_values.max(dim=2) # batch x num_prediction_heads
            q_values = (q_values == max_q_values.view(-1, self.num_prediction_heads, 1)) # batch x num_prediction_heads x actions
            action = q_values.sum(1).argmax(-1).reshape(-1) # majority vote per prediction heads for each batch item: batch x num_prediction_heads x actions -> batch x actions -> batch
        else:
            # greedy action from current exploration head
            action = q_values[range(observation.size(0)),prediction_head_mask,:].argmax(dim=-1).reshape(-1) # batch x action -> batch 
        
        return action, (th.sigmoid(intent_logits) >= 0.5)

    @th.no_grad() 
    def uncertainty(self, obs: th.Tensor) -> th.Tensor:
        # softmax to get policy distribution
        q_values, _ = self.forward(obs.to(self.device), deterministic=True) # batch x num_prediction_heads x actions
        if self.actions_in_state_space:
            # obs: batch x actions x state
            mask = (obs.abs().sum(-1) == 0.0).unsqueeze(1).to(self.device) # batch x actions
            q_values = th.masked_fill(q_values, mask, float('-inf'))
        q_probs = q_values.softmax(dim=-1) # softmax across actions (per head)

        head_normalization = 1. / self.num_prediction_heads

        h1 = Categorical(head_normalization * q_probs.sum(1)).entropy() # batch x actions -> batch
        h2 = th.sum(Categorical(q_probs).entropy(), dim=-1) # batch x num_heads x actions -> batch x num_heads -> batch

        div = h1 - head_normalization * h2 # batch
        return th.clamp(div, min=0) # ensure positive definiteness (numerical instabilities)


class CustomDuelingNoisyQNetwork(BasePolicy):
    action_space: spaces.Discrete

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Discrete,
        shared_layer_sizes: List[int],
        value_layer_sizes: List[int],
        advantage_layer_sizes: List[int],
        state_dims: StateDims,
        normalization_layers: bool = False,
        dropout_rate: float = 0.0,
        activation_fn: Type[nn.Module] = nn.ReLU,
        q_value_clipping: int = 0.0,
        noise_std_init: float = 0.1,
        **kwargs
    ) -> None:
        super().__init__(
            observation_space,
            action_space,
            features_extractor=None,
            normalize_images=False,
        )
        assert 0 < noise_std_init

        self.q_value_clipping = q_value_clipping
        self.intent_prediction = False
        self.activation_fn = activation_fn
        self.dropout_rate = dropout_rate
        self.state_dims = state_dims
        self.actions_in_state_space = len(observation_space.shape) > 1
        self.output_dim = 1 if self.actions_in_state_space else action_space.n
        self.state_input_size = state_dims.state_vector - state_dims.state_action_subvector
        self.noise_std_init = noise_std_init

        # network architecture
        shared_layers = []
        action_inputs = []
        value_layers = []
        advantage_layers = []

        # create shared (and optionally, action input) layers
        shared_layer_dim = self.state_input_size
        action_layer_dim = state_dims.state_action_subvector if self.actions_in_state_space else 0
        for layer_size in shared_layer_sizes:
            shared_layer = nn.Linear(shared_layer_dim, layer_size)
            shared_layers.append(shared_layer)
            add_weight_normalization(shared_layers, layer_size, normalization_layers)
            activation = activation_fn()
            shared_layers.append(activation)
            shared_layer_dim = layer_size

            if self.actions_in_state_space:
                action_layer = nn.Linear(action_layer_dim, layer_size)
                action_inputs.append(action_layer)
                add_weight_normalization(action_inputs, layer_size, normalization_layers)
                activation = activation_fn()
                action_inputs.append(activation)
                action_layer_dim = layer_size

        # create value layers: (state_input_dim - action_input_dim) -> 1
        value_layer_dim = shared_layer_dim
        for layer_size in value_layer_sizes:
            value_layer = nn.Linear(value_layer_dim, layer_size)
            value_layers.append(value_layer)
            add_weight_normalization(value_layers, layer_size, normalization_layers)
            activation = activation_fn()
            value_layers.append(activation)
            value_layer_dim = layer_size
        value_output_layer = nn.Linear(value_layer_dim, 1) # value output layer
        value_layers.append(value_output_layer)

        # advantage layer: shared_layer_sizes[-1] -> actions
        advantage_layer_dim = shared_layer_dim + action_layer_dim
        for layer_size in advantage_layer_sizes:
            adv_layer = nn.Linear(advantage_layer_dim, layer_size)
            advantage_layers.append(adv_layer)
            add_weight_normalization(advantage_layers, layer_size, normalization_layers)
            activation = activation_fn()
            advantage_layers.append(activation)
            advantage_layer_dim = layer_size

        # connect layers to module
        self.shared_net = nn.ModuleList(shared_layers)
        self.action_input_net = nn.ModuleList(action_inputs)
        self.value_net = nn.ModuleList(value_layers)
        self.advantage_net = nn.ModuleList(advantage_layers)

        # advantage output layer: advantage_layer_dim -> actions
        self.advantage_output_layer = NoisyLinear(advantage_layer_dim, self.output_dim, std_init=noise_std_init)

    def get_shared_module_names(self) -> List[str]:
        return ["shared_net", "action_input_net"]

    def calculate_shared_stream(self, obs: th.Tensor, mask: Union[th.Tensor, None], deterministic: bool) -> th.Tensor:
        shared_stream = obs
        for layer in self.shared_net:
            shared_stream = layer(shared_stream)
            if (not deterministic) and self.dropout_rate > 0.0:
                shared_stream = F.dropout(shared_stream, p=self.dropout_rate)
            shared_stream = mask_output(shared_stream, mask)
        return shared_stream
    
    def calculate_value_stream(self, shared_stream: th.Tensor, mask: Union[th.Tensor, None], deterministic: bool) -> th.Tensor:
        value_stream = shared_stream
        for layer_idx, layer in enumerate(self.value_net):
            value_stream = layer(value_stream)
            if (not deterministic) and layer_idx < len(self.value_net) - 1 and self.dropout_rate > 0.0:
                # no dropout in last layer, or if deterministic output is required
                value_stream = F.dropout(value_stream, p=self.dropout_rate)
            value_stream = mask_output(value_stream, mask)
        return value_stream
    
    def calculate_action_stream(self, action_input: th.Tensor, mask: Union[th.Tensor, None], deterministic: bool):
        action_stream = action_input
        for layer in self.action_input_net:
            action_stream = layer(action_stream)
            if (not deterministic) and self.dropout_rate > 0.0:
                action_stream = F.dropout(action_stream, p=self.dropout_rate)
            action_stream = mask_output(action_stream, mask)
        return action_stream

    def calculate_advantage_stream(self, shared_stream: th.Tensor, action_stream: Union[th.Tensor, None], mask: Union[th.Tensor, None], deterministic: bool) -> th.Tensor:
        advantage_stream = shared_stream
        if self.actions_in_state_space:
            advantage_stream = th.cat((advantage_stream, action_stream), -1)
        for layer_idx, layer in enumerate(self.advantage_net):
            advantage_stream = layer(advantage_stream)
            if (not deterministic) and layer_idx < len(self.advantage_net) - 1 and self.dropout_rate > 0.0:
                # no dropout in last layer, or if deterministic output is required 
                advantage_stream = F.dropout(advantage_stream, p=self.dropout_rate)
            advantage_stream = mask_output(advantage_stream, mask)
        return advantage_stream

    def fuse_streams(self, value_stream: th.Tensor, advantage_stream: th.Tensor, num_actions: int) -> th.Tensor:
        if self.actions_in_state_space:
            # adapt dimensions
            value_stream = value_stream.squeeze(-1)
            advantage_stream = advantage_stream.squeeze(-1) # batch x actions x 1 -> batch x actions (because we only have 1 output here)
            
        # calculate advantage mean
        advantage_stream_mean = advantage_stream.sum(-1) / num_actions
        # combine value and advantage streams into Q values
        q = value_stream + advantage_stream - advantage_stream_mean.unsqueeze(-1) # batch x actions
        return q

    def _forward_until_output_layer(self, obs: th.Tensor, deterministic: bool = False) -> th.Tensor:
        # do full forward pass without output layer (required for uncertainty estimate)
         # preprocess inputs
        mask = None # mask will be 0.0 for rows that should be masked, so we can multiply with them and eliminate gradient propagation for these
        num_actions = self.state_dims.num_actions
        action_stream = None
        state_input = obs
        if self.actions_in_state_space:
            # obs: batch x actions x state
            mask = (~(obs.abs().sum(-1) == 0.0)).float() # batch x actions
            num_actions = mask.sum(-1) # batch (contains number of actions per batch)
            mask = mask.unsqueeze(-1) # batch x actions x 1

            state_input = obs[:, :, :self.state_input_size] # state input portion of input
            action_input = obs[:, :, self.state_input_size:] # action input portion of input

            action_stream = self.calculate_action_stream(action_input, mask, deterministic)
            
        # calculate streams
        shared_stream = self.calculate_shared_stream(state_input, mask, deterministic)
        value_stream = self.calculate_value_stream(shared_stream, mask, deterministic)
        advantage_stream = self.calculate_advantage_stream(shared_stream, action_stream, mask, deterministic)
        return num_actions, shared_stream, value_stream, advantage_stream # batch x stream_dim 

    def forward(self, obs: th.Tensor, deterministic: bool = False) -> th.Tensor:
        """
        Predict the q-values.

        :param obs: Observation
        :return: The estimated Q-Value for each action.
        """
        num_actions, shared_stream, value_stream, advantage_stream = self._forward_until_output_layer(obs=obs, deterministic=deterministic)
        advantage_stream = self.advantage_output_layer(advantage_stream)

        # Fuse streams into q-values
        q = self.fuse_streams(value_stream, advantage_stream, num_actions)

        if self.q_value_clipping > 0.0:
            q = q.clip(min=None, max=self.q_value_clipping) # don't mask min because of "inf" masking
        return q
       
    def _predict(self, observation: th.Tensor, deterministic: bool = True, **kwargs) -> th.Tensor:
        q_values = self(observation, deterministic=deterministic)
        if self.actions_in_state_space:
            # obs: batch x actions x state
            mask = (observation.abs().sum(-1) == 0.0) # batch x actions
            q_values = th.masked_fill(q_values, mask, float('-inf'))

        if not deterministic:
            # sample new noise variables
            self.advantage_output_layer.reset_noise()
        
        # Greedy action
        action = q_values.argmax(dim=-1).reshape(-1)
        return action
    
    @th.no_grad()
    def uncertainty(self, obs: th.Tensor) -> th.Tensor:
        batch_size = obs.size(0)

        # predict greedy actions
        best_actions = self._predict(observation=obs.to(self.device), deterministic=True) # batch

        # calculate variance between actions
        _, _, _, advantage_stream = self._forward_until_output_layer(obs=obs, deterministic=False) # batch x stream_dim
        # TODO masking?
        uncertainties = []
        for batch_idx in range(batch_size):
            phi_s = advantage_stream[batch_idx,:]
            sigma = th.diag(self.advantage_output_layer.weight_sigma[best_actions[batch_idx], :].pow(2)) # adv_stream -> adv_stream x adv_stream
            bias_sigma = self.advantage_output_layer.bias_sigma[best_actions[batch_idx]].pow(2) # dim: 1
            uncertainties.append(th.bilinear(phi_s, phi_s, sigma, bias=bias_sigma)) # dim: 1
        uncertainties = th.stack(uncertainties)
        return uncertainties
        
        
    
class CustomDuelingNoisyQNetworkWithIntentPrediction(CustomDuelingNoisyQNetwork):
    action_space: spaces.Discrete
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Discrete,
        shared_layer_sizes: List[int],
        value_layer_sizes: List[int],
        advantage_layer_sizes: List[int],
        state_dims: StateDims,
        normalization_layers: bool = False,
        dropout_rate: float = 0.0,
        activation_fn: Type[nn.Module] = nn.ReLU,
        intent_loss_weight: float = 0.0,
        q_value_clipping: int = 0.0,
        noise_std_init: float = 0.0,
        **kwargs
    ) -> None:
        assert intent_loss_weight > 0.0
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            shared_layer_sizes=shared_layer_sizes,
            value_layer_sizes=value_layer_sizes,
            advantage_layer_sizes=advantage_layer_sizes,
            state_dims=state_dims,
            normalization_layers=normalization_layers,
            dropout_rate=dropout_rate,
            activation_fn=activation_fn,
            q_value_clipping=q_value_clipping,
            noise_std_init=noise_std_init
        )

        # intent prediction head
        self.intent_loss_weight = intent_loss_weight
        self.intent_prediction = True
        self.intent_head = nn.Sequential(
            nn.Linear(shared_layer_sizes[-1], 256),
            nn.SELU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(256, 1)
        )

    def forward(self, obs: th.Tensor, deterministic: bool = False) -> th.Tensor:
        """
        Predict the q-values.

        :param obs: Observation
        :return: The estimated Q-Value for each action.
        """
        # preprocess inputs
        num_actions, shared_stream, value_stream, advantage_stream = self._forward_until_output_layer(obs=obs, deterministic=deterministic)
        advantage_stream = self.advantage_output_layer(advantage_stream)

        # intent prediction
        intent_logits = shared_stream.max(dim=1)[0] if self.actions_in_state_space else shared_stream  # batch x shared_dim
        intent_logits = self.intent_head(intent_logits).squeeze(-1) # batch x 1 -> batch
        
        # Fuse streams into q-values
        q = self.fuse_streams(value_stream, advantage_stream, num_actions)

        if self.q_value_clipping > 0.0:
            q = q.clip(min=None, max=self.q_value_clipping) # don't mask min because of "inf" masking

        return q, intent_logits
       
    def _predict(self, observation: th.Tensor, deterministic: bool = True, **kwargs) -> th.Tensor:
        q_values, intent_logits = self(observation, deterministic)
        if self.actions_in_state_space:
            # obs: batch x actions x state
            mask = (observation.abs().sum(-1) == 0.0) # batch x actions
            q_values = th.masked_fill(q_values, mask, float('-inf'))

        if not deterministic:
            # sample new noise variables
            self.advantage_output_layer.reset_noise()

        # Greedy action
        action = q_values.argmax(dim=1).reshape(-1)
        return action, (th.sigmoid(intent_logits) >= 0.5)

    @th.no_grad()
    def uncertainty(self, obs: th.Tensor) -> th.Tensor:
        batch_size = obs.size(0)

        # predict greedy actions
        best_actions, _ = self._predict(observation=obs.to(self.device), deterministic=True)

        # calculate variance between actions
        _, _, _, advantage_stream = self._forward_until_output_layer(obs=obs, deterministic=False) # batch x stream_dim
        # TODO masking?
        uncertainties = []
        for batch_idx in range(batch_size):
            phi_s = advantage_stream[batch_idx,:]
            sigma = th.diag(self.advantage_output_layer.weight_sigma[best_actions[batch_idx], :].pow(2)) # adv_stream -> adv_stream x adv_stream
            bias_sigma = self.advantage_output_layer.bias_sigma[best_actions[batch_idx]].pow(2) # dim: 1
            uncertainties.append(th.bilinear(phi_s, phi_s, sigma, bias=bias_sigma)) # dim: 1
        uncertainties = th.stack(uncertainties)
        return uncertainties