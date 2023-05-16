
from copy import deepcopy
from typing import List, Tuple, Union, Dict, Optional, Type, Any

import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.weight_norm import weight_norm

from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, FlattenExtractor
from stable_baselines3.dqn.policies import DQNPolicy, QNetwork, BasePolicy
from stable_baselines3.common.type_aliases import Schedule


from encoding.state import StateDims


def add_weight_normalization(layer: nn.Module):
    return weight_norm(layer, dim=None)


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
        normalization_layers: bool = False,
        dropout_rate: float = 0.0,
        activation_fn: Type[nn.Module] = nn.ReLU,
    ) -> None:
        super().__init__(
            observation_space,
            action_space,
            features_extractor=None,
            normalize_images=False,
        )

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
            if normalization_layers:
                layer = add_weight_normalization(layer)
            layers.append(layer)
            activation = activation_fn()
            layers.append(activation)
            current_input_dim = layer_size
        # output layer
        output_layer = nn.Linear(current_input_dim, self.output_dim)
        layers.append(output_layer)

        self.q_net = nn.ModuleList(layers)
        print("NETWORK", self.q_net)

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
        return q

    def _predict(self, observation: th.Tensor, deterministic: bool = True) -> th.Tensor:
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
    ) -> None:
        super().__init__(
            observation_space,
            action_space,
            features_extractor=None,
            normalize_images=False,
        )

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
            if normalization_layers:
                shared_layer = add_weight_normalization(shared_layer)
            shared_layers.append(shared_layer)
            activation = activation_fn()
            shared_layers.append(activation)
            shared_layer_dim = layer_size

            if self.actions_in_state_space:
                action_layer = nn.Linear(action_layer_dim, layer_size)
                if normalization_layers:
                    action_layer = add_weight_normalization(action_layer)
                action_inputs.append(action_layer)
                activation = activation_fn()
                action_inputs.append(activation)
                action_layer_dim = layer_size

        # create value layers: (state_input_dim - action_input_dim) -> 1
        value_layer_dim = shared_layer_dim
        for layer_size in value_layer_sizes:
            value_layer = nn.Linear(value_layer_dim, layer_size)
            if normalization_layers:
                value_layer = add_weight_normalization(value_layer)
            value_layers.append(value_layer)
            activation = activation_fn()
            value_layers.append(activation)
            value_layer_dim = layer_size
        value_output_layer = nn.Linear(value_layer_dim, 1) # value output layer
        value_layers.append(value_output_layer)

        # advantage layer: shared_layer_sizes[-1] -> actions
        advantage_layer_dim = shared_layer_dim + action_layer_dim
        for layer_size in advantage_layer_sizes:
            adv_layer = nn.Linear(advantage_layer_dim, layer_size)
            if normalization_layers:
                adv_layer = add_weight_normalization(adv_layer)
            advantage_layers.append(adv_layer)
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


    def forward(self, obs: th.Tensor, deterministic: bool = False) -> th.Tensor:
        """
        Predict the q-values.

        :param obs: Observation
        :return: The estimated Q-Value for each action.
        """
        # preprocess inputs
        mask = None # mask will be 0.0 for rows that should be masked, so we can multiply with them and eliminate gradient propagation for these
        num_actions = self.state_dims.num_actions
        state_input = obs
        if self.actions_in_state_space:
            # obs: batch x actions x state
            mask = (~(obs.abs().sum(-1) == 0.0)).float() # batch x actions
            num_actions = mask.sum(-1) # batch (contains number of actions per batch)
            mask = mask.unsqueeze(-1) # batch x actions x 1

            state_input = obs[:, :, :self.state_input_size] # state input portion of input
            action_input = obs[:, :, self.state_input_size:] # action input portion of input

            action_stream = action_input
            for layer in self.action_input_net:
                action_stream = layer(action_stream)
                if (not deterministic) and self.dropout_rate > 0.0:
                    action_stream = F.dropout(action_stream, p=self.dropout_rate)
                action_stream = mask_output(action_stream, mask)
            
        # calculate streams
        shared_stream = state_input
        for layer in self.shared_net:
            shared_stream = layer(shared_stream)
            if (not deterministic) and self.dropout_rate > 0.0:
                shared_stream = F.dropout(shared_stream, p=self.dropout_rate)
            shared_stream = mask_output(shared_stream, mask)
        
        value_stream = shared_stream
        for layer_idx, layer in enumerate(self.value_net):
            value_stream = layer(value_stream)
            if (not deterministic) and layer_idx < len(self.value_net) - 1 and self.dropout_rate > 0.0:
                # no dropout in last layer, or if deterministic output is required
                value_stream = F.dropout(value_stream, p=self.dropout_rate)
            value_stream = mask_output(value_stream, mask)

        advantage_stream = shared_stream
        if self.actions_in_state_space:
            advantage_stream = th.cat((advantage_stream, action_stream), -1)
        for layer_idx, layer in enumerate(self.advantage_net):
            advantage_stream = layer(advantage_stream)
            if (not deterministic) and layer_idx < len(self.advantage_net) - 1 and self.dropout_rate > 0.0:
                # no dropout in last layer, or if deterministic output is required 
                advantage_stream = F.dropout(advantage_stream, p=self.dropout_rate)
            advantage_stream = mask_output(advantage_stream, mask)

        if self.actions_in_state_space:
            # adapt dimensions
            value_stream = value_stream.squeeze(-1)
            advantage_stream = advantage_stream.squeeze(-1) # batch x actions x 1 -> batch x actions (because we only have 1 output here)
            
        # calculate advantage mean
        advantage_stream_mean = advantage_stream.sum(-1) / num_actions
        # combine value and advantage streams into Q values
        q = value_stream + advantage_stream - advantage_stream_mean.unsqueeze(-1) # batch x actions

        return q
       
    def _predict(self, observation: th.Tensor, deterministic: bool = True) -> th.Tensor:
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
    ) -> None:
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            shared_layer_sizes=shared_layer_sizes,
            value_layer_sizes=value_layer_sizes,
            advantage_layer_sizes=advantage_layer_sizes,
            state_dims=state_dims,
            normalization_layers=normalization_layers,
            dropout_rate=dropout_rate,
            activation_fn=activation_fn
        )

        # intent prediction head
        self.intent_head = nn.Sequential(
            nn.Linear(shared_layer_sizes[-1], 256),
            nn.SELU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(256, 1)
        )

    def forward(self, obs: th.Tensor) -> th.Tensor:
        """
        Predict the q-values.

        :param obs: Observation
        :return: The estimated Q-Value for each action.
        """
        # preprocess inputs
        mask = None # mask will be 0.0 for rows that should be masked, so we can multiply with them and eliminate gradient propagation for these
        num_actions = self.state_dims.num_actions
        state_input = obs
        if self.actions_in_state_space:
            # obs: batch x actions x state
            mask = (~(obs.abs().sum(-1) == 0.0)).float() # batch x actions
            num_actions = mask.sum(-1) # batch (contains number of actions per batch)

            state_input = obs[:, :, :self.state_input_size] # state input portion of input
            action_input = obs[:, :, self.state_input_size:] # action input portion of input

            state_input = (state_input, mask.unsqueeze(-1)) # input requires mask now
            action_input = (action_input, mask.unsqueeze(-1)) # input requires mask now
            
            action_stream = self.action_input_net(action_input)
        
        # calculate streams
        shared_stream = self.shared_net(state_input)
        value_stream = self.value_net(shared_stream)
        advantage_stream = shared_stream
        if self.actions_in_state_space:
            advantage_stream = th.cat((advantage_stream, action_stream), -1)
        advantage_stream = self.advantage_net(shared_stream)

        if self.actions_in_state_space:
            # unpack values from (value, mask) tuples and adapt dimensions
            value_stream = value_stream[0]
            value_stream = value_stream.squeeze(-1)
            advantage_stream = advantage_stream[0]
            advantage_stream = advantage_stream.squeeze(-1) # batch x actions x 1 -> batch x actions (because we only have 1 output here)
            
        # calculate advantage mean
        advantage_stream_mean = advantage_stream.sum(-1) / num_actions
        # combine value and advantage streams into Q values
        q = value_stream + advantage_stream - advantage_stream_mean.unsqueeze(1) # batch x actions

        # intent prediction
        intent_logits = shared_stream.mean(dim=1) if self.actions_in_state_space else shared_stream  # batch x shared_dim
        intent_logits = self.intent_head(intent_logits) # batch x 1
        
        return q, intent_logits
       
    def _predict(self, observation: th.Tensor, deterministic: bool = True) -> th.Tensor:
        q_values, intent_logits = self(observation)
        if self.actions_in_state_space:
            # obs: batch x actions x state
            mask = (observation.abs().sum(-1) == 0.0) # batch x actions
            q_values = th.masked_fill(q_values, mask, float('-inf'))

        # Greedy action
        action = q_values.argmax(dim=1).reshape(-1)
        return action, intent_logits
    



class CustomDQNPolicy(DQNPolicy):
    """
    Policy class with Q-Value Net and target net for DQN

    :param observation_space: Observation space
    :param action_space: Action space
    :param lr_schedule: Learning rate schedule (could be constant)
    :param net_arch: The specification of the policy and value networks.
    :param activation_fn: Activation function
    :param features_extractor_class: Features extractor to use.
    :param features_extractor_kwargs: Keyword arguments
        to pass to the features extractor.
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param optimizer_class: The optimizer to use,
        ``th.optim.Adam`` by default
    :param optimizer_kwargs: Additional keyword arguments,
        excluding the learning rate, to pass to the optimizer
    """
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Discrete,
        lr_schedule: Schedule,
        net_arch,
        normalization_layers: bool = False,
        activation_fn: Type[nn.Module] = nn.ReLU,
        features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.normalization_layers = normalization_layers

        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch, # net_arch -> replaced here by hidden_layer_sizes
            activation_fn,
            features_extractor_class,
            features_extractor_kwargs,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            normalize_images=normalize_images,
        )

    def make_q_net(self) -> QNetwork:
        # Make sure we always have separate networks for features extractors etc
        net_args = deepcopy(self._update_features_extractor(self.net_args, features_extractor=None))
        del net_args['normalize_images']
        del net_args['features_extractor']
        del net_args['features_dim']
        del net_args['activation_fn']
        
        arch = net_args.pop('net_arch')
        net_cls = arch.pop('net_cls')

        return net_cls(**net_args, **arch).to(self.device) # TODO th.compile(
        

# @torch.no_grad()
# def select_actions_eps_greedy(self, training: bool, q_values: torch.FloatTensor, action_mask: Union[torch.FloatTensor, None], epsilon: float) -> torch.LongTensor:
#     """ Epsilon-greedy policy.

#     Args:
#         node_keys: current node keys
#         state_vectors: current state (dimension batch x state_dim    or num_actions x state_dim)
#         epsilon: current scheduled exploration rate

#     Returns:
#         List of action indices for action selected by the agent for the current states
#         List of predicted intent classes if supported by model, else None
#     """
#     # TODO mask STOP action in first turn?

#     # epsilon greedy exploration
#     if training and random.random() < epsilon:
#         # exploration
#         if torch.is_tensor(action_mask):
#             # only choose between allowed actions after masking
#             allowed_action_indices = (~action_mask).float() # batch x num_max_actions
#             next_action_indices = [random.choice(allowed_action_indices[batch_idx].nonzero().view(-1).tolist()) for batch_idx in range(allowed_action_indices.size(0))] # 1-dim list with one valid action index per batch item
#         else:
#             # no masking: random choice
#             next_action_indices = [random.randint(0, self.adapter.num_actions - 1) for _ in range(q_values.size(0))]
#     else:
#         # exploitation
#         final_values = q_values
#         if torch.is_tensor(action_mask):
#             final_values = torch.masked_fill(q_values, action_mask[:,:q_values.size(-1)], float('-inf'))
#         next_action_indices = final_values.argmax(-1).tolist()

#     return next_action_indices


# def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
#         slope = (end_e - start_e) / duration
#         return max(slope * t + start_e, end_e)

# class DQNAlgorithm(Algorithm):
#     def __init__(self, buffer, targets,
#         timesteps_per_reset: int,
#         reset_exploration_times: int,
#         max_grad_norm: float,
#         batch_size: int,
#         gamma: float,
#         exploration_fraction: float,
#         eps_start: float,
#         eps_end: float,
#         train_frequency: int,
#         warmup_turns: int,
#         target_network_update_frequency: int,
#         q_value_clipping: float
#         ) -> None:
#         global INSTANCES
#         INSTANCES[InstanceType.ALGORITHM] = self
#         print("DQN Trainer")

#         self.eps_start = eps_start
#         self.eps_end = eps_end
#         self.exploration_fraction = exploration_fraction
#         self.timesteps_per_reset = timesteps_per_reset


#     def run_single_timestep(self, engine, timestep: int):
#         """
#         Args:
#             timestep: GLOBAL timestep (not step in current episode)
#         """
#         epsilon = linear_schedule(start_e=self.eps_start, end_e=self.eps_end, duration=self.exploration_fraction * self.timesteps_per_reset, t=timestep % self.timesteps_per_reset)

#         # TODO get observation as vector
#         # TODO select actions based on eps-greedy
#         # TODO call environment step() using action
#         # TODO get next observation as vector
#         # TODO store experience in replay buffer