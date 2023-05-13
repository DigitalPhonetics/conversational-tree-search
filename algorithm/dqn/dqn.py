
from typing import List, Tuple, Union, Dict, Optional, Type, Any

import torch as th
import torch.nn as nn
from torch.nn.utils.weight_norm import weight_norm

from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, FlattenExtractor
from stable_baselines3.dqn.policies import DQNPolicy, QNetwork, BasePolicy
from stable_baselines3.common.type_aliases import Schedule



def add_weight_normalization(layer: nn.Module):
    return weight_norm(layer, dim=None)


class MaskingLayer(nn.Module):
    def __init__(self, original_layer: nn.Module) -> None:
        super().__init__()
        self.original_layer = original_layer
    
    def forward(self, input: Tuple[th.Tensor, th.Tensor]):
        """
        Args:
            input[0]: batch x state_dim
            input[1]: mask: batch x actions x state_dim
        """
        return self.original_layer(input[0]) * input[1], input[1]


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

        # network architecture
        self.dropout_rate = dropout_rate

        # create layers
        layers = []
        current_input_dim = observation_space.shape[-1]
        self.actions_in_state_space = len(observation_space.shape) > 1
        self.output_dim = 1 if self.actions_in_state_space else action_space.n

        for layer_size in hidden_layer_sizes:
            layer = nn.Linear(current_input_dim, layer_size)
            if normalization_layers:
                layer = add_weight_normalization(layer)
            if self.actions_in_state_space:
                layer = MaskingLayer(layer)
            layers.append(layer)
            activation = activation_fn()
            if self.actions_in_state_space:
                activation = MaskingLayer(activation)
            layers.append(activation)
            if dropout_rate > 0.0:
                layers.append(nn.Dropout(p=dropout_rate))
            current_input_dim = layer_size
        # output layer
        output_layer = nn.Linear(current_input_dim, self.output_dim)
        if self.actions_in_state_space:
            output_layer = MaskingLayer(output_layer)
        layers.append(output_layer)

        self.q_net = nn.Sequential(*layers)

    def forward(self, obs: th.Tensor) -> th.Tensor:
        """
        Predict the q-values.

        :param obs: Observation
        :return: The estimated Q-Value for each action.
        """
        mask = None # mask will be 0.0 for rows that should be masked, so we can multiply with them and eliminate gradient propagation for these
        x = obs
        if self.actions_in_state_space:
            # obs: batch x actions x state
            mask = (~(obs.abs().sum(-1) == 0.0)).float() # batch x actions
            x = (obs, mask.unsqueeze(-1)) # input requires mask now
        
        q = self.q_net(x)
        if self.actions_in_state_space:
            # reshape outputs (discard mask, which is q[1])
            q = q[0]
            q = q.view(-1, self.action_space.n)
        return q

    def _predict(self, observation: th.Tensor, deterministic: bool = True) -> th.Tensor:
        q_values = self(observation)
        if self.actions_in_state_space:
            # obs: batch x actions x state
            mask = (observation.abs().sum(-1) == 0.0) # batch x actions
            q_values = th.masked_fill(q_values, mask, float('-inf'))

        # Greedy action
        action = q_values.argmax(dim=1).reshape(-1)
        return action
    


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
        hidden_layer_sizes: List[int],
        normalization_layers: bool = False,
        activation_fn: Type[nn.Module] = nn.ReLU,
        features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.hidden_layer_sizes = hidden_layer_sizes
        self.normalization_layers = normalization_layers

        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            None, # net_arch -> replaced here by hidden_layer_sizes
            activation_fn,
            features_extractor_class,
            features_extractor_kwargs,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            normalize_images=normalize_images,
        )

    def make_q_net(self) -> QNetwork:
        # Make sure we always have separate networks for features extractors etc
        net_args = self._update_features_extractor(self.net_args, features_extractor=None)
        del net_args['net_arch']
        del net_args['normalize_images']
        del net_args['features_extractor']
        del net_args['features_dim']
        return th.compile(CustomQNetwork(hidden_layer_sizes=self.hidden_layer_sizes,
                              normalization_layers=self.normalization_layers,
                              **net_args
            ).to(self.device))


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