from typing import Any, Dict, List, Union, NamedTuple
import gym.spaces as spaces
import torch
from stable_baselines3.common.buffers import BaseBuffer

from encoding.state import StateEncoding


class ReplayBufferSamples(NamedTuple):
    observations: Union[torch.tensor, torch.nn.utils.rnn.PackedSequence]
    actions: torch.Tensor
    next_observations: Union[torch.tensor, torch.nn.utils.rnn.PackedSequence]
    dones: torch.Tensor
    rewards: torch.Tensor
    infos: Dict[str, List[Any]]


class RolloutBuffer(BaseBuffer):
    """
    Rollout buffer used in on-policy algorithms

    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param device:
    :param n_envs: Number of parallel environments
    :param optimize_memory_usage: Enable a memory efficient variant
        of the replay buffer which reduces by almost a factor two the memory used,
        at a cost of more complexity.
        See https://github.com/DLR-RM/stable-baselines3/issues/37#issuecomment-637501195
        and https://github.com/DLR-RM/stable-baselines3/pull/28#issuecomment-637559274
    :param handle_timeout_termination: Handle timeout termination (due to timelimit)
        separately and treat the task as infinite horizon task.
        https://github.com/DLR-RM/stable-baselines3/issues/284
    """

    def __init__(
        self,
        max_steps: int,
        state_enc: StateEncoding,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[torch.device, str] = "cpu",
        gamma: float = 0.99,
        n_envs: int = 1
    ):
        super(RolloutBuffer, self).__init__(max_steps, observation_space, action_space, device, n_envs=n_envs)
        self.action_config = state_enc.action_config
        self.max_steps = self.max_steps
        self.gamma = gamma
        self.reset()
        

    def reset(self):
        self.observations = torch.empty((self.max_steps + 1, self.n_envs) + self.obs_shape)
        self.actions = torch.empty((self.max_steps + 1, self.n_envs))
        self.rewards = torch.zeros((self.max_steps + 1, self.n_envs))
        self.values = torch.empty((self.max_steps + 1, self.n_envs))
        self.next_values = torch.empty((self.max_steps + 1, self.n_envs))
        self.dones = torch.ones((self.max_steps + 1, self.n_envs))
        self.infos = [[] for _ in range(self.n_envs)]
        self.buffer_lengths = torch.zeros(self.n_envs)
        self.advantages = None
        self.step = 0

    def add(
        self,
        env_indices: torch.LongTensor,
        obs: torch.FloatTensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
        infos: Dict[str, Any],
        values: torch.FloatTensor,
        next_values: torch.FloatTensor
    ) -> None:

        # Copy to avoid modification by reference
        self.observations[self.step, env_indices] = obs.cpu().clone().detach()
        self.actions[self.step, env_indices] = action.cpu().clone().detach()
        self.rewards[self.step, env_indices] = reward.cpu().clone().detach()
        self.dones[self.step, env_indices] = done.cpu().clone().detach()
        self.values[self.step, env_indices] = values.cpu().clone().detach()
        self.next_values[self.step, env_indices] = next_values.cpu().cone().detach()
        for env_idx, info in zip(env_indices.list(), infos):
            self.infos[env_idx].append(info)
        self.buffer_lengths[env_indices] += 1
        self.step += 1

    @torch.no_grad()
    def _calculate_advantages(self):
        returns = torch.zeros_like(self.rewards)
        for t in reversed(range(self.max_steps)):
            nextnonterminal = 1.0 - self.dones[t + 1]
            next_return = returns[t + 1]
            returns[t] = self.rewards[t] + self.gamma * nextnonterminal * next_return
        self.advantages = returns - self.values

    @torch.no_grad()
    def sample(self, minibatch_size: int) -> ReplayBufferSamples:
        if not self.advantages:
            self._calculate_advantages()

        # TODO sample minibatch

        # bootstrap value
        next_value = self.next_values.reshape(1, -1)
        returns = torch.zeros_like(self.rewards)
        for t in reversed(range(self.step)):
            if t == self.max_steps - 1:
                nextnonterminal = 1.0 - next_done
                next_return = next_value
            else:
                nextnonterminal = 1.0 - dones[t + 1]
                next_return = returns[t + 1]
            returns[t] = rewards[t] + args.gamma * nextnonterminal * next_return
        advantages = returns - values


    # def _get_samples(self, batch_inds: torch.LongTensor, env: Optional[VecNormalize] = None) -> ReplayBufferSamples:
    #     if self.action_config == ActionConfig.ACTIONS_IN_ACTION_SPACE:
    #         obs = self.observations[batch_inds, 0, :].clone().detach().to(self.device)
    #     else:
    #         obs = pack_sequence([self.observations[batch_ind].clone().detach().to(self.device) for batch_ind in batch_inds], enforce_sorted=False)

    #     if self.optimize_memory_usage:
    #         if self.action_config == ActionConfig.ACTIONS_IN_ACTION_SPACE:
    #             next_obs = self.observations[(batch_inds + 1) % self.buffer_size, 0, :].clone().detach().to(self.device)
    #         else:
    #             next_obs = pack_sequence([self.observations[(batch_ind + 1) % self.buffer_size].clone().detach().to(self.device) for batch_ind in batch_inds], enforce_sorted=False)
    #     else:
    #         if self.action_config == ActionConfig.ACTIONS_IN_ACTION_SPACE:
    #             next_obs = self.next_observations[batch_inds, 0, :].clone().detach().to(self.device)
    #         else:
    #             next_obs = pack_sequence([self.next_observations[batch_ind].clone().detach().to(self.device) for batch_ind in batch_inds], enforce_sorted=False)

      
    #     return ReplayBufferSamples(obs,
    #         self.actions[batch_inds, 0, :].clone().detach().to(self.device),
    #         next_obs,
    #         # Only use dones that are not due to timeouts
    #         # deactivated by default (timeouts is initialized as an array of False)
    #         self.dones[batch_inds, 0].clone().detach().reshape(-1, 1).to(self.device),
    #         self.rewards[batch_inds, 0].clone().detach().reshape(-1, 1).to(self.device),
    #         {key: [self.infos[key][batch_ind] for batch_ind in batch_inds] for key in self.infos})
