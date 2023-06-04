from typing import Any, Dict, Optional, Union, List, NamedTuple
from copy import deepcopy

from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.type_aliases import TensorDict

import gymnasium.spaces as spaces
import numpy as np
import torch as th

from utils.utils import EnvInfo
from encoding.state import StateDims


class CustomReplayBufferSamples(NamedTuple):
    observations: TensorDict
    actions: th.Tensor
    next_observations: TensorDict
    dones: th.Tensor
    rewards: th.Tensor
    infos: List[Dict[EnvInfo, Any]]


class CustomReplayBuffer:
    """
    Replay buffer used in off-policy algorithms like SAC/TD3.

    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param device: PyTorch device
    :param n_envs: Number of parallel environments
    :param optimize_memory_usage: Enable a memory efficient variant
        of the replay buffer which reduces by almost a factor two the memory used,
        at a cost of more complexity.
        See https://github.com/DLR-RM/stable-baselines3/issues/37#issuecomment-637501195
        and https://github.com/DLR-RM/stable-baselines3/pull/28#issuecomment-637559274
        Cannot be used in combination with handle_timeout_termination.
    :param handle_timeout_termination: Handle timeout termination (due to timelimit)
        separately and treat the task as infinite horizon task.
        https://github.com/DLR-RM/stable-baselines3/issues/284
    """

    def __init__(
        self,
        buffer_size: int,
        observation_space,
        action_space,
        device: Union[th.device, str] = "cpu",
        **kwargs
    ):  
        # self.device = device
        self.device = device

        # buffers
        self.obs = th.zeros(buffer_size, *observation_space.shape)
        self.next_obs = th.zeros(buffer_size, *observation_space.shape)
        self.action = th.zeros(buffer_size, dtype=th.int32)
        self.done = th.zeros(buffer_size, dtype=th.float32)
        self.reward = th.zeros(buffer_size, dtype=th.float32)
        self.infos = np.empty(shape=(buffer_size,), dtype=object)
        self.artificial_transition = th.zeros(buffer_size, dtype=th.int32) # can stay on CPU

        # pointers / counters
        self.capacity = buffer_size
        self.pos = 0
        self.full = False

    def clear(self):
        self.full = False
        self.pos = 0

    def __len__(self):
        if self.full:
            return self.capacity
        return self.pos

    def add_single_transition(self,
        obs: th.Tensor,
        next_obs: th.Tensor,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: Dict[str, Any],
        is_artificial: bool = False # for HER generated replay episodes
    ) -> None:
        self.obs[self.pos] = obs.clone().detach()
        self.next_obs[self.pos] = next_obs.clone().detach()
        self.action[self.pos] = action.item()
        self.reward[self.pos] = reward.item()
        self.done[self.pos] = done.item()
        self.infos[self.pos] = deepcopy(infos)
        self.artificial_transition[self.pos] = int(is_artificial)

        self.pos += 1
        if self.pos == self.capacity:
            self.full = True
            self.pos = 0

    def add(
        self,
        obs: th.Tensor,
        next_obs: th.Tensor,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: List[Dict[str, Any]],
        is_aritificial: bool = False,
    ) -> None:
        batch_size = len(infos)
        end_pos = self.pos + batch_size
        if end_pos >= self.capacity:
            # doesn't fit completely - split batch
            for batch_idx, info in enumerate(infos):
                self.add_single_transition(obs[batch_idx], next_obs[batch_idx],
                                            action[batch_idx], reward[batch_idx],
                                            done[batch_idx], info, is_aritificial)
        else:
            # does fit - add batch
            self.obs[self.pos:end_pos] = obs.clone().detach()
            self.next_obs[self.pos:end_pos] = next_obs.clone().detach()
            self.action[self.pos:end_pos] = th.from_numpy(action)
            self.reward[self.pos:end_pos] = th.from_numpy(reward)
            self.done[self.pos:end_pos] = th.from_numpy(done)
            self.artificial_transition[self.pos:end_pos] = int(is_aritificial)
            # Copy to avoid mutation by reference
            for batch_idx, info in enumerate(infos):
                self.infos[self.pos+batch_idx] = deepcopy(info)
            self.pos = end_pos
            if end_pos == self.capacity:
                self.full = True
                self.pos = 0
            
    def sample(self, batch_size: int, env: Optional[VecNormalize] = None) -> CustomReplayBufferSamples:
        """
        Sample elements from the replay buffer.

        :param batch_size: Number of element to sample
        :param env: associated gym VecEnv
            to normalize the observations/rewards when sampling
        :return:
        """
        upper_bound = self.capacity if self.full else self.pos
        batch_inds = th.randint(low=0, high=upper_bound, size=(batch_size,))
       
        # Sample randomly the env idx
        return CustomReplayBufferSamples(
            self.obs[batch_inds].clone().detach().to(self.device),
            self.action[batch_inds].clone().detach().to(self.device).view(-1, 1),
            self.next_obs[batch_inds].clone().detach().to(self.device),
            self.done[batch_inds].clone().detach().to(self.device).view(-1, 1),
            self.reward[batch_inds].clone().detach().to(self.device).view(-1, 1),
            [self.infos[batch_idx] for batch_idx in batch_inds.tolist()]
        )