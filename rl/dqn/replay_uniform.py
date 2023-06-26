from copy import deepcopy
from typing import Any, Dict, List, Union, NamedTuple
import torch as th
import numpy as np

from encoding.state import StateEncoding
from rl.utils import EnvInfo, State


class ReplayBufferSamples(NamedTuple):
    observations: Dict[str, th.Tensor]
    actions: th.Tensor
    next_observations: Dict[str, th.Tensor]
    dones: th.Tensor
    rewards: th.Tensor
    infos: List[Dict[EnvInfo, Any]]



class UniformReplayBuffer:
    def __init__(
        self,
        buffer_size: int,
        state_enc: StateEncoding,
        device: Union[th.device, str] = "cpu",
    ):
        self.buffer_size = buffer_size
        self.pos = 0
        self.full = False
        self.device = device
        self.action_config = state_enc.action_config

        # buffers
        # get action dimension in (1 if actions in action space, else num_actions)
        action_dim = state_enc.space_dims.num_actions if state_enc.action_config.in_state_space else 1
        self.obs = th.zeros(buffer_size, action_dim, state_enc.space_dims.state_vector)
        self.next_obs = th.zeros(buffer_size, action_dim, state_enc.space_dims.state_vector)
        self.action = th.zeros(buffer_size, dtype=th.int64)
        self.done = th.zeros(buffer_size, dtype=th.float32)
        self.reward = th.zeros(buffer_size, dtype=th.float32)
        self.infos = np.empty(shape=(buffer_size,), dtype=object)
        self.artificial_transition = th.zeros(buffer_size, dtype=th.int64) # can stay on CPU


    def __len__(self):
        return self.buffer_size if self.full else self.pos
    
    def add_single_transition(self,
        obs: th.Tensor,
        next_obs: th.Tensor,
        action: th.Tensor,
        reward: th.Tensor,
        done: th.Tensor,
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
        action: th.Tensor,
        reward: th.Tensor,
        done: th.Tensor,
        infos: List[Dict[EnvInfo, Any]],
        is_aritificial: bool = False
    ):
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

    def sample(self, batch_size: int) -> ReplayBufferSamples:
        """
        Sample elements from the replay buffer.
        Custom sampling when using memory efficient variant,
        as we should not sample the element with index `self.pos`
        See https://github.com/DLR-RM/stable-baselines3/pull/28#issuecomment-637559274

        :param batch_size: Number of element to sample
        :param env: associated gym VecEnv
            to normalize the observations/rewards when sampling
        :return:
        """
        upper_bound = self.capacity if self.full else self.pos
        batch_inds = th.randint(low=0, high=upper_bound, size=(batch_size,))
       
        # Sample randomly the env idx
        return ReplayBufferSamples(
            self.obs[batch_inds].clone().detach().to(self.device),
            self.action[batch_inds].clone().detach().to(self.device).view(-1, 1),
            self.next_obs[batch_inds].clone().detach().to(self.device),
            self.done[batch_inds].clone().detach().to(self.device).view(-1, 1),
            self.reward[batch_inds].clone().detach().to(self.device).view(-1, 1),
            [self.infos[batch_idx] for batch_idx in batch_inds.tolist()]
        )