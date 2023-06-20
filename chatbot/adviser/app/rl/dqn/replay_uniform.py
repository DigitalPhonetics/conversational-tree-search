from copy import deepcopy
from typing import Any, Dict, List, Union, NamedTuple
import torch
from chatbot.adviser.app.rl.spaceAdapter import SpaceAdapter

from chatbot.adviser.app.rl.utils import EnvInfo, StateEntry


class ReplayBufferSamples(NamedTuple):
    observations: Dict[StateEntry, List[Any]]
    actions: torch.Tensor
    next_observations: Dict[StateEntry, List[Any]]
    dones: torch.Tensor
    rewards: torch.Tensor
    infos: Dict[str, List[Any]]
    indices: List[int]


class UniformReplayBuffer:
    def __init__(
        self,
        buffer_size: int,
        adapter: SpaceAdapter,
        device: Union[torch.device, str] = "cpu",
    ):
        self.buffer_size = buffer_size
        self.pos = 0
        self.full = False
        self.device = device
        self.action_config = adapter.configuration.action_config

        self.observations = { }
        self.next_observations = { }
        self.actions = [None] * self.buffer_size
        self.rewards = [None] * self.buffer_size
        self.dones = [None] * self.buffer_size
        self.infos = { key: [None] * self.buffer_size for key in EnvInfo }

    def __len__(self):
        return self.buffer_size if self.full else self.pos

    def add(
        self,
        env_id: int,
        obs: Dict[StateEntry, Any],
        next_obs: Dict[StateEntry, Any],
        action: int,
        reward: float,
        done: bool,
        infos: Dict[EnvInfo, Any],
        global_step: int
    ):

        # Reshape needed when using multiple envs with discrete observations
        # as numpy cannot broadcast (n_discrete,) to (n_discrete, 1)
        # if isinstance(self.observation_space, spaces.Discrete):
        #     obs = obs.reshape((1,) + self.obs_shape)
        #     next_obs = next_obs.reshape((self.n_envs,) + self.obs_shape)

        # Same, for actions
        # if isinstance(self.action_space, spaces.Discrete):
        #     action = action.reshape((self.n_envs, self.action_dim))

        # Copy to avoid modification by reference
        for key in set(obs.keys()).union(next_obs.keys()):
            if not key in self.observations:
                self.observations[key] = [None] * self.buffer_size
                self.next_observations[key] = [None] * self.buffer_size
            if key in obs:
                self.observations[key][self.pos] = obs[key].clone().detach() if torch.is_tensor(obs[key]) else deepcopy(obs[key]) 
            if key in next_obs:
                self.next_observations[key][self.pos] = next_obs[key].clone().detach() if torch.is_tensor(next_obs[key]) else deepcopy(next_obs[key])
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.dones[self.pos] = done
        for key in EnvInfo:
            self.infos[key][self.pos] = deepcopy(infos[key])

        self.pos += 1
        if self.pos == self.buffer_size:
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
        batch_inds = torch.randint(low=0, high=len(self), size=(batch_size,)).tolist()

        obs = { key: [deepcopy(self.observations[key.value][index]) for index in batch_inds] for key in StateEntry }
        next_obs = { key: [deepcopy(self.next_observations[key.value][index]) for index in batch_inds] for key in StateEntry }

        actions = torch.tensor([self.actions[idx] for idx in batch_inds], dtype=torch.long, device=self.device).unsqueeze(1)
        dones = torch.tensor([self.dones[idx] for idx in batch_inds], dtype=torch.float, device=self.device).unsqueeze(1)
        rewards = torch.tensor([self.rewards[idx] for idx in batch_inds], dtype=torch.float, device=self.device).unsqueeze(1)
        infos = { key: [deepcopy(self.infos[key][index]) for index in batch_inds] for key in EnvInfo }
      
        return ReplayBufferSamples(observations=obs,
            next_observations=next_obs,
            actions=actions,
            rewards=rewards,
            dones=dones,
            infos=infos,
            indices=batch_inds)
