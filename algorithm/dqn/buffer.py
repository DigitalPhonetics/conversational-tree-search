from typing import Any, Dict, Optional, Union, List, NamedTuple
from copy import deepcopy
import random

from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.type_aliases import TensorDict

import numpy as np
import torch as th

from utils.utils import EnvInfo


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
        self.action = th.zeros(buffer_size, dtype=th.int64)
        self.done = th.zeros(buffer_size, dtype=th.float32)
        self.reward = th.zeros(buffer_size, dtype=th.float32)
        self.infos = np.empty(shape=(buffer_size,), dtype=object)
        self.artificial_transition = th.zeros(buffer_size, dtype=th.int64) # can stay on CPU

        # pointers / counters
        self.capacity = buffer_size
        self.pos = 0
        self.full = False
    
    def save_params(self) -> Dict[str, Any]:
        return {
            "buffer_size": self.buffer_size,
            "obs": self.obs,
            "next_obs": self.next_obs,
            "done": self.done,
            "reward": self.reward,
            "infos": self.infos,
            "artificial_transition": self.artificial_transition,
            "capacity": self.capacity,
            "pos": self.pos,
            "full": self.full
        }

    def load_params(self, data):
        self.obs = data['obs']
        self.next_obs = data['next_obs']
        self.done = data['done']
        self.reward = data['reward']
        self.infos = data['infos']
        self.artificial_transition = data['artificial_transition']
        self.capacity = data['capacity']
        self.pos = data['pos']
        self.full = data['full']

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
        if "terminal_observation" in infos:
            # delete, because terminal_observation is already stored in next_observation by stable-baselines!
            del infos['terminal_observation']
        self.infos[self.pos] = deepcopy(infos)
        # save memory
        del self.infos[self.pos][EnvInfo.USER_UTTERANCE_HISTORY]
        del self.infos[self.pos][EnvInfo.SYSTEM_UTTERANCE_HISTORY]
         # replay info
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
            if "terminal_observation" in info:
                    # delete, because terminal_observation is already stored in next_observation by stable-baselines!
                    del info['terminal_observation']
            # Copy to avoid mutation by reference
            for batch_idx, info in enumerate(infos):
                self.infos[self.pos+batch_idx] = deepcopy(info)
            # save memory
            del self.infos[self.pos+batch_idx][EnvInfo.USER_UTTERANCE_HISTORY]
            del self.infos[self.pos+batch_idx][EnvInfo.SYSTEM_UTTERANCE_HISTORY]
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
    
    def reset_last_transition_indices(self):
        pass 
    


class PrioritizedReplayBufferSamples(NamedTuple):
    observations: TensorDict
    actions: th.Tensor
    next_observations: TensorDict
    dones: th.Tensor
    rewards: th.Tensor
    infos: List[Dict[EnvInfo, Any]]
    weights: np.ndarray
    indices: np.ndarray


class SumTree:
    def __init__(self, capacity):
        self.write = 0
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.n_entries = 0
        # self.pending_idx = set()

    # update to the root node
    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    # find sample on leaf node
    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[0]

    # store priority and sample
    def add(self, p):
        idx = self.write + self.capacity - 1
        # self.pending_idx.add(idx)

        self.update(idx, p)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0

        if self.n_entries < self.capacity:
            self.n_entries += 1

    # update priority
    def update(self, idx, p):
        # if idx not in self.pending_idx:
        #     return
        # self.pending_idx.remove(idx)

        change = p - self.tree[idx]
        self.tree[idx] = p
        self._propagate(idx, change)


    # get priority and sample
    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1
        # self.pending_idx.add(idx)
        return (idx, self.tree[idx], dataIdx)


class PrioritizedReplayBuffer(CustomReplayBuffer):
    def __init__(self,
            buffer_size: int,
            observation_space,
            action_space,
            alpha: float,
            beta: float,
            device: Union[th.device, str] = "cpu",
            **kwargs):
        super().__init__(buffer_size=buffer_size, observation_space=observation_space, action_space=action_space, device=device, **kwargs)
        self.tree = SumTree(buffer_size)
        self.max_priority = 1.0
        self.alpha = alpha
        self.beta = beta
        self.e = (1.0/buffer_size)

    def save_params(self) -> Dict[str, Any]:
        return super().save_params() | {
            "tree": {
                "write": self.tree.write,
                "capacity": self.tree.capacity,
                "tree": self.tree.tree,
                "n_entries": self.tree.n_entries
            },
            "max_priority": self.max_priority,
            "alpha": self.alpha,
            "beta": self.beta,
            "e": self.e
        }
    
    def load_params(self, data):
        self.tree.write = data['tree']['write']
        self.tree.tree = data['tree']['tree']
        self.tree.capacity = data['tree']['capacity']
        self.tree.n_entries = data['tree']['n_entries']
        self.max_priority = data['max_priority']
        super().load_params(data)

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
        if "terminal_observation" in infos:
            # delete, because terminal_observation is already stored in next_observation by stable-baselines!
            del infos['terminal_observation']
        self.infos[self.pos] = deepcopy(infos)
        # save memory
        del self.infos[self.pos][EnvInfo.USER_UTTERANCE_HISTORY]
        del self.infos[self.pos][EnvInfo.SYSTEM_UTTERANCE_HISTORY]
        # replay info
        self.artificial_transition[self.pos] = int(is_artificial)
        self.tree.add(self.max_priority)

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
                if "terminal_observation" in info:
                    # delete, because terminal_observation is already stored in next_observation by stable-baselines!
                    del info['terminal_observation']
                self.infos[self.pos+batch_idx] = deepcopy(info)
                # save memory
                del self.infos[self.pos+batch_idx][EnvInfo.USER_UTTERANCE_HISTORY]
                del self.infos[self.pos+batch_idx][EnvInfo.SYSTEM_UTTERANCE_HISTORY]
                self.tree.add(self.max_priority)
            self.pos = end_pos
            if end_pos == self.capacity:
                self.full = True
                self.pos = 0

    def sample(self, batch_size: int, env: Optional[VecNormalize] = None) -> PrioritizedReplayBufferSamples:
        segment = self.tree.total() / batch_size

        obs = []
        next_obs = []
        actions = []
        rewards = []
        dones = []
        infos = []
        indices = []
        weights = []

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            (idx, p, data_index) = self.tree.get(s)

            if data_index < self.tree.n_entries:
                obs.append(self.obs[data_index].clone().detach().unsqueeze(0))
                next_obs.append(self.next_obs[data_index].clone().detach().unsqueeze(0))
                actions.append(self.action[data_index])
                rewards.append(self.reward[data_index])
                dones.append(self.done[data_index])
                infos.append(deepcopy(self.infos[data_index]))
                indices.append(idx)
                weights.append(p / self.tree.total())
            else: 
                # invalid index
                continue
                
        empty = len(indices) == 0
        while len(indices) < batch_size:
            # This should rarely happen
            if empty:
                rand_idx = random.randint(0, len(self) - 1)
                obs.append(self.obs[rand_idx].clone().detach().unsqueeze(0))
                next_obs.append(self.next_obs[rand_idx].clone().detach().unsqueeze(0))
                actions.append(self.action[rand_idx])
                rewards.append(self.reward[rand_idx])
                dones.append(self.done[rand_idx])
                infos.append(deepcopy(self.infos[rand_idx]))
                treeIdx = rand_idx + self.tree.capacity - 1
                indices.append(treeIdx)
                weights.append(self.tree.tree[treeIdx]) 
            else:
                rand_idx = random.randint(0, len(indices) - 1)
                obs.append(obs[rand_idx].clone().detach().unsqueeze(0))
                next_obs.append(next_obs[rand_idx].clone().detach().unsqueeze(0))
                actions.append(actions[rand_idx])
                rewards.append(rewards[rand_idx])
                dones.append(dones[rand_idx])
                infos.append(deepcopy(infos[rand_idx]))
                indices.append(indices[rand_idx])
                weights.append(weights[rand_idx])

        weights = th.pow(self.tree.n_entries * th.tensor(weights, device=self.device), -self.beta)
        weights = weights / weights.max()
       
        return PrioritizedReplayBufferSamples(
            th.cat(obs, dim=0).to(self.device),
            th.tensor(actions, dtype=th.long, device=self.device).view(-1, 1),
            th.cat(next_obs, dim=0).to(self.device),
            th.tensor(dones, dtype=th.float, device=self.device).view(-1, 1),
            th.tensor(rewards, device=self.device).view(-1, 1),
            infos,
            weights,
            indices
        )
    def update_beta(self, beta: float):
        self.beta = beta

    def update_weights(self, batch_inds: np.ndarray, weights: np.ndarray):
        for idx, priority in zip(batch_inds, weights.tolist()):
            prio = (priority + self.e) ** self.alpha
            self.max_priority = max(self.max_priority, prio)
            self.tree.update(idx, prio)



class PrioritizedLAPReplayBuffer(PrioritizedReplayBuffer):
    def __init__(self,
            buffer_size: int,
            observation_space,
            action_space,
            alpha: float,
            beta: float,
            device: Union[th.device, str] = "cpu",
            **kwargs):
        super().__init__(buffer_size=buffer_size, observation_space=observation_space, action_space=action_space, alpha=alpha, beta=beta, device=device, **kwargs)

    def update_weights(self, batch_inds: np.ndarray, weights: np.ndarray):
        for idx, priority in zip(batch_inds, weights.tolist()):
            prio = max(priority ** self.alpha, 1.0) # clip samples with low priority to at least 1: LAP algorithm
            self.max_priority = max(self.max_priority, prio)
            self.tree.update(idx, prio)

