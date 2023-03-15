from copy import deepcopy
import numpy as np
from typing import Any, List
from typing import Dict, NamedTuple
import torch
import random
from chatbot.adviser.app.rl.spaceAdapter import SpaceAdapter
from chatbot.adviser.app.rl.utils import EnvInfo, StateEntry


class PrioritizedReplayBufferSamples(NamedTuple):
    observations: Dict[StateEntry, List[Any]]
    actions: torch.Tensor
    next_observations: Dict[StateEntry, List[Any]]
    dones: torch.Tensor
    rewards: torch.Tensor
    weights: np.ndarray
    indices: np.ndarray
    infos: Dict[EnvInfo, List[Any]]


class SumTree:
    def __init__(self, capacity):
        self.write = 0
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.n_entries = 0
        # self.pending_idx = set()

        self.observations =  { }
        self.next_observations =  { }
        self.rewards = [None] * capacity
        self.dones = [None] * capacity
        self.infos = { key: [None] * self.capacity for key in EnvInfo }
        self.actions = [None] * capacity

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
    def add(self, p, obs, next_obs, action, reward, done, info):
        idx = self.write + self.capacity - 1
        # self.pending_idx.add(idx)

        for key in obs:
            if not key in self.observations:
                self.observations[key] = [None] * self.capacity
                self.next_observations[key] = [None] * self.capacity
            self.observations[key][self.write] = obs[key].clone().detach().cpu() if torch.is_tensor(obs[key]) else deepcopy(obs[key]) 
            self.next_observations[key][self.write] = next_obs[key].clone().detach().cpu() if torch.is_tensor(next_obs[key]) else deepcopy(next_obs[key])
        self.actions[self.write] = action
        self.rewards[self.write] = reward
        self.dones[self.write] = done
        for key in EnvInfo:
            self.infos[key][self.write] = deepcopy(info[key])

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



class PrioritizedReplayBuffer:
    def __init__(self, buffer_size, adapter: SpaceAdapter, device: str = "cpu", alpha: float = 0.6, beta: float = 0.4):
        self.tree = SumTree(buffer_size)
        self.max_priority = 1.0
        self.action_config = adapter.configuration.action_config
        self.device = device
        self.alpha = alpha
        self.beta = beta
        self.e = (1.0/buffer_size)

    def add(self, env_id: int, obs: Dict[StateEntry, Any], next_obs: Dict[StateEntry, Any], action: int, reward: float, done: bool, info: Dict[EnvInfo, Any], global_step: int) -> None:
        if next_obs != None:
            self.tree.add(self.max_priority, obs, next_obs, action, reward, done, info)
    
    def __len__(self):
        return self.tree.n_entries

    def sample(self, batch_size):
        segment = self.tree.total() / batch_size

        obs = { key: [] for key in self.tree.observations }
        next_obs = { key: [] for key in self.tree.next_observations }
        actions = []
        rewards = []
        dones = []
        infos = { key: [] for key in EnvInfo }
        indices = []
        weights = []

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            (idx, p, data_index) = self.tree.get(s)

            if data_index < self.tree.n_entries:
                for key in self.tree.observations:
                    obs[key].append(deepcopy(self.tree.observations[key][data_index]))
                    next_obs[key].append(deepcopy(self.tree.next_observations[key][data_index]))
                actions.append(self.tree.actions[data_index])
                rewards.append(self.tree.rewards[data_index])
                dones.append(self.tree.dones[data_index])
                for key in EnvInfo:
                    infos[key].append(deepcopy(self.tree.infos[key][data_index]))
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
                for key in self.tree.observations:
                    obs[key].append(deepcopy(self.tree.observations[key][rand_idx]))
                    next_obs[key].append(deepcopy(self.tree.next_observations[key][rand_idx]))
                actions.append(self.tree.actions[rand_idx])
                rewards.append(self.tree.rewards[rand_idx])
                dones.append(self.tree.dones[rand_idx])
                for key in EnvInfo:
                    infos[key].append(deepcopy(self.tree.infos[key][rand_idx]))
                treeIdx = rand_idx + self.tree.capacity - 1
                indices.append(treeIdx)
                weights.append(self.tree.tree[treeIdx]) 
            else:
                rand_idx = random.randint(0, len(indices) - 1)
                obs.append(obs[rand_idx].clone().detach())
                next_obs.append(next_obs[rand_idx].clone().detach())
                for key in StateEntry:
                    obs[key].append(deepcopy(obs[key][rand_idx]))
                    next_obs[key].append(deepcopy(next_obs[key][rand_idx]))
                actions.append(actions[rand_idx])
                rewards.append(rewards[rand_idx])
                dones.append(dones[rand_idx])
                for key in EnvInfo:
                    infos[key].append(deepcopy(infos[key][rand_idx]))
                indices.append(indices[rand_idx])
                weights.append(weights[rand_idx])
        
        actions = torch.tensor(actions, device=self.device).unsqueeze(1)
        dones=torch.tensor(dones, device=self.device, dtype=torch.float).unsqueeze(1)
        rewards=torch.tensor(rewards, device=self.device).unsqueeze(1)
        weights = torch.pow(self.tree.n_entries * torch.tensor(weights, device=self.device), -self.beta)
        weights = weights / weights.max()

        return PrioritizedReplayBufferSamples(
            observations=obs, actions=actions,
            next_observations=next_obs, dones=dones,
            rewards=rewards, weights=weights,
            indices=indices, infos=infos)

    def update_beta(self, beta: float):
        self.beta = beta

    def update_weights(self, batch_inds: np.ndarray, weights: np.ndarray):
        for idx, priority in zip(batch_inds, weights.tolist()):
            prio = (priority + self.e) ** self.alpha
            self.max_priority = max(self.max_priority, prio)
            self.tree.update(idx, prio)



class PrioritizedLAPReplayBuffer(PrioritizedReplayBuffer):
    def __init__(self, buffer_size, adapter: SpaceAdapter, device: str = "cpu", alpha: float = 0.6, beta: float = 0.4):
        super().__init__(buffer_size, adapter, device, alpha, beta)

    def update_weights(self, batch_inds: np.ndarray, weights: np.ndarray):
        for idx, priority in zip(batch_inds, weights.tolist()):
            prio = max(priority ** self.alpha, 1.0) # clip samples with low priority to at least 1: LAP algorithm
            self.max_priority = max(self.max_priority, prio)
            self.tree.update(idx, prio)

