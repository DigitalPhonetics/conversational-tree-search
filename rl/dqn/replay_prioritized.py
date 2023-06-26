from copy import deepcopy
import numpy as np
from typing import Any, List
from typing import Dict, NamedTuple
import torch as th
import random
from encoding.state import StateEncoding
from rl.dqn.replay_uniform import UniformReplayBuffer

from rl.utils import EnvInfo, State


class PrioritizedReplayBufferSamples(NamedTuple):
    observations: th.Tensor
    actions: th.Tensor
    next_observations: th.Tensor
    dones: th.Tensor
    rewards: th.Tensor
    weights: np.ndarray
    indices: np.ndarray
    infos: List[Dict[EnvInfo, Any]]


class SumTree:
    def __init__(self, capacity):
        self.write = 0
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.n_entries = 0
       
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
        self.update(idx, p)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0

        if self.n_entries < self.capacity:
            self.n_entries += 1

    # update priority
    def update(self, idx, p):
        change = p - self.tree[idx]
        self.tree[idx] = p
        self._propagate(idx, change)


    # get priority and sample
    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1
        return (idx, self.tree[idx], dataIdx)


class PrioritizedReplayBuffer:
    def __init__(self, buffer_size, state_enc: StateEncoding, device: str = "cpu", alpha: float = 0.6, beta: float = 0.4):
        self.tree = SumTree(buffer_size)
        self.data = UniformReplayBuffer(buffer_size=buffer_size, state_enc=state_enc, device=device)
        self.max_priority = 1.0
        self.action_config = state_enc.action_config
        self.device = device
        self.alpha = alpha
        self.beta = beta
        self.e = (1.0/buffer_size)

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
        if th.is_tensor(next_obs):
            self.data.add(obs, next_obs, action, reward, done, infos, is_aritificial)
            for i in range(len(infos)):
                self.tree.add(self.max_priority)
    
    def __len__(self):
        return self.tree.n_entries

    def sample(self, batch_size):
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
                for key in self.tree.observations:
                    obs.append(self.data.obs[data_index].clone().detach())
                    next_obs.append(self.data.next_obs[data_index].clone().detach())
                actions.append(self.data.action[data_index].clone().detach())
                rewards.append(self.data.reward[data_index].clone().detach())
                dones.append(self.data.dones[data_index].clone().detach())
                infos.append(deepcopy(self.data.infos[data_index]))
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
                    obs.append(self.data.obs[rand_idx].clone().detach())
                    next_obs.append(self.data.obs[rand_idx].clone().detach())
                actions.append(self.data.action[rand_idx].clone().detach())
                rewards.append(self.data.reward[rand_idx].clone().detach())
                dones.append(self.data.dones[rand_idx].clone().detach())
                infos.append(deepcopy(self.data.infos[rand_idx]))
                treeIdx = rand_idx + self.tree.capacity - 1
                indices.append(treeIdx)
                weights.append(self.tree.tree[treeIdx]) 
            else:
                rand_idx = random.randint(0, len(indices) - 1)
                obs.append(obs[rand_idx].clone().detach())
                next_obs.append(next_obs[rand_idx].clone().detach())
                infos.append(deepcopy(self.data.infos[rand_idx]))
                actions.append(actions[rand_idx].clone().detach())
                rewards.append(rewards[rand_idx].clone().detach())
                dones.append(dones[rand_idx].clone().detach())
                indices.append(indices[rand_idx])
                weights.append(weights[rand_idx])
        
        actions = th.cat(actions, 0).unsqueeze(1).to(self.device)
        dones=th.cat(dones, 0).unsqueeze(1).to(self.device)
        rewards=th.cat(rewards, 0).unsqueeze(1).to(self.device)
        weights = th.tensor(weights).to(self.device)
        weights = th.pow(self.tree.n_entries * weights, -self.beta)
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
    def __init__(self, buffer_size, state_enc: StateEncoding, device: str = "cpu", alpha: float = 0.6, beta: float = 0.4):
        super().__init__(buffer_size, state_enc, device, alpha, beta)

    def update_weights(self, batch_inds: np.ndarray, weights: np.ndarray):
        for idx, priority in zip(batch_inds, weights.tolist()):
            prio = max(priority ** self.alpha, 1.0) # clip samples with low priority to at least 1: LAP algorithm
            self.max_priority = max(self.max_priority, prio)
            self.tree.update(idx, prio)

