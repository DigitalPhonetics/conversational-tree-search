from typing import Any, Dict
import torch
from torch.nn.functional import softmax
from algorithm.dqn.buffer import CustomReplayBufferSamples

class DQNTarget:
    def __init__(self, gamma: float, **kwargs) -> None:
        self.gamma = gamma

    @torch.no_grad()
    def target(self, next_q_values: torch.FloatTensor, data: CustomReplayBufferSamples, q: torch.FloatTensor) -> torch.FloatTensor:
        raise NotImplementedError


class StandardTarget(DQNTarget):
    @torch.no_grad()
    def target(self, next_q_values: torch.FloatTensor, data: CustomReplayBufferSamples, q: torch.FloatTensor) -> torch.FloatTensor:
        # Compute the next Q-values using the target network
        # Follow greedy policy: use the one with the highest value
        next_q_values, _ = next_q_values.max(dim=1)
        # Avoid potential broadcast issue
        next_q_values = next_q_values.reshape(-1, 1)
        # 1-step TD target
        target_q_values = data.rewards + (1 - data.dones) * self.gamma * next_q_values
        return target_q_values

class MuenchausenTarget(DQNTarget):
    def __init__(self, gamma: float, tau: float, alpha: float, clipping: float, **kwargs) -> None:
        super().__init__(gamma=gamma)
        self.tau = tau
        self.alpha = alpha
        self.clipping = clipping

    def _munchausen_stable_logsoftmax(self, q: torch.FloatTensor, tau: float) -> torch.FloatTensor:
        v = q.max(-1, keepdim=True)[0]
        tau_lse = v + tau * torch.log(
            torch.sum(
                torch.exp((q - v)/tau), dim=-1, keepdim=True
            )
        )
        return q - tau_lse # batch x 1

    # def _munchausen_stable_softmax(self, q: torch.FloatTensor, tau: float) -> torch.FloatTensor:
    #     return torch.softmax((q-q.max(-1, keepdim=True)[0])/tau, -1) # batch

    @torch.no_grad()
    def target(self, next_q_values: torch.FloatTensor, data: CustomReplayBufferSamples, q_old: torch.FloatTensor) -> torch.FloatTensor:
        # Compute the next Q-values using the target network
        mask = next_q_values > float('-inf')
        sum_term = softmax(next_q_values / self.tau, dim=-1) * (next_q_values - self._munchausen_stable_logsoftmax(next_q_values, self.tau)) # batch x actions
        log_policy = self._munchausen_stable_logsoftmax(q_old, self.tau).gather(-1, data.actions).view(-1) # batch x actions -> batch
        if self.clipping != 0:
            log_policy = torch.clip(log_policy, min=self.clipping, max=0)
        return data.rewards.flatten() + self.alpha*log_policy + self.gamma * sum_term.masked_fill(~mask, 0.0).sum(-1) * (1.0 - data.dones.flatten())


class TsallisMuenchausenTarget(DQNTarget):
    def __init__(self, gamma: float, tau: float, alpha: float, q_star: float = 2.0, **kwargs) -> None:
        super().__init__(gamma=gamma)
        self.tau = tau
        self.alpha = alpha
        self.q_star = q_star

    def _munchausen_stable_logsoftmax(self, q: torch.FloatTensor, tau: float) -> torch.FloatTensor:
        v = q.max(-1, keepdim=True)[0]
        tau_lse = v + tau * torch.log(
            torch.sum(
                torch.exp((q - v)/tau), dim=-1, keepdim=True
            )
        )
        return q - tau_lse # batch x 1
    
    def _normalize(q):
        # calculate sparsemax normalization
        next_q_sorted = q.sort(dim=-1, descending=True)[0] # batch x actions
        sum_ = torch.zeros(q.size(0), device=q.device) # batch
        S_size = torch.zeros(q.size(0), device=q.device) # batch
        for i in range(next_q_sorted.size(-1)):
            # loop over all actions
            sum_so_far = next_q_sorted[:, :i].sum(-1) # batch
            S_j = 1 + (i+1)*next_q_sorted[:, i] > sum_so_far # batch, evaluate condition
            S_size += S_j # batch, count actions that satisfy condition (denominator)
            sum_ += S_j * next_q_sorted[:, i] # batch, accumulate nominator (only for actions satisfying condition)
        return torch.clip(q - (sum_-1)/S_size, min=0)

    # def _munchausen_stable_softmax(self, q: torch.FloatTensor, tau: float) -> torch.FloatTensor:
    #     return torch.softmax((q-q.max(-1, keepdim=True)[0])/tau, -1) # batch

    @torch.no_grad()
    def target(self, next_q_values: torch.FloatTensor, data: CustomReplayBufferSamples, q_old: torch.FloatTensor) -> torch.FloatTensor:
        # Compute the next Q-values using the target network
        mask = next_q_values > float('-inf')
        
        # calculate policy 
        scaled_q = next_q_values / self.tau
        pi = self._normalize(scaled_q)
        # DEBUGGING
        debug = pi.sum(-1)
        print(debug)

        log_policy = (torch.pow(pi, self.q_star - 1) - 1) / (self.q_star - 1)