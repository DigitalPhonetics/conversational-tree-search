
import random
from typing import List, Tuple, Union
from config import INSTANCES, InstanceType
from algorithm.algorithm import Algorithm

import torch
from torch.nn.utils.rnn import pad_packed_sequence


# TODO add q value clipping here!
# TODO use NetworkOutputs here!
# TODO add get_action_masks() method to some utils class
# - should we remove mask input from 'select_actions_eps_greedy' and mask in the algorithm class instead?


@torch.no_grad()
def pad_and_mask(self, state_batch: Union[torch.FloatTensor, torch.nn.utils.rnn.PackedSequence]) -> Tuple[torch.FloatTensor, torch.LongTensor, torch.FloatTensor]:
    """
    Returns:
        padded_sequence: padded state_batch tensor (batch x max_actions x state_dim)
        num_actions: number of actions per batch item (batch)
        mask: mask for outputs where num_actions for batch item was smaller than max_actions (batch x max_actions)
    """
    if isinstance(state_batch, torch.nn.utils.rnn.PackedSequence):
        padded_sequence, num_actions = pad_packed_sequence(state_batch, batch_first=True) # batch x max_actions x state_dim,  batch
    else:
        assert state_batch.ndim == 2, "can't forward batched state in non-packed form"
        padded_sequence = state_batch.unsqueeze(0) # 1 x actions x embedding_dim
        num_actions = torch.tensor([padded_sequence.size(1)], dtype=torch.long)
    num_actions = num_actions.to(padded_sequence.device)
    max_actions = padded_sequence.size(1)
    mask = ~(torch.arange(max_actions, device=padded_sequence.device)[None, :] < num_actions[:, None]) # batch x max_actions (negate mask!)

    return padded_sequence, num_actions, mask




@torch.no_grad()
def select_actions_eps_greedy(self, training: bool, q_values: torch.FloatTensor, action_mask: Union[torch.FloatTensor, None], epsilon: float) -> torch.LongTensor:
    """ Epsilon-greedy policy.

    Args:
        node_keys: current node keys
        state_vectors: current state (dimension batch x state_dim    or num_actions x state_dim)
        epsilon: current scheduled exploration rate

    Returns:
        List of action indices for action selected by the agent for the current states
        List of predicted intent classes if supported by model, else None
    """
    # TODO mask STOP action in first turn?

    # epsilon greedy exploration
    if training and random.random() < epsilon:
        # exploration
        if torch.is_tensor(action_mask):
            # only choose between allowed actions after masking
            allowed_action_indices = (~action_mask).float() # batch x num_max_actions
            next_action_indices = [random.choice(allowed_action_indices[batch_idx].nonzero().view(-1).tolist()) for batch_idx in range(allowed_action_indices.size(0))] # 1-dim list with one valid action index per batch item
        else:
            # no masking: random choice
            next_action_indices = [random.randint(0, self.adapter.num_actions - 1) for _ in range(q_values.size(0))]
    else:
        # exploitation
        final_values = q_values
        if torch.is_tensor(action_mask):
            final_values = torch.masked_fill(q_values, action_mask[:,:q_values.size(-1)], float('-inf'))
        next_action_indices = final_values.argmax(-1).tolist()

    return next_action_indices


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
        slope = (end_e - start_e) / duration
        return max(slope * t + start_e, end_e)

class DQNAlgorithm(Algorithm):
    def __init__(self, buffer, targets,
        timesteps_per_reset: int,
        reset_exploration_times: int,
        max_grad_norm: float,
        batch_size: int,
        gamma: float,
        exploration_fraction: float,
        eps_start: float,
        eps_end: float,
        train_frequency: int,
        warmup_turns: int,
        target_network_update_frequency: int,
        q_value_clipping: float
        ) -> None:
        global INSTANCES
        INSTANCES[InstanceType.ALGORITHM] = self
        print("DQN Trainer")

        self.eps_start = eps_start
        self.eps_end = eps_end
        self.exploration_fraction = exploration_fraction
        self.timesteps_per_reset = timesteps_per_reset


    def run_single_timestep(self, engine, timestep: int):
        """
        Args:
            timestep: GLOBAL timestep (not step in current episode)
        """
        epsilon = linear_schedule(start_e=self.eps_start, end_e=self.eps_end, duration=self.exploration_fraction * self.timesteps_per_reset, t=timestep % self.timesteps_per_reset)

        # TODO get observation as vector
        # TODO select actions based on eps-greedy
        # TODO call environment step() using action
        # TODO get next observation as vector
        # TODO store experience in replay buffer