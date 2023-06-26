
from typing import List, Tuple, Union
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical

from torch.nn.utils.weight_norm import weight_norm
from torch.nn.utils.rnn import pad_packed_sequence

from encoding.state import StateEncoding

# def parse_args():
#     # fmt: off
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--total-timesteps", type=int, default=500000,
#         help="total timesteps of the experiments")
#     parser.add_argument("--learning-rate", type=float, default=2.5e-4,
#         help="the learning rate of the optimizer")
#     parser.add_argument("--num-steps", type=int, default=128,
#         help="the number of steps to run in each environment per policy rollout")
#     parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
#         help="Toggle learning rate annealing for policy and value networks")
#     parser.add_argument("--gae", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
#         help="Use GAE for advantage computation")
#     parser.add_argument("--gamma", type=float, default=0.99,
#         help="the discount factor gamma")
#     parser.add_argument("--gae-lambda", type=float, default=0.95,
#         help="the lambda for the general advantage estimation")
#     parser.add_argument("--num-minibatches", type=int, default=4,
#         help="the number of mini-batches")
#     parser.add_argument("--update-epochs", type=int, default=4,
#         help="the K epochs to update the policy")
#     parser.add_argument("--norm-adv", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
#         help="Toggles advantages normalization")
#     parser.add_argument("--clip-coef", type=float, default=0.2,
#         help="the surrogate clipping coefficient")
#     parser.add_argument("--clip-vloss", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
#         help="Toggles whether or not to use a clipped loss for the value function, as per the paper.")
#     parser.add_argument("--ent-coef", type=float, default=0.01,
#         help="coefficient of the entropy")
#     parser.add_argument("--vf-coef", type=float, default=0.5,
#         help="coefficient of the value function")
#     parser.add_argument("--max-grad-norm", type=float, default=0.5,
#         help="the maximum norm for the gradient clipping")
#     parser.add_argument("--target-kl", type=float, default=None,
#         help="the target KL divergence threshold")
#     args = parser.parse_args()
#     args.batch_size = int(args.num_envs * args.num_steps)
#     args.minibatch_size = int(args.batch_size // args.num_minibatches)
#     # fmt: on
#     return args



def _add_weight_normalization(layer: torch.nn.Module, normalize: bool):
    if normalize:
        return weight_norm(layer, dim=None)
    return layer


class PPONetwork(nn.Module):
    def __init__(self, state_enc: StateEncoding,
                       hidden_layer_sizes: List[int] = [1024, 1024, 512],
                       activation_fn = nn.ReLU,
                       dropout_rate: float = 0.0,
                       normalization_layers: bool = False,
                       q_value_clipping: float = 0.0) -> None:
        super().__init__()
        self.state_enc = state_enc
        self.q_value_clipping = q_value_clipping # actor q value clipping

        # Actor-Critic
        self.critic = nn.ModuleList()
        self.actor = nn.ModuleList()
        current_input_dim = state_enc.space_dims.state_vector
        for layer_size in hidden_layer_sizes:
            # critic
            self.critic.append(_add_weight_normalization(nn.Linear(current_input_dim, layer_size), normalization_layers))
            self.critic.append(activation_fn())
            if dropout_rate > 0.0:
                self.critic.append(nn.Dropout(p=dropout_rate))

            # actor
            self.actor.append(_add_weight_normalization(nn.Linear(current_input_dim, layer_size), normalization_layers))
            self.actor.append(activation_fn())
            if dropout_rate > 0.0:
                self.actor.append(nn.Dropout(p=dropout_rate))
            current_input_dim = layer_size
        # output layers
        self.critic.append(nn.Linear(current_input_dim, 1))
        self.actor.append(nn.Linear(current_input_dim, state_enc.space_dims.action_vector))
    
 
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

    def _forward_actor(self, state_batch: Union[torch.FloatTensor, torch.nn.utils.rnn.PackedSequence]):
        """
        Returns: action logits (batch x action_dim)
        """
        output = state_batch
        if self.adapter.configuration.action_config == self.state_enc.action_config.in_state_space:
            padded_sequence, num_actions, mask = self.pad_and_mask(output)
            output = padded_sequence  #  batch x max_actions x state_dim 
            for layer in self.actor:
                output = layer(output)
                output = torch.masked_fill(output, mask.unsqueeze(2), 0.0)
            # output: batch x max_actions x action_dim = 1
            output = output.squeeze(2) # batch x max_actions

            # mask off outputs for inputs with less actions than max_actions
            output = torch.masked_fill(output, mask, float('-inf')) # batch x max_actions  (mask Q values)
        else:
            for layer in self.actor:
                output = layer(output)
        
        return output

    def _forward_critic(self, state_batch: Union[torch.FloatTensor, torch.nn.utils.rnn.PackedSequence]):
        """
        Returns: state values (batch x 1)
        """
        # what to output for critic in case of actions in state space? maybe average critic predictions across all
        # OR is there a possiblity about cutting out the action-specific parts from the state space? 
        # => 1 super easy option: only take 1st input (which is always for action STOP) and model will never learn what the action-encoding part in the state space means :) 
        output = state_batch
        if self.state_enc.action_config.in_state_space:
            padded_sequence, num_actions, mask = self.pad_and_mask(state_batch)
            output = padded_sequence[:, 0, :] # batch x max_actions x state_dim => batch x state_dim
        for layer in self.critic:
            output = layer(output)
    
        # q-value clipping
        if self.q_value_clipping > 0.0:
            output = output.clip(min=None, max=self.q_value_clipping)
        return output
    
    def get_value(self, x, node_ids: List[int] = None):
        return self._forward_critic(x)

    @torch.no_grad()
    def select_action(self, state_batch, node_ids: List[int] = None) -> List[int]:
        logits = self._forward_actor(state_batch)
        if self.adapter.configuration.action_masking and node_ids:
            logits = torch.masked_fill(logits, self.adapter.get_action_masks(node_keys=node_ids)[:,:logits.size(-1)], float('-inf'))
        return Categorical(logits=logits).sample().tolist()

    def get_actions_and_values(self, state_batch: Union[torch.FloatTensor, torch.nn.utils.rnn.PackedSequence], actions=None, node_ids: List[int] = None):
        logits = self._forward_actor(state_batch) # batch x action_dim

        if self.adapter.configuration.action_masking and node_ids:
            logits = torch.masked_fill(logits, self.adapter.get_action_masks(node_keys=node_ids)[:,:logits.size(-1)], float('-inf'))

        probs = Categorical(logits=logits)
        actions = probs.sample() if actions is None else actions.clone().detach().to(logits.device) # batch_size
        return actions, probs.log_prob(actions), probs.entropy(), self._forward_critic(state_batch)

