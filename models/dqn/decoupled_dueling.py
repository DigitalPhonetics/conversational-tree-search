from typing import List

import torch
from models.dqn.base import DQNBase

class DecoupledDuelingDQN(DQNBase):
    def __init__(self, intent_prediction: bool,
                shared_layer_sizes: List[int],
                value_layer_sizes: List[int],
                advantage_layer_sizes: List[int],
                dropout_rate: float,
                normalization_layers: bool,
                activation_fn: torch.nn.Module) -> None:
        super().__init__()

