from typing import List

import torch


class DQN:
    def __init__(self, intent_prediction: bool,
                    hidden_layer_sizes: List[int],
                    dropout_rate: float,
                    normalization_layers: bool,
                    activation_fn: torch.nn.Module) -> None:
        print("DQN")