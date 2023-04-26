from typing import Union, Tuple

import torch

class Encoding:
    def __init__(self, device: str) -> None:
        self.device = device

    def get_encoding_dim(self) -> int:
        raise NotImplementedError

    def encode(self, **kwargs) -> torch.FloatTensor:
        raise NotImplementedError

    def batch_encode(self, **kwargs) -> torch.FloatTensor:
        raise NotImplementedError
