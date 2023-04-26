from encoding.base import Encoding

from typing import List 
import torch
import torch.nn.functional as F

from config import ActionType


class ActionTypeEncoding(Encoding):
    def __init__(self, device: str) -> None:
        super().__init__(device)
        self.device = device

    def get_encoding_dim(self) -> int:
        return 2 # question = 0, answer = 1

    @torch.no_grad()
    def encode(self, action: int) -> torch.FloatTensor:
        return F.one_hot(torch.tensor([action > ActionType.ASK], dtype=torch.long, device=self.device), num_classes=self.get_encoding_dim())

    @torch.no_grad()
    def batch_encode(self, actions: List[int]) -> torch.FloatTensor:
        """ 
        Returns:
            intent class (one-hot encoded): batch x 2
        """
        return F.one_hot(torch.tensor([[action > ActionType.ASK] for action in actions], dtype=torch.long, device=self.device), num_classes=self.get_encoding_dim())
        

