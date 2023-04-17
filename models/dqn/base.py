
from dataclasses import dataclass
import random
from typing import List, Optional, Tuple, Union
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence



@dataclass
class NetworkOutput:
    logits: torch.FloatTensor
    intent_logits: Optional[torch.FloatTensor]


class DQNBase(nn.Module):
    def __init__(self) -> None:
        """
        Args:
            attention_fns: mapping from input (str) to: {attention_meachanism: str, active: bool, activation: str}
        """
        super().__init__()
    
