

from dataclasses import dataclass
from enum import Enum
from typing import List, Tuple, Union

import torch

from encoding.base import Encoding
from data.dataset import DialogNode


class TextEmbeddingPooling(Enum):
    CLS = "CLS"
    MEAN = "mean"
    MAX = "max"
    NONE = "none"


@dataclass
class TextEmbeddingConfig:
    active: bool
    pooling: TextEmbeddingPooling
    noise_std: float
    caching: bool
    ckpt_name: str
    embedding_dim: int
    _target_: str



class TextEmbeddings(Encoding):
    def __init__(self, device: str, ckpt_name: str, embedding_dim: int, torch_compile: bool) -> None:
        super().__init__(device)
        self.device = device
        self.embedding_dim = embedding_dim
        self.ckpt_name = ckpt_name
    
    def get_encoding_dim(self):
        return self.embedding_dim

    @torch.no_grad()
    def _encode(self, text: str) -> torch.FloatTensor:
        raise NotImplementedError

    @torch.no_grad()
    def _batch_encode(self, text: List[str]) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        raise NotImplementedError

    @torch.no_grad()
    def batch_encode(self, text: List[str]) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        """
        Returns:
            encodings: batch x max_length x embedding_size
            mask: batch x max_length (mask is 0 where padding occurs) or None, if not applicable
        """
        return self._batch_encode(text)

    @torch.no_grad()
    def encode(self, text: Union[str, None]) -> torch.FloatTensor:
        return self._encode(text=text).detach()

    @torch.no_grad()
    def embed_node_text(self, node: DialogNode) -> torch.FloatTensor:
        """
        Returns:
            In case of
            * distiluse-base-multilingual-cased: (1, 512)
        """
        return self.encode(node.content.text)
