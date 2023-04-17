import torch

from chatbot.adviser.app.rl.layers.attention.attention import Attention
import models.masked_softmax as util


class CosineAttention(Attention):
    """
    Computes attention between a vector and a matrix using cosine similarity.

    Registered as an `Attention` with name "cosine".
    """

    def _forward_internal(self, vector: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
        a_norm = vector / (
            vector.norm(p=2, dim=-1, keepdim=True) + util.tiny_value_of_dtype(vector.dtype)
        )
        b_norm = matrix / (
            matrix.norm(p=2, dim=-1, keepdim=True) + util.tiny_value_of_dtype(matrix.dtype)
        )
        return torch.bmm(a_norm.unsqueeze(dim=1), b_norm.transpose(-1, -2)).squeeze(1)
