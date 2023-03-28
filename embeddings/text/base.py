

from dataclasses import dataclass

@dataclass
class TextEmbeddingConfig:
    active: bool
    pooling: str
    noise_std: float
    caching: bool
    ckpt_name: str
    embedding_dim: int
    cache_db_index: int
