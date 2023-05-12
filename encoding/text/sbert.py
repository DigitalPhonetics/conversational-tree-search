from typing import Tuple, List, Union
import os

import transformers

from encoding.text.base import TextEmbeddings
transformers.logging.set_verbosity_error()
import torch



class SentenceEmbeddings(TextEmbeddings):
    SIMILARITY_THRESHOLD = 0.01 # TODO find acceptable threshold

    def __init__(self, device: str, ckpt_name: str, embedding_dim: int) -> None:
        from sentence_transformers import SentenceTransformer
        super().__init__(device, ckpt_name, embedding_dim)
        path = f".models/{ckpt_name.replace('/', '_')}"
        name_or_path = path if os.path.exists(path) else ckpt_name
        self.bert_sentence_embedder = torch.compile(SentenceTransformer(path, device=device, cache_folder = '.models').to(device))

    @torch.no_grad()
    def _encode(self, text: Union[str, None]) -> torch.FloatTensor:
        """
        Returns:
            In case of
            * distiluse-base-multilingual-cased: (1, 512)
        """
        if text:
            return self.bert_sentence_embedder.encode(text, convert_to_numpy=False, convert_to_tensor=True, show_progress_bar=False, device=self.device).unsqueeze(0).unsqueeze(1)
        else:
            return torch.zeros(1, 1, self.embedding_dim, dtype=torch.float, device=self.device)

    @torch.no_grad()
    def _batch_encode(self, text: List[str]) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        """
        Returns:
            encodings: batch x 512
            mask: None (we don't need masks here since output is already pooled)
        """
        return self.bert_sentence_embedder.encode(text, convert_to_numpy=False, convert_to_tensor=True, show_progress_bar=False, device=self.device)

