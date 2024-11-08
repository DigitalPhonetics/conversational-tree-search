from typing import Tuple, List, Union
import os

import transformers

from encoding.text.base import TextEmbeddings
transformers.logging.set_verbosity_error()
import torch



class SentenceEmbeddings(TextEmbeddings):
    SIMILARITY_THRESHOLD = 0.01 # TODO find acceptable threshold

    def __init__(self, device: str, ckpt_name: str, embedding_dim: int, torch_compile: bool) -> None:
        from sentence_transformers import SentenceTransformer
        super().__init__(device, ckpt_name, embedding_dim, torch_compile)
        self.bert_sentence_embedder = SentenceTransformer(ckpt_name, device=device, cache_folder = "/mount/arbeitsdaten/asr-2/vaethdk/resources/weights").to(device)
        if torch_compile:
            print("COMPILE EMBEDDING=TRUE")
            self.bert_sentence_embedder = torch.compile(self.bert_sentence_embedder)
        print("COMPILE EMBEDDING=FALSE")

    @torch.no_grad()
    def _encode(self, text: Union[str, None], normalize_embeddings: bool, query_instruction: Union[str, None] = None) -> torch.FloatTensor:
        """
        Returns:
            In case of
            * distiluse-base-multilingual-cased: (1, 512)
        """
        if text:
            return self.bert_sentence_embedder.encode(text, convert_to_numpy=False, convert_to_tensor=True, show_progress_bar=False, normalize_embeddings=normalize_embeddings, device=self.device).unsqueeze(0).unsqueeze(1)
        else:
            return torch.zeros(1, 1, self.embedding_dim, dtype=torch.float, device=self.device)

    @torch.no_grad()
    def _batch_encode(self, text: List[str], normalize_embeddings: bool, query_instruction: Union[str, None] = None) -> Tuple[torch.FloatTensor]:
        """
        Returns:
            encodings: batch x 512
            mask: None (we don't need masks here since output is already pooled)
        """
        return self.bert_sentence_embedder.encode(text, normalize_embeddings=normalize_embeddings, convert_to_numpy=False, convert_to_tensor=True, show_progress_bar=False, device=self.device)

