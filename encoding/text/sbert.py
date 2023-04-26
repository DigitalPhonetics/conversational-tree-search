from typing import Tuple, List, Union
import transformers

from encoding.text.base import TextEmbeddings
transformers.logging.set_verbosity_error()
import torch



class SentenceEmbeddings(TextEmbeddings):
    SIMILARITY_THRESHOLD = 0.01 # TODO find acceptable threshold

    def __init__(self, device: str, ckpt_name: str, embedding_dim: int) -> None:
        from sentence_transformers import SentenceTransformer
        super().__init__(device, ckpt_name, embedding_dim)
        self.bert_sentence_embedder = SentenceTransformer(ckpt_name, device=device, cache_folder = '.models')

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

