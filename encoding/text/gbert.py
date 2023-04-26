from typing import Tuple, List, Union
import transformers

from encoding.text.base import TextEmbeddings
transformers.logging.set_verbosity_error()
import torch

class GBertEmbeddings(TextEmbeddings):
    SIMILARITY_THRESHOLD = 0.01 # TODO find acceptable threshold

    def __init__(self, device: str, ckpt_name: str, embedding_dim: int) -> None:
        from transformers import AutoTokenizer
        from transformers import AutoModelForMaskedLM
        super().__init__(device, ckpt_name, embedding_dim)
        self.tokenizer = AutoTokenizer.from_pretrained(ckpt_name, use_fast=True, cache_dir=".models/gbert", truncation_side='left')
        self.bert = AutoModelForMaskedLM.from_pretrained(ckpt_name, cache_dir=".models/gbert-tokenizer", output_hidden_states = True).to(device)


    @torch.no_grad()
    def _encode(self, text: Union[str, None]) -> torch.FloatTensor:
        """
        Returns:
            In case of
            * distiluse-base-multilingual-cased: (1, 512)
        """
        if text:
            enc = self.tokenizer(text, add_special_tokens=True, padding=True, return_tensors="pt", truncation=True) # 1st token: CLS, last token: SEP
            return self.bert(**{key: enc[key].to(self.device) for key in enc}).hidden_states[-1]
        else:
            return torch.zeros(1, 1, self.embedding_dim, dtype=torch.float, device=self.device)

    @torch.no_grad()
    def _batch_encode(self, text: List[str]) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        """
        Returns:
            encodings: batch x max_length x 1024
            mask: batch x max_length (mask is 0 where padding occurs)
        """
        enc = self.tokenizer(text, add_special_tokens=True, padding=True, return_tensors="pt", truncation=True) # 1st token: CLS, last token: SEP
        return self.bert(**{key: enc[key].to(self.device) for key in enc}).hidden_states[-1]
        


class FinetunedGBertEmbeddings(TextEmbeddings):
    SIMILARITY_THRESHOLD = 0.01 # TODO find acceptable threshold

    def __init__(self, device: str, ckpt_name: str, embedding_dim: int) -> None:
        from transformers import AutoTokenizer
        from transformers import AutoModelForMaskedLM
        super().__init__(device, ckpt_name, embedding_dim)
        self.tokenizer = AutoTokenizer.from_pretrained('deepset/gbert-large', use_fast=True, cache_dir=".models/gbert", truncation_side='left')
        self.bert = AutoModelForMaskedLM.from_pretrained('.models/' + ckpt_name, output_hidden_states = True).to(device)


    @torch.no_grad()
    def _encode(self, text: Union[str, None]) -> torch.FloatTensor:
        """
        Returns:
            In case of
            * distiluse-base-multilingual-cased: (1, 512)
        """
        if text:
            enc = self.tokenizer(text, add_special_tokens=True, return_tensors="pt", truncation=True) # 1st token: CLS, last token: SEP
            return self.bert(**{key: enc[key].to(self.device) for key in enc}).hidden_states[-1]
        else:
            return torch.zeros(1, 1, self.embedding_dim, dtype=torch.float, device=self.device)

    @torch.no_grad()
    def _batch_encode(self, text: List[str]) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        """
        Returns:
            encodings: batch x max_length x 1024
            mask: batch x max_length (mask is 0 where padding occurs)
        """
        enc = self.tokenizer(text, add_special_tokens=True, padding=True, return_tensors="pt", truncation=True) # 1st token: CLS, last token: SEP
        return self.bert(**{key: enc[key].to(self.device) for key in enc}).hidden_states[-1]
        
