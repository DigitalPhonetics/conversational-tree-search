from enum import Enum
from typing import Tuple, List, Union
import transformers
transformers.logging.set_verbosity_error()
import torch
from chatbot.adviser.app.encoding.encoding import Encoding
from chatbot.adviser.app.rl.dataset import  DialogNode


class TextEmbeddingPooling(Enum):
    CLS = "CLS"
    MEAN = "mean"
    MAX = "max"
    NONE = "none"
    RNN = "RNN"


RNN_OUTPUT_SIZE = 300


class TextEmbeddings(Encoding):
    def __init__(self, device: str, embedding_dim: int) -> None:
        super().__init__(device)
        self.device = device
        self.embedding_dim = embedding_dim
    
    def get_encoding_dim(self):
        return self.embedding_dim

    def _encode(self, text: str) -> torch.FloatTensor:
        raise NotImplementedError

    def _batch_encode(self, text: List[str]) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        raise NotImplementedError

    
    def batch_encode(self, text: List[str]) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        """
        Returns:
            encodings: batch x max_length x embedding_size
            mask: batch x max_length (mask is 0 where padding occurs) or None, if not applicable
        """
        return self._batch_encode(text)
        


    def encode(self, text: Union[str, None]) -> torch.FloatTensor:
        return self._encode(text=text).detach()

    def embed_node_text(self, node: DialogNode) -> torch.FloatTensor:
        """
        Returns:
            In case of
            * distiluse-base-multilingual-cased: (1, 512)
        """
        return self.encode(node.content.text)



# TODO clear after DB change
class SentenceEmbeddings(TextEmbeddings):
    SIMILARITY_THRESHOLD = 0.01 # TODO find acceptable threshold

    def __init__(self, device: str, pretrained_name: str = 'distiluse-base-multilingual-cased', embedding_dim: int = 512) -> None:
        from sentence_transformers import SentenceTransformer
        super().__init__(device, embedding_dim)
        self.pretrained_name = pretrained_name
        self.bert_sentence_embedder = torch.compile(SentenceTransformer(pretrained_name, device=device, cache_folder = '/mount/arbeitsdaten/asr-2/vaethdk/resources/weights').to(device))

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

    def _batch_encode(self, text: List[str]) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        """
        Returns:
            encodings: batch x 512
            mask: None (we don't need masks here since output is already pooled)
        """
        return self.bert_sentence_embedder.encode(text, convert_to_numpy=False, convert_to_tensor=True, show_progress_bar=False, device=self.device), None


# TODO clear after DB change
class GBertEmbeddings(TextEmbeddings):
    SIMILARITY_THRESHOLD = 0.01 # TODO find acceptable threshold

    def __init__(self, device: str, pretrained_name: str = 'deepset/gbert-large', embedding_dim: int = 1024) -> None:
        from transformers import AutoTokenizer
        from transformers import AutoModelForMaskedLM
        super().__init__(device, embedding_dim)
        self.pretrained_name = pretrained_name
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_name, use_fast=True, cache_dir="/mount/arbeitsdaten/asr-2/vaethdk/resources/weights", truncation_side='left')
        self.bert = torch.compile(AutoModelForMaskedLM.from_pretrained(pretrained_name, cache_dir="/mount/arbeitsdaten/asr-2/vaethdk/resources/weights", output_hidden_states = True).to(device))


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

    def __init__(self, device: str, pretrained_name: str = 'gbert-finetuned', embedding_dim: int = 1024) -> None:
        from transformers import AutoTokenizer
        from transformers import AutoModelForMaskedLM
        super().__init__(device, embedding_dim)
        self.pretrained_name = pretrained_name
        self.tokenizer = AutoTokenizer.from_pretrained('deepset/gbert-large', use_fast=True, cache_dir="/mount/arbeitsdaten/asr-2/vaethdk/resources/weights", truncation_side='left')
        self.bert = torch.compile(AutoModelForMaskedLM.from_pretrained('.models/' + pretrained_name, output_hidden_states = True).to(device))


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

    def _batch_encode(self, text: List[str]) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        """
        Returns:
            encodings: batch x max_length x 1024
            mask: batch x max_length (mask is 0 where padding occurs)
        """
        enc = self.tokenizer(text, add_special_tokens=True, padding=True, return_tensors="pt", truncation=True) # 1st token: CLS, last token: SEP
        return self.bert(**{key: enc[key].to(self.device) for key in enc}).hidden_states[-1]
        
