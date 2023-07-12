from chatbot.adviser.app.encoding.encoding import Encoding

from typing import List 
import torch
import torch.nn.functional as F

from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification

class IntentEncoding(Encoding):
    def __init__(self, device: str, ckpt_dir: str = './.models/intentpredictor') -> None:
        super().__init__(device)
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained('deepset/gbert-large', use_fast=True, cache_dir=".models/gbert", truncation_side='left')
        self.model = AutoModelForSequenceClassification.from_pretrained(ckpt_dir, output_hidden_states = True).to(device)

    def get_encoding_dim(self) -> int:
        return 2 # question = 0, answer = 1

    def encode(self, dialog_node_text: str, current_user_utterance: str, **kwargs) -> torch.FloatTensor:
        tok = self.tokenizer(text=dialog_node_text, text_pair=current_user_utterance, truncation=True, return_tensors="pt")
        tok = {key: tok[key].to(self.device) for key in tok}
        class_idx = self.model(**tok).logits.argmax(-1).item()
        return F.one_hot(torch.tensor([class_idx], dtype=torch.long, device=self.device), num_classes=2)

    def batch_encode(self, dialog_node_text: List[str], current_user_utterance: List[str], **kwargs) -> torch.FloatTensor:
        """ 
        Returns:
            intent class (one-hot encoded): batch x 2
        """
        tok = self.tokenizer(text=dialog_node_text, text_pair=current_user_utterance, padding=True, truncation=True, return_tensors="pt")
        tok = {key: tok[key].to(self.device) for key in tok}
        class_idx = self.model(**tok).logits.argmax(-1) # batch
        return F.one_hot(torch.tensor(class_idx, dtype=torch.long, device=self.device), num_classes=2) # batch x 2
        

