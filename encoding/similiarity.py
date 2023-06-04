from typing import List, Tuple
import torch
from data.dataset import GraphDataset, DialogNode
from utils.utils import EMBEDDINGS
import redisai as rai
import redis
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
from torch.nn.utils.rnn import pad_sequence


class AnswerSimilarityEncoding:
    def __init__(self, device: str, model_name: str, dialog_tree: GraphDataset, caching: bool) -> None:
        self.device = device
        self.encoding_dim = dialog_tree.get_max_node_degree()
        self.similarity_model = SentenceTransformer(model_name, device=device, cache_folder = '.models')

    def get_encoding_dim(self) -> int:
        # NOTE only valid for actions in state space
        return 1 # only one similarity per action  #self.encoding_dim

    @torch.no_grad()
    def encode(self, current_user_utterance: str, dialog_node: DialogNode, noise: float=0.0) -> torch.FloatTensor:
        # num_answers = dialog_node.answers.count()
        num_answers = len(dialog_node.answers)
        if not current_user_utterance:
            # nothing to compare -> no similarities
            return torch.zeros(1, num_answers, 1)
        if num_answers == 0:
            return torch.zeros(1, 1, 1)
        elif num_answers == 1:
            # only one option - always most similar
            return torch.ones(1, 1, 1)

        utterance_emb = self._embed_text(current_user_utterance)
        if noise > 0.0:
            utterance_emb = torch.normal(mean=utterance_emb, std=noise*torch.abs(utterance_emb))

        # answer_embs = torch.cat([self._embed_text(answer.content.text, cache_prefix="a_").unsqueeze(0) for answer in dialog_node.answers.all().order_by("answer_index")], 0) # answers x 512
        answer_embs = torch.cat([self._embed_text(answer.text, cache_prefix="a_") for answer in dialog_node.answers], 0) # answers x 512
        embedding = cos_sim(utterance_emb, answer_embs)[0].flatten() # num_answers
        return embedding.unsqueeze(0).unsqueeze(-1) # 1 x max_actions x 1

    # @torch.no_grad()
    # def batch_encode(self, current_user_utterance: List[str], dialog_node: List[DialogNode], noise: float = 0.0) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
    #     """
    #     Returns:
    #         similaries: batch x max_actions
    #         mask: batch x max_actions (0 where padded)
    #     """
    #     embeddings = self.similarity_model.encode(current_user_utterance, convert_to_tensor=True, show_progress_bar=False) # batch x 512
    #     if noise > 0.0:
    #         embeddings = torch.normal(mean=embeddings, std=noise*torch.abs(embeddings))
    #     # answer_embeddings = [self.similarity_model.encode([answer.content.text for answer in node.answers.all().order_by("answer_index")], convert_to_tensor=True, show_progress_bar=False) for node in dialog_node] # List (batch), entry: answers x 512
    #     answer_embeddings = [self.similarity_model.encode([answer.content.text for answer in node.answers], convert_to_tensor=True, show_progress_bar=False) for node in dialog_node] # List (batch), entry: answers x 512
    #     mask = pad_sequence([~(answer_emb.abs().sum(-1) == 0) for answer_emb in answer_embeddings], batch_first=True) # batch x max_actions
    #     similarities = pad_sequence([cos_sim(enc, answer_embeddings[i])[0] for i, enc in enumerate(embeddings)]) # batch x max_actions
    #     return similarities, mask


    # PREFIXES: a_ for action text
    @torch.no_grad()
    def _embed_text(self, text: str, cache_prefix=""):
        embeddings = None
        cache_key = f"{cache_prefix}{text}"

        # encoding
        embeddings = self.similarity_model.encode(text if text else "", convert_to_tensor=True, show_progress_bar=False, device=self.device).unsqueeze(0).unsqueeze(1)
            
        # result
        return embeddings.squeeze(0).cpu()

   
