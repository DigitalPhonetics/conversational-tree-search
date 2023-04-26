from typing import List, Tuple
import torch
from chatbot.adviser.app.rl.dialogtree import DialogTree
from chatbot.adviser.app.rl.utils import EMBEDDINGS
from chatbot.adviser.app.rl.dataset import DialogNode
import redisai as rai
import redis
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
from torch.nn.utils.rnn import pad_sequence


class AnswerSimilarityEncoding:
    def __init__(self, device: str, model_name: str, dialog_tree: DialogTree, caching: bool) -> None:
        self.device = device
        self.encoding_dim = dialog_tree.get_max_node_degree()
        self.caching = caching
        self.cache_connection = rai.Client(host='localhost', port=64123, db=3) if caching else None
        self.similarity_model = SentenceTransformer(model_name, device=device, cache_folder = '.models')

    def get_encoding_dim(self) -> int:
        # NOTE only valid for actions in state space
        return 1 # only one similarity per action  #self.encoding_dim

    @torch.no_grad()
    def encode(self, current_user_utterance: str, dialog_node: DialogNode, noise: float=0.0) -> torch.FloatTensor:
        # num_answers = dialog_node.answers.count()
        num_answers = dialog_node.answer_count()
        if not current_user_utterance:
            # nothing to compare -> no similarities
            return torch.zeros(1, num_answers, 1, device=self.device)
        if num_answers == 0:
            return torch.zeros(1, 1, 1, device=self.device)
        elif num_answers == 1:
            # only one option - always most similar
            return torch.ones(1, 1, 1, device=self.device)

        utterance_emb = self._embed_text(current_user_utterance)
        if noise > 0.0:
            utterance_emb = torch.normal(mean=utterance_emb, std=noise*torch.abs(utterance_emb))

        # answer_embs = torch.cat([self._embed_text(answer.content.text, cache_prefix="a_").unsqueeze(0) for answer in dialog_node.answers.all().order_by("answer_index")], 0) # answers x 512
        answer_embs = torch.cat([self._embed_text(answer.content.text, cache_prefix="a_") for answer in dialog_node.answers], 0) # answers x 512
        embedding = cos_sim(utterance_emb, answer_embs)[0].flatten() # num_answers
        return embedding.unsqueeze(0).unsqueeze(-1) # 1 x max_actions x 1

    @torch.no_grad()
    def batch_encode(self, current_user_utterance: List[str], dialog_node: List[DialogNode], noise: float = 0.0) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        """
        Returns:
            similaries: batch x max_actions
            mask: batch x max_actions (0 where padded)
        """
        embeddings = self.similarity_model.encode(current_user_utterance, convert_to_tensor=True, show_progress_bar=False) # batch x 512
        if noise > 0.0:
            embeddings = torch.normal(mean=embeddings, std=noise*torch.abs(embeddings))
        # answer_embeddings = [self.similarity_model.encode([answer.content.text for answer in node.answers.all().order_by("answer_index")], convert_to_tensor=True, show_progress_bar=False) for node in dialog_node] # List (batch), entry: answers x 512
        answer_embeddings = [self.similarity_model.encode([answer.content.text for answer in node.answers], convert_to_tensor=True, show_progress_bar=False) for node in dialog_node] # List (batch), entry: answers x 512
        mask = pad_sequence([~(answer_emb.abs().sum(-1) == 0) for answer_emb in answer_embeddings], batch_first=True) # batch x max_actions
        similarities = pad_sequence([cos_sim(enc, answer_embeddings[i])[0] for i, enc in enumerate(embeddings)]) # batch x max_actions
        return similarities, mask


    # PREFIXES: a_ for action text
    @torch.no_grad()
    def _embed_text(self, text: str, cache_prefix=""):
        embeddings = None
        cache_key = f"{cache_prefix}{text}"

        # encoding and caching
        if self.caching and text and (not text.isnumeric()):
            try:
                embeddings = torch.tensor(self.cache_connection.tensorget(cache_key)).to(self.device) # 1 x tokens x encoding_dim
            except redis.exceptions.ResponseError:
                # key does not exist
                pass
        if not torch.is_tensor(embeddings):
            # key did not exist
            embeddings = self.similarity_model.encode(text if text else "", convert_to_tensor=True, show_progress_bar=False).unsqueeze(0).unsqueeze(1)
            if self.caching and text and (not text.isnumeric()):
                self.cache_connection.tensorset(cache_key, embeddings.clone().detach().cpu().numpy())
            
        # result
        return embeddings.squeeze(0)

   
