from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
from typing import List
from chatbot.adviser.app.encoding.similiarity import AnswerSimilarityEncoding
import chatbot.adviser.app.rl.dataset as Data

import torch
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification

from chatbot.adviser.app.rl.dialogtree import DialogTree
from chatbot.adviser.app.rl.dataset import DialogNode
from chatbot.adviser.app.rl.utils import AutoSkipMode


class Intent(Enum):
    GUIDED = 0
    FREE = 1


class IntentTracker:
    def __init__(self, device: str = "cpu", ckpt_dir='./.models/intentpredictor') -> None:
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained('deepset/gbert-large', use_fast=True, cache_dir=".models/gbert", truncation_side='left')
        self.model = AutoModelForSequenceClassification.from_pretrained(ckpt_dir, output_hidden_states = True).to(device)
        self.model.eval()
        self.free_counter = 0
        self.guided_counter = 0

    def get_intent(self, dialog_node: DialogNode, gen_user_utterance: str) -> Intent:
        tok = self.tokenizer(text=dialog_node.content.text, text_pair=gen_user_utterance, truncation=True, return_tensors="pt")
        tok = {key: tok[key].to(self.device) for key in tok}
        class_idx = self.model(**tok).logits.argmax(-1).item()
        return Intent(class_idx)
            

@dataclass
class FAQSearchResult:
    query: str
    similarity: float
    top_k: int
    goal_node_key: int


class FAQPolicy:
    def __init__(self, dialog_tree: DialogTree, similarity_model: SentenceTransformer, top_k: int, noise: float) -> None:
        self.similarityModel = similarity_model
        self.corpus_keys = []
        self.corpus_texts = []
        self.topk = top_k
        self.noise = noise

        node_keys = set()
        for faq in Data.objects[dialog_tree.version].faq_list():
            if not faq.dialog_node_key in node_keys:
                node_keys.add(faq.dialog_node_key)
                self.corpus_keys.append(faq.dialog_node_key)
                self.corpus_texts.append(Data.objects[dialog_tree.version].node_by_key(faq.dialog_node_key).content.text)

        self.copus_embedding = self.similarityModel.encode(self.corpus_texts, convert_to_tensor=True)

    def top_k(self, query: str) -> List[FAQSearchResult]:
        query_emb = self.similarityModel.encode(query, convert_to_tensor=True, show_progress_bar=False)
        if self.noise > 0.0:
            query_emb = torch.normal(mean=query_emb, std=self.noise*torch.abs(query_emb))
        cos_scores = cos_sim(query_emb, self.copus_embedding)[0]
        top_results = torch.topk(cos_scores, k=self.topk)
            
        results = []
        counter = 0
        for score, idx in zip(top_results[0], top_results[1]):
            results.append(FAQSearchResult(deepcopy(query), similarity=score, top_k=counter, goal_node_key=self.corpus_keys[idx]))
            counter += 1
        
        assert len(results) == self.topk
        return results


class GuidedPolicy:
    def __init__(self, similarity_model: AnswerSimilarityEncoding, stop_action: bool, auto_skip: AutoSkipMode, noise: float) -> None:
        self.similarity_model = similarity_model
        self.stop_action_decrement = 1 - int(stop_action)
        self.noise = noise
        self.auto_skip = auto_skip
        # assert auto_skip != AutoSkipMode.NONE, "need an auto-skip mode other than NONE for baseline"

    def reset(self):
        self.turns = 0

    def get_action(self, dialog_node: DialogNode, user_utterance: str, last_sys_act):
        self.turns += 1
        
        if dialog_node.node_type == "infoNode":
            if last_sys_act == 1:
                action = 2 # skip to connected node, since info node was already asked
            else:
                action = 1 # ASK info node
        elif dialog_node.node_type == "userInputNode":
            if last_sys_act == 1:
                action = 2 # skip to 1st answer
            else:
                action = 1 # ASK variable
        elif dialog_node.node_type == "userResponseNode":
            if last_sys_act == 1:
                # skip
                if self.auto_skip == AutoSkipMode.SIMILARITY:
                    cos_scores = self.similarity_model.encode(current_user_utterance=user_utterance if user_utterance else "", dialog_node=dialog_node, noise=self.noise)
                    most_similar_answer_idx = cos_scores.view(-1).argmax(-1).item()
                    action = most_similar_answer_idx + 2 # add 2 because of STOP and ASK
                else:
                    # TODO implement oracle auto skip mode
                    raise NotImplementedError
            else:
                action = 1 # ASK response node
        else:
            raise f"Node type not handled by policy: {dialog_node.node_type}"
        
        return action - self.stop_action_decrement