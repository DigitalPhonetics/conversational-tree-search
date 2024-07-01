import importlib
from typing import Any, Dict, List, Union
from utils.utils import State
from config import StateConfig, ActionType
import torch
from data.dataset import DialogNode, GraphDataset
from encoding.action import ActionTypeEncoding
from encoding.bst import BSTEncoding
from encoding.nodeinfo.nodetype import NodeTypeEncoding
from encoding.nodeinfo.position import TreePositionEncoding, AnswerPositionEncoding

from encoding.text.base import TextEmbeddingConfig, TextEmbeddingPooling, TextEmbeddings


def requires_text_embedding(obj):
    return hasattr(obj, '__dict__') and "ckpt_name" in obj


class Cache:
    def __init__(self, device: str, data: GraphDataset, state_config: StateConfig, torch_compile: bool) -> None:
        self.device = device
        self.text_embeddings: Dict[str, TextEmbeddings] = {}
        self.other_embeddings = {}
        
        # extract all state input keys that require text embeddings and load the embeddings (use text inputs)
        # (make sure to only load one instance of each different embedding type)
        text_embedding_keys = filter(lambda x: requires_text_embedding(state_config[x]), state_config)
        self.state_embedding_cfg: Dict[State, TextEmbeddingConfig] = {}
        for state_input_key in text_embedding_keys:
            # save each configuration individually, but share the embeddings instance
            embedding_cfg: TextEmbeddingConfig = state_config[state_input_key]
            self.state_embedding_cfg[State(state_input_key)] = embedding_cfg
            cls_name_components = embedding_cfg._target_.split(".")
            if not embedding_cfg._target_ in self.text_embeddings:
                print(f"Loading Embedding (caching: {embedding_cfg.caching}) {embedding_cfg._target_} ...")
                embedding_cls: TextEmbeddings = getattr(importlib.import_module(".".join(cls_name_components[:-1])), cls_name_components[-1])
                embedding_instance = embedding_cls(device=device, ckpt_name=embedding_cfg.ckpt_name, embedding_dim=embedding_cfg.embedding_dim, torch_compile=torch_compile)
                self.text_embeddings[embedding_cfg._target_] = embedding_instance

        # extract all state input keys that require positional embeddings (use DialogNode inputs)
        if state_config.node_position:
            self.node_position_encoding = TreePositionEncoding(device=device, data=data) 
        if state_config.action_position:
            self.action_position_encoding = AnswerPositionEncoding(device=device, data=data)

        # extract other input keys that require encoding (use individual inputs)
        self.node_type_encoding = NodeTypeEncoding(device=device, data=data) if state_config.node_type else None
        self.action_type_encoding = ActionTypeEncoding(device=device)
        self.bst_encoding = BSTEncoding(device=device, data=data) if state_config.beliefstate else None
        # self.intent_encoding = IntentEncoding(device=device, ckpt_dir=TODO)
        # TODO similarity encoding

    @torch.no_grad()
    def _apply_noise(self, state_input_key: State, embeddings: torch.FloatTensor) -> torch.FloatTensor:
        if self.state_embedding_cfg[state_input_key].noise_std > 0.0:
            return torch.normal(mean=embeddings, std=self.state_embedding_cfg[state_input_key].noise_std*torch.abs(embeddings))
        return embeddings

    @torch.no_grad()
    def _apply_pooling(self, embeddings: torch.FloatTensor, pooling: TextEmbeddingPooling) -> torch.FloatTensor:
        # pooling
        assert isinstance(pooling, TextEmbeddingPooling)
        if pooling == TextEmbeddingPooling.MEAN:
            return embeddings.mean(1)   # 1 x seq_len x 1024 => average to 1 x 1024 to get sentence embedding
        elif pooling == TextEmbeddingPooling.CLS:
            return embeddings[:, 0, :]  # extract CLS embedding 1 x seq_len x 1024 =>  1 x 1024 
        elif pooling == TextEmbeddingPooling.MAX:
            return embeddings.max(1)[0] #1 x seq_len x 1024 => max-pool to 1 x 1024 to get sentence embedding
        else:
            return embeddings # return unprocessed sequence: 1 x 1 x embedding_dim -> 1 x embedding_dim

    @torch.no_grad()
    def _encode(self, state_input_key: State, text_embedding_name: str, caching: bool, cache_key: str, encode_fn: Any, value: Union[str, List[str]]):
        pooling = self.state_embedding_cfg[state_input_key].pooling
        embeddings = encode_fn(value).detach().cpu()
        embeddings = self._apply_noise(state_input_key=state_input_key, embeddings=embeddings)
        embeddings = self._apply_pooling(embeddings, pooling)
        return embeddings


    @torch.no_grad()
    def encode_text(self, state_input_key: State, text: str):
        text_embedding_name = self.state_embedding_cfg[state_input_key]._target_
        text_embedding: TextEmbeddings = self.text_embeddings[text_embedding_name]
        return self._encode(state_input_key=state_input_key,
                            text_embedding_name=text_embedding_name,
                            caching=text and (not text.isnumeric()),
                            cache_key=text.lower(),
                            encode_fn=text_embedding.encode,
                            value=text).squeeze(0)
    
    @torch.no_grad()
    def batch_encode_text(self, state_input_key: State, text: List[str]):
        text_embedding_name = self.state_embedding_cfg[state_input_key]._target_
        text_embedding: TextEmbeddings = self.text_embeddings[text_embedding_name]

        # embedding
        embeddings = text_embedding.batch_encode(text) # batch x embedding_dim

        # noise
        embeddings = self._apply_noise(state_input_key=state_input_key, embeddings=embeddings)
        embeddings = self._apply_pooling(embeddings, self.state_embedding_cfg[state_input_key].pooling)

        return embeddings.detach().cpu()

    @torch.no_grad()
    def encode_answer_text(self, node: DialogNode):
        """
        Returns:
            encoded answers (torch.FloatTensor): num_actions x embedding_size, if pooled
                                                 num_actions x max_length x embeddings_size, if pooling = None
        """
        text_embedding_name = self.state_embedding_cfg[State.ACTION_TEXT]._target_
        text_embedding: TextEmbeddings = self.text_embeddings[text_embedding_name]
        return self._encode(state_input_key=State.ACTION_TEXT,
                            text_embedding_name=text_embedding_name,
                            caching=len(node.answers) > 0,
                        cache_key=f"actions_{node.key}",
                            encode_fn=text_embedding.batch_encode, # returns: num_actions x max_length x embedding_size
                            value=[node.answer_by_index(i).text for i in range(len(node.answers))])
    
    @torch.no_grad()
    def batch_encode_answer_text(self, node: List[DialogNode], action_space_dim: int):
        """
        Returns:
            Padded tensor: len(node) x action_space_dim x 2 + embedding_dim (+2: includes action type encoding)
        """
        text_embedding_name = self.state_embedding_cfg[State.ACTION_TEXT]._target_
        text_embedding: TextEmbeddings = self.text_embeddings[text_embedding_name]

        # get dimensions
        num_nodes = len(node)
        num_answers = [len(n.answers) for n in node]

        embeddings = torch.zeros(num_nodes, action_space_dim, 2+text_embedding.get_encoding_dim(), dtype=torch.float)

        # sharding to keep memory constraints
        MAX_BATCH_SIZE = 768
        batch_start_index = 0
        batch_end_index = 0
        current_batch = []
        node_index = 0
        
        while batch_end_index < len(node):
            # build next batch
            while batch_end_index < len(node) and len(current_batch) < MAX_BATCH_SIZE:
                if len(node[batch_end_index].answers) == 0:
                    # no answers in node - skip
                    batch_end_index += 1
                    continue
                if len(current_batch) + len(node[batch_end_index].answers) < MAX_BATCH_SIZE:
                    # extend batch
                    current_batch += [node[batch_end_index].answer_by_index(i).text for i in range(len(node[batch_end_index].answers))]
                    batch_end_index += 1
                else:
                    # reached max batch size
                    break
            
            # batch encode current shard
            if len(current_batch) > 0:
                # encode
                enc = text_embedding.batch_encode(current_batch).detach().cpu() # batch x embedding_dim
                # noise
                enc = self._apply_noise(state_input_key=State.ACTION_TEXT, embeddings=enc)
                enc = enc.detach().cpu()
                
                # assign answer embeddings to correct tensor positions
                enc_start = 0
                for answer_count in num_answers[batch_start_index:batch_end_index]:
                    # Add action type (SKIP) encoding
                    embeddings[node_index, :answer_count, ActionType.ASK.value] = 1.0
                    if answer_count > 0:
                        # Add answer text encoding
                        embeddings[node_index, :answer_count, 2:] = enc[enc_start:enc_start+answer_count]
                    enc_start += answer_count
                    node_index += 1
                batch_start_index = batch_end_index
                current_batch = []

        return embeddings

           