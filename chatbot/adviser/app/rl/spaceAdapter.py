from dataclasses import dataclass
from enum import Enum
from functools import reduce
import itertools
from typing import Any, Dict, List 
from chatbot.adviser.app.encoding.encoding import Encoding
from chatbot.adviser.app.encoding.intent import IntentEncoding
from chatbot.adviser.app.rl.dialogtree import DialogTree
from chatbot.adviser.app.encoding.text import RNN_OUTPUT_SIZE, TextEmbeddingPooling, TextEmbeddings
from chatbot.adviser.app.encoding.position import TreePositionEncoding, NodeTypeEncoding, AnswerPositionEncoding
from chatbot.adviser.app.encoding.bst import BSTEncoding
from chatbot.adviser.app.encoding.similiarity import AnswerSimilarityEncoding
from chatbot.adviser.app.rl.layers.attention.attention_factory import AttentionActivationConfig, AttentionMechanismConfig, AttentionVectorAggregation
from chatbot.adviser.app.rl.utils import AutoSkipMode, StateEntry
from torch.nn.utils.rnn import pack_sequence
import chatbot.adviser.app.rl.dataset as Data

import torch
import torch.nn.functional as F


class ActionConfig(Enum):
    ACTIONS_IN_ACTION_SPACE = 0
    ACTIONS_IN_STATE_SPACE = 1


@dataclass
class SpaceAdapterInput:

    def post_init(self, **kwargs):
        pass

    def get_state_dim(self) -> int:
        raise NotImplementedError

    def get_action_dim(self) -> int:
        raise NotImplementedError

    def get_action_state_subvec_dim(self) -> int:
        raise NotImplementedError

    def encode(self, **kwargs) -> Dict[str, torch.FloatTensor]:
        raise NotImplementedError

    def toJson(self) -> Dict[str, Any]:
        raise NotImplementedError
    
    @classmethod
    def fromJson(cls, data: dict):
        raise NotImplementedError


@dataclass
class SpaceAdapterConfiguration(SpaceAdapterInput):
    text_embedding: str
    action_config: ActionConfig
    action_masking: bool
    stop_action: bool
    auto_skip: AutoSkipMode
    use_answer_synonyms: bool

    def post_init(self, tree: DialogTree, **kwargs):
        self.tree = tree

    def get_state_dim(self) -> int:
        return 0 

    def get_action_dim(self) -> int:
        return 0

    def get_action_state_subvec_dim(self) -> int:
        return 0

    def encode(self, **kwargs) -> Dict[str, torch.FloatTensor]:
        return {}

    def toJson(self) -> Dict[str, Any]:
        return {
            "configuration": {
                "text_embedding": self.text_embedding,
                "action_config": self.action_config.value,
                "action_masking": self.action_masking,
                "stop_action": self.stop_action
            }
        }
    
    @classmethod
    def fromJson(cls, data: dict):
        action_config = ActionConfig(data.pop('action_config'))
        return cls(action_config=action_config, **data)


@dataclass
class TextEmbeddingConfig:
    active: bool
    pooling: TextEmbeddingPooling

    def toJson(self):
        return {
            "active": self.active,
            "pooling": self.pooling.value,
        }
    
    @classmethod
    def fromJson(cls, data: dict):
        pooling = TextEmbeddingPooling(data.pop('pooling'))
        return cls(pooling=pooling, **data)


@dataclass
class IntentEmbeddingConfig:
    active: bool
    ckpt_dir: str

    def toJson(self):
        return {
            "active": self.active,
            "ckpt_dir": self.ckpt_dir
        }
    
    @classmethod
    def fromJson(cls, data: dict):
        return cls(**data)


@dataclass
class AnswerSimilarityEmbeddingConfig:
    active: bool
    model_name: str

    def toJson(self):
        return {
            "active": self.active,
            "model_name": self.model_name,
        }
    
    @classmethod
    def fromJson(cls, data: dict):
        return cls(**data)





class TextEncoderWrapper(Encoding):
    def __init__(self, text_embedding: TextEmbeddings, pooling: TextEmbeddingPooling, encode_input_key: str, allow_noise: bool = True) -> None:
        super().__init__(device=text_embedding.device)
        self.text_embedding = text_embedding
        self.pooling = pooling
        self.encode_input_key = encode_input_key 
        self.allow_noise = allow_noise

    def encode(self, model: torch.nn.Module, noise: float, **kwargs) -> torch.FloatTensor:
        if self.encode_input_key == 'action_text':
            return torch.cat([self._embed_text(answer.content.text, model=model, noise=noise) for answer in kwargs['dialog_node'].answers], dim=0).unsqueeze(0)
        else:
            return self._embed_text(text=kwargs[self.encode_input_key], model=model, noise=noise)

    def batch_encode(self, model: torch.nn.Module, **kwargs) -> torch.FloatTensor:
        raise NotImplementedError


    def _embed_text(self, text: str, model: torch.nn.Module = None, noise: float = 0.0):
        embeddings = None
        # encoding
        embeddings = self.text_embedding.encode(text=text)
            
        # pooling of 1 x TOKENS x ENCODING_DIM
        if self.pooling == TextEmbeddingPooling.MEAN:
            sentence_emb = embeddings.mean(1)   # 1 x seq_len x 1024 => average to 1 x 1024 to get sentence embedding
        elif self.pooling == TextEmbeddingPooling.CLS:
            sentence_emb = embeddings[:, 0, :]  # extract CLS embedding 1 x seq_len x 1024 =>  1 x 1024 
        elif self.pooling == TextEmbeddingPooling.MAX:
            sentence_emb = embeddings.max(1)[0] #1 x seq_len x 1024 => max-pool to 1 x 1024 to get sentence embedding
        elif self.pooling == TextEmbeddingPooling.RNN:
            sentence_emb = model.process_rnn(embeddings, rnn_key=self.encode_input_key) # 1 x seq_len x 1024 => 1 x 512
        else:
            sentence_emb = embeddings # also applies to RNN pooling - return unprocessed sequence

        # generalization / robustness
        if self.allow_noise and noise > 0.0:
            sentence_emb = torch.normal(mean=sentence_emb, std=noise*torch.abs(sentence_emb))

        return sentence_emb

    def get_encoding_dim(self) -> int:
        return self.text_embedding.get_encoding_dim() if self.pooling != TextEmbeddingPooling.RNN else RNN_OUTPUT_SIZE


class TextHistoryEncoderWrapper(Encoding):
    def __init__(self, text_embedding: TextEmbeddings, pooling: TextEmbeddingPooling, encode_input_key: str) -> None:
        super().__init__(device=text_embedding.device)
        self.pooling = pooling
        if pooling == TextEmbeddingPooling.RNN:
            self.sys_text_enc_wrapper = TextEncoderWrapper(text_embedding=text_embedding, pooling=TextEmbeddingPooling.MEAN, allow_noise=False, encode_input_key='dialog_node_text')
            self.usr_text_enc_wrapper = TextEncoderWrapper(text_embedding=text_embedding, pooling=TextEmbeddingPooling.MEAN, allow_noise=True, encode_input_key='current_user_utterance')
        else:
            self.text_embedding = text_embedding
            self.combined_text_enc_wrapper = TextEncoderWrapper(text_embedding=text_embedding, pooling=pooling, allow_noise=True, encode_input_key='dialog_history')
    
    def encode(self, model: torch.nn.Module, dialog_history: str, noise: float, **kwargs) -> torch.FloatTensor:
        if self.pooling == TextEmbeddingPooling.RNN:
            # Embed each entry in the list of history tuples and concatenate the result.
            # Then, feed the result through an RNN
            embeddings = map(lambda entry: [self.sys_text_enc_wrapper.encode(model, noise=noise, dialog_node_text=entry[0]),
                                            self.usr_text_enc_wrapper.encode(model, noise=noise, current_user_utterance=entry[1])], dialog_history) # list of pairs of embeddings: [[sys1, usr1], [sys2, usr2], ...]
            embeddings = list(itertools.chain.from_iterable(embeddings)) # list of embeddings [sys1, usr1, sys2, usr2, ...]
            embeddings = torch.cat(embeddings, dim=0) # turns x embedding_dim 
            speaker_indicator = torch.ones(embeddings.size(0), dtype=torch.float, device=self.device) # turns
            speaker_indicator[::2] = 0.0 # set every second element to 0 to indicate speaker change (-> system: 0, user: 1)
            embeddings = torch.cat([embeddings, speaker_indicator.unsqueeze(-1)], dim=1) # turns x embedding_dim -> turns x embedding_dim + 1
            embeddings = model.process_rnn(embeddings.unsqueeze(0), 'dialog_history') # 1 x turns x embedding_dim + 1-> 1 x 1 x 512
            return embeddings.squeeze(0) # 1 x 512
        else:
            # Combine list of history tuples [(sys1, usr1), (sys2, usr2), ...] into one string and embed it
            combined_dialog_history = " ".join(map(lambda entry: f"SYSTEM: {entry[0]} NUTZER: {entry[1]}", dialog_history))
            return self.combined_text_enc_wrapper.encode(model, noise=noise, dialog_history=combined_dialog_history)

    def _pre_encode_history(self, model: torch.nn.Module, dialog_history: str, noise: float, **kwargs):
        embeddings = map(lambda entry: [self.sys_text_enc_wrapper.encode(model, noise=noise, dialog_node_text=entry[0]),
                                            self.usr_text_enc_wrapper.encode(model, noise=noise, current_user_utterance=entry[1])], dialog_history) # list of pairs of embeddings: [[sys1, usr1], [sys2, usr2], ...]
        embeddings = list(itertools.chain.from_iterable(embeddings)) # list of embeddings [sys1, usr1, sys2, usr2, ...]
        embeddings = torch.cat(embeddings, dim=0) # turns x embedding_dim 
        speaker_indicator = torch.ones(embeddings.size(0), dtype=torch.float, device=self.device) # turns
        speaker_indicator[::2] = 0.0 # set every second element to 0 to indicate speaker change (-> system: 0, user: 1)
        speaker_indicator[::2] = 0.0 # set every second element to 0 to indicate speaker change (-> system: 0, user: 1)
        return torch.cat([embeddings, speaker_indicator.unsqueeze(-1)], dim=1) # turns x embedding_dim -> turns x embedding_dim + 1


    def batch_encode(self, model: torch.nn.Module, noise: float, dialog_history: List[str], **kwargs) -> torch.FloatTensor:
        if self.pooling == TextEmbeddingPooling.RNN:
            embeddings = pack_sequence([self._pre_encode_history(model, dialog_history=history, noise=noise) for history in dialog_history], enforce_sorted=False)
            return model.process_rnn(embeddings, 'dialog_history').squeeze(0) # 1 x batch x 512 -> batch x 512
        else:
            return torch.cat([self.encode(model, dialog_history=history, noise=noise, **kwargs) for history in dialog_history], -1) # batch x embedding_dim
    
    def get_encoding_dim(self) -> int:
        return self.text_embedding.get_encoding_dim() if self.pooling != TextEmbeddingPooling.RNN else RNN_OUTPUT_SIZE


class SysActEncoder(Encoding):
    def __init__(self, device: str, tree: DialogTree, stop_action: bool):
        super().__init__(device=device)
        self.num_actions = 2 if stop_action else 1 
        self.num_actions += tree.get_max_node_degree()
        self.action_decrement = 0 if stop_action else -1
    
    def encode(self, last_sysact: int, **kwargs) -> torch.FloatTensor:
        return F.one_hot(torch.tensor([last_sysact + self.action_decrement], dtype=torch.long, device=self.device), num_classes=self.num_actions) # 1 x num_actions

    def get_encoding_dim(self) -> int:
        return self.num_actions


@dataclass 
class SpaceAdapterSpaceInput(SpaceAdapterInput):
    last_system_action: bool
    beliefstate: bool
    current_node_position: bool
    current_node_type: bool
    user_intent_prediction: IntentEmbeddingConfig
    answer_similarity_embedding: AnswerSimilarityEmbeddingConfig
    dialog_node_text: TextEmbeddingConfig
    original_user_utterance: TextEmbeddingConfig
    current_user_utterance: TextEmbeddingConfig
    dialog_history: TextEmbeddingConfig
    action_text: TextEmbeddingConfig
    action_position: bool

    def post_init(self, device: str, tree: DialogTree, text_embedding: TextEmbeddings, action_config: ActionConfig, stop_action: bool, **kwargs):
        self.encoders = {}
        self.rnn_encoders = {}
        self.action_state_subvec_dim = 0
        self.action_dim = 0
        self.state_dim = 0

        self.text_encoding_dim = text_embedding.get_encoding_dim()

        if action_config == ActionConfig.ACTIONS_IN_ACTION_SPACE:
            self.action_dim += 2 if stop_action else 1 # STOP, ASK
            self.action_dim += tree.get_max_node_degree() # 1 Q-value per action STOP, ASK, SKIP_1, ..., SKIP_N 
        elif action_config == ActionConfig.ACTIONS_IN_STATE_SPACE:
            self.state_dim += 3 if stop_action else 2 # one-hot encoding for action type (STOP, ASK, SKIP)
            self.action_state_subvec_dim = 3 if stop_action else 2 # one-hot encoding for action type (STOP, ASK, SKIP)
            self.action_dim += 1 # output is only 1 Q-value (action is in encoded in input)


        if action_config == ActionConfig.ACTIONS_IN_ACTION_SPACE and self.action_position:
            raise f"Can't have action position in ACTION_SPACE configuration"
        if self.last_system_action:
            self.encoders['last_system_action'] = SysActEncoder(device=device, tree=tree, stop_action=stop_action)
        if self.beliefstate:
            self.encoders['beliefstate'] = BSTEncoding(device=device, version=0)
        if self.current_node_position:
            self.encoders['current_node_position'] = TreePositionEncoding(device=device)
        if self.current_node_type:
            self.encoders['current_node_type'] = NodeTypeEncoding(device=device)
        if self.user_intent_prediction.active:
            self.encoders['user_intent_prediction'] = IntentEncoding(device=device, ckpt_dir=self.user_intent_prediction.ckpt_dir)
        if self.answer_similarity_embedding.active:
            self.encoders['action_answer_similarity_embedding'] = AnswerSimilarityEncoding(device=device, dialog_tree=tree, model_name=self.answer_similarity_embedding.model_name)
            self.action_state_subvec_dim += 1
        else:
            print("NO ANSWER SIMILARITY EMBEDDING")
        if self.dialog_node_text.active:
            text_enc = TextEncoderWrapper(text_embedding=text_embedding, pooling=self.dialog_node_text.pooling, allow_noise=False, encode_input_key='dialog_node_text')
            if self.dialog_node_text.pooling == TextEmbeddingPooling.RNN:
                self.rnn_encoders['dialog_node_text'] = text_enc
            else:
                self.encoders['dialog_node_text'] = text_enc
        if self.original_user_utterance.active:
            text_enc = TextEncoderWrapper(text_embedding=text_embedding, pooling=self.original_user_utterance.pooling, allow_noise=True, encode_input_key='original_user_utterance')
            if self.original_user_utterance.pooling == TextEmbeddingPooling.RNN:
                self.rnn_encoders['original_user_utterance'] = text_enc
            else:
                self.encoders['original_user_utterance'] = text_enc
        if self.current_user_utterance.active:
            text_enc = TextEncoderWrapper(text_embedding=text_embedding, pooling=self.current_user_utterance.pooling, allow_noise=True, encode_input_key='current_user_utterance')
            if self.current_user_utterance.pooling == TextEmbeddingPooling.RNN:
                self.rnn_encoders['current_user_utterance'] = text_enc
            else:
                self.encoders['current_user_utterance'] = text_enc
        if self.dialog_history.active:
            text_enc = TextHistoryEncoderWrapper(text_embedding=text_embedding, pooling=self.dialog_history.pooling, encode_input_key='dialog_history')
            if self.dialog_history.pooling == TextEmbeddingPooling.RNN:
                self.rnn_encoders['dialog_history'] = text_enc
            else:
                self.encoders['dialog_history'] = text_enc
        if self.action_text.active:
            if action_config == ActionConfig.ACTIONS_IN_ACTION_SPACE:
                raise "Can't have action text embedding in action space"
            text_enc = TextEncoderWrapper(text_embedding=text_embedding, pooling=self.action_text.pooling, allow_noise=False, encode_input_key='action_text')
            if self.action_text.pooling == TextEmbeddingPooling.RNN:
                self.rnn_encoders['action_text'] = text_enc
            else:
                self.encoders['action_text'] = text_enc
            self.action_state_subvec_dim += text_enc.get_encoding_dim()
        if self.action_position:
            if action_config == ActionConfig.ACTIONS_IN_ACTION_SPACE:
                raise "Can't have action position encoding in action space"
            self.encoders['action_position'] = AnswerPositionEncoding(device=device, dialog_tree=tree)
            self.action_state_subvec_dim += self.encoders['action_position'].get_encoding_dim()

        self.state_dim += sum(map(lambda enc: enc.get_encoding_dim(), list(self.encoders.values()) + list(self.rnn_encoders.values())))
        
    def get_text_encoding_dim(self) -> int:
        return self.text_encoding_dim

    def get_state_dim(self) -> int:
        return self.state_dim

    def get_action_state_subvec_dim(self) -> int:
        return self.action_state_subvec_dim

    def get_action_dim(self) -> int:
        return self.action_dim

    def encode(self, **kwargs) -> Dict[str, torch.FloatTensor]:
        # each encoding has size 1 x encoding_dim
        enc = {
            encoding_key: self.encoders[encoding_key].encode(**kwargs) for encoding_key in self.encoders if not "action_text" in encoding_key
        }
        if 'action_text' in self.encoders:
            if kwargs['dialog_node'].answer_count() > 0:
                enc['action_text'] = self.encoders['action_text'].encode(**kwargs)
            else:
                enc['action_text'] = None
        return enc

    def toJson(self) -> Dict[str, Any]:
        return {
            "state": {
                "last_system_action": self.last_system_action,
                "beliefstate": self.beliefstate,
                "current_node_position": self.current_node_position,
                "current_node_type": self.current_node_type,
                "user_intent_prediction": self.user_intent_prediction.toJson(),
                "dialog_node_text": self.dialog_node_text.toJson(),
                "original_user_utterance": self.original_user_utterance.toJson(),
                "current_user_utterance": self.current_user_utterance.toJson(),
                "dialog_history": self.dialog_history.toJson(),
                "action_text": self.action_text.toJson(),
                "action_position": self.action_position
            }
        }

    @classmethod
    def fromJson(cls, data: dict):
        user_intent_prediction = IntentEmbeddingConfig.fromJson(data.pop('user_intent_prediction'))
        dialog_node_text = TextEmbeddingConfig.fromJson(data.pop('dialog_node_text'))
        original_user_utterance = TextEmbeddingConfig.fromJson(data.pop('original_user_utterance'))
        current_user_utterance = TextEmbeddingConfig.fromJson(data.pop('current_user_utterance'))
        dialog_history = TextEmbeddingConfig.fromJson(data.pop('dialog_history'))
        action_text = TextEmbeddingConfig.fromJson(data.pop('action_text'))
        return cls(user_intent_prediction=user_intent_prediction,
                    dialog_node_text=dialog_node_text,
                    original_user_utterance=original_user_utterance,
                    current_user_utterance=current_user_utterance,
                    dialog_history=dialog_history,
                    action_text=action_text,
                    **data)


@dataclass
class SpaceAdapterAttentionQueryInput(SpaceAdapterInput):
    input: List[str]
    aggregation: AttentionVectorAggregation
    pooling: TextEmbeddingPooling
    allow_noise: bool

    def post_init(self, device: str, tree: DialogTree, text_embedding: TextEmbeddings, action_config: ActionConfig, action_masking: bool):
        self.state_dim = 0
        self.encoders = {}
        assert self.pooling != TextEmbeddingPooling.NONE, "Need a pooling method for the attention query vectors!"
        for input_vector in self.input:
            self.encoders[input_vector] = TextEncoderWrapper(text_embedding=text_embedding, allow_noise=self.allow_noise, pooling=self.pooling, encode_input_key=input_vector)

        if self.aggregation == AttentionVectorAggregation.CONCATENATE:
            self.state_dim += len(self.input) * text_embedding.get_encoding_dim()
        else:
            self.state_dim += text_embedding.get_encoding_dim()

    def get_state_dim(self) -> int:
        return self.state_dim

    def get_action_dim(self) -> int:
        return 0

    def get_action_state_subvec_dim(self) -> int:
        return 0

    def encode(self, model: torch.nn.Module, **kwargs) -> Dict[str, torch.FloatTensor]:
        return {
            input_vector: self.encoders[input_vector].encode(model, **kwargs) for input_vector in self.input    
        }

    def toJson(self) -> Dict[str, Any]:
        return {
            "vectors": {
                "input": self.input,
                "aggregation": self.aggregation.value,
                "pooling": self.pooling.value,
            }
        }

    @classmethod
    def fromJson(cls, data: dict):
        pooling = TextEmbeddingPooling(data.pop('pooling'))
        aggregation = AttentionVectorAggregation(data.pop('aggregation'))
        return cls(pooling=pooling, aggregation=aggregation, **data)


@dataclass
class SpaceAdapterAttentionInput(SpaceAdapterInput):
    active: bool
    name: str
    queries: SpaceAdapterAttentionQueryInput
    matrix: str
    activation: AttentionActivationConfig
    attention_mechanism: AttentionMechanismConfig
    allow_noise: bool

    def _encode_attention(self, model: torch.nn.Module, name: str, vectors: List[torch.FloatTensor], matrix: torch.FloatTensor, **kwargs) -> torch.FloatTensor:
        context = []
        for vector in vectors:
            context.append(model.process_attention(name=name, query=vector, matrix=matrix))
        return context

    def post_init(self, device: str, tree: DialogTree, text_embedding: TextEmbeddings, action_config: ActionConfig, action_masking: bool):
        self.device = device
        if self.active:
            self.queries.post_init(device, tree, text_embedding, action_config, action_masking)
            self.state_dim = self.queries.get_state_dim()
            if self.matrix == 'dialog_history':
                self.matrix_enc = TextHistoryEncoderWrapper(text_embedding=text_embedding, pooling=TextEmbeddingPooling.NONE, encode_input_key=self.matrix)
            else:
                self.matrix_enc = TextEncoderWrapper(text_embedding=text_embedding, pooling=TextEmbeddingPooling.NONE, allow_noise=self.allow_noise, encode_input_key=self.matrix)
        else:
            self.state_dim = 0

    def get_state_dim(self) -> int:
        return self.state_dim

    def get_vector_dim(self):
        return self.matrix_enc.get_encoding_dim()

    def get_action_dim(self) -> int:
        return 0

    def get_action_state_subvec_dim(self) -> int:
        return 0

    def encode(self, model: torch.nn.Module, **kwargs) -> Dict[str, torch.FloatTensor]:
        if not self.active:
            return {}

        context = self._encode_attention(
            model=model,
            name=self.name,
            vectors=self.queries.encode(model=model, **kwargs).values(),
            matrix=self.matrix_enc.encode(model, **kwargs)
        )
        
        attn = None
        if len(context) == 1:
            attn = context[0]
        elif self.queries.aggregation == AttentionVectorAggregation.CONCATENATE:
            attn = torch.cat(context, dim=-1) # 1 x encoding_dim
        elif self.queries.aggregation == AttentionVectorAggregation.MAX:
            attn = reduce(torch.max, context) # 1 x encoding_dim
        elif self.queries.aggregation == AttentionVectorAggregation.MEAN:
            attn = reduce(torch.add, context) / len(context) # 1 x encoding_dim
        elif self.queries.aggregation == AttentionVectorAggregation.SUM:
            attn = reduce(torch.add, context) # 1 x encoding_dim

        return {f"attn_{self.name}": attn}

    def toJson(self) -> Dict[str, Any]:
        return {
            "active": self.active,
            "name": self.name,
            "queries": self.queries.toJson(),
            "matrix": self.matrix,
            "activation": self.activation.value,
            "attention_mechanism": self.attention_mechanism.value,
        }
    
    @classmethod
    def fromJson(cls, data: dict):
        queries = SpaceAdapterAttentionQueryInput.fromJson(data.pop('queries')['vectors'])
        activation = AttentionActivationConfig(data.pop('activation'))
        attention_mechanism = AttentionMechanismConfig(data.pop('attention_mechanism'))
        return cls(queries=queries, activation=activation, attention_mechanism=attention_mechanism, **data)



class SpaceAdapter:
    def __init__(self, device: str, dialog_tree: DialogTree,
                    configuration: SpaceAdapterConfiguration,
                    state: SpaceAdapterSpaceInput,
                    attention: List[SpaceAdapterAttentionInput]) -> None:

        self.device = device
        self.state_dim = configuration.get_state_dim() + state.get_state_dim() + sum([attn.get_state_dim() for attn in attention])
        self.action_dim = configuration.get_action_dim() + state.get_action_dim() + sum([attn.get_action_dim() for attn in attention])
        self.action_state_subvec_dim = configuration.get_action_state_subvec_dim() + state.get_action_state_subvec_dim() + sum([attn.get_action_state_subvec_dim() for attn in attention])
        self.num_actions = 2 if configuration.stop_action else 1 # STOP and ASK
        self.num_actions += dialog_tree.get_max_node_degree()    # + max answers
        self.action_masks = None
        self.dialog_tree = dialog_tree
        self.configuration = configuration
        self.stateinput = state
        self.attentioninput = attention

        if configuration.action_masking:
            self.action_masks = self._calculate_action_masks(dialog_tree=dialog_tree)
        
        self.embedder_fns = {}
        self.state_action_embedder_fns = {}
        self.model = None

    def set_model(self, model: torch.nn.Module):
        self.model = model

    def _calculate_action_masks(self, dialog_tree: DialogTree) -> Dict[int, torch.FloatTensor]:
        """ Pre-calculate action mask per node s.t. it is usable without database lookups """
        masks = {}
        idx_decrement = 0 if self.configuration.stop_action else -1
        # for node in DialogNode.objects.filter(version=self.dialog_tree.version):
        for node in Data.objects[self.dialog_tree.version].nodes():
            mask = torch.zeros(self.num_actions, dtype=torch.bool, device=self.device)
            if node.node_type == 'userResponseNode':
                # skip_answer_count = node.answers.count()
                skip_answer_count = node.answer_count()
                assert skip_answer_count <= dialog_tree.get_max_node_degree()
                if skip_answer_count > 0:
                    mask[skip_answer_count+2+idx_decrement:] = True # no invalid SKIP actions (0: STOP, 1: ASk, 2,...N: SKIP)
                elif skip_answer_count == 0:
                    mask[2+idx_decrement:] = True # no SKIP actions because node has no answers
            elif node.node_type == 'infoNode':
                mask[3+idx_decrement:] = True # no invalid SKIP actions (only 1 skip possible)
            elif node.node_type == 'userInputNode':
                if self.configuration.stop_action:
                    mask[0] = True # no STOP action
                mask[3+idx_decrement:] = True # no invalid SKIP actions (only 1 skip possible)
            else:
                # don't care about startNode and logicNode, they are automatically processed without agent
                continue
            masks[node.key] = mask
        return masks

    def get_state_dim(self) -> int:
        return self.state_dim

    def get_actionstatesubvector_dim(self) -> int:
        return self.action_state_subvec_dim

    def get_action_dim(self) -> int:
        return self.action_dim

    def get_text_encoding_dim(self) -> int:
        return self.stateinput.get_text_encoding_dim()
    
    @torch.no_grad()
    def encode(self, state: Dict[StateEntry, Any]) -> Dict[str, Any]:
        assert not isinstance(self.model, type(None)), "requires call to set_model() before using state_vector()"

        # encode the state space
        node = state[StateEntry.DIALOG_NODE.value]
        all_encodings = self.stateinput.encode(
            model=self.model,
            dialog_node=node,
            dialog_node_text=node.content.text,
            original_user_utterance=state[StateEntry.ORIGINAL_USER_UTTERANCE.value],
            current_user_utterance=state[StateEntry.CURRENT_USER_UTTERANCE.value],
            system_utterance_history=state[StateEntry.SYSTEM_UTTERANCE_HISTORY.value],
            user_utterance_history=state[StateEntry.USER_UTTERANCE_HISTORY.value],
            dialog_history=state[StateEntry.DIALOG_HISTORY.value],
            bst=state[StateEntry.BST.value],
            last_sysact=state[StateEntry.LAST_SYSACT.value],
            noise=state[StateEntry.NOISE.value]
        )
        # NOTE: attention encoding already done in state_vector()!
        # NOTE: RNN encoding already done in state_vector()!

        # encode actions in state space
        if self.configuration.action_config == ActionConfig.ACTIONS_IN_STATE_SPACE:
             # expand flat encoding vector to matrix of size num_node_actions x state_dim
            action_encodings = []
            action_idx_decrement = 0 if self.configuration.stop_action else -1
            num_actions = 1
            if self.configuration.stop_action:
                enc = F.one_hot(torch.tensor([0], dtype=torch.long, device=self.device), num_classes=self.action_state_subvec_dim)  # STOP action (not possible for logic- and userInput nodes)
                # NOTE: we don't have to concatenate with a seperate 0-tensor for the similarity embedding, since it's value will be zero anyways
                action_encodings.append(enc.unsqueeze(1))
                num_actions += 1

            # ASK action possible for all nodes
            ask_enc = F.one_hot(torch.tensor([1+action_idx_decrement], dtype=torch.long, device=self.device), num_classes=self.action_state_subvec_dim)
            # NOTE: we don't have to concatenate with a seperate 0-tensor for the similarity embedding, since it's value will be zero anyways
            action_encodings.append(ask_enc.unsqueeze(1)) 
           
            if node.connected_node_key is not None:
                # infoNode, logicNode, userInputNode -> don't have action text / position to embed
                # add 1 SKIP action
                num_actions += 1
                enc = F.one_hot(torch.tensor([2+action_idx_decrement], dtype=torch.long, device=self.device), num_classes=self.action_state_subvec_dim)
                # NOTE: we don't have to concatenate with a seperate 0-tensor for the similarity embedding, since it's value will be zero anyways
                action_encodings.append(enc.unsqueeze(1)) # SKIP 1
            else:
                # add SKIP actions for all answers
                # num_answers = node.answers.count()
                num_answers = node.answer_count()
                if num_answers > 0:
                    num_actions += num_answers
                    skip_action_enc = F.one_hot(torch.tensor([2+action_idx_decrement] * num_answers, dtype=torch.long, device=self.device), num_classes=3+action_idx_decrement).unsqueeze(0) # [0,0,1] * #answers -> SKIP action code
                    answer_enc_keys = [key for key in all_encodings.keys() if "action_" in key]
                    answer_enc = torch.cat([skip_action_enc] + [all_encodings[answer_enc_key] for answer_enc_key in answer_enc_keys], -1)  #  #answers x action_state_subvec_dim
                    action_encodings.append(answer_enc)
            
            # assemble action encodings and concatenate them with the regular state space encoding
            action_encodings = torch.cat(action_encodings, dim=1).squeeze(0) # num_actions x action_state_subvec_dim
            assert action_encodings.dim() == 2, f"Expected action encoding to have 2 dimensions, got {action_encodings.dim()} with sizes {action_encodings.size()}"
            assert action_encodings.size(0) == num_actions, f"Expected action encoding to have size {num_actions} in 1st dimension, got {action_encodings.size()}"
            assert action_encodings.size(1) == self.action_state_subvec_dim, f"Expected action encoding to have size {self.action_state_subvec_dim} in 2nd dimension, got {action_encodings.size()}"

            all_encodings["action_enc"] = action_encodings

        return state | {f"ENCODED_{key}": all_encodings[key] for key in all_encodings }

    def _concat_single_state_dict(self, state: Dict[str, List[Any]], index: int) -> torch.FloatTensor:
        enc = torch.cat([state[key][index].to(self.device) for key in state if not "action_" in key], -1) # 1 x (state_dim - action_state_subvec_dim)
        assert enc.dim() == 2, f"Expected state encoding to have 2 dimensions, got {enc.dim()} with sizes {enc.size()}"
        assert enc.size(0) == 1, f"Expected state encoding to have size 1 in 1st dimension, got {enc.size()}"
        assert enc.size(1) == self.state_dim - self.action_state_subvec_dim, f"Expected state encoding to have size {self.state_dim - self.action_state_subvec_dim} in 2nd dimension, got {enc.size()}"

        if "ENCODED_action_enc" in state:
            action_encodings = state["ENCODED_action_enc"][index].to(self.device)
            num_actions = action_encodings.size(0)
            enc = enc.repeat(num_actions, 1)
            enc = torch.cat((enc, action_encodings), -1) # num_actions x state_dim
            assert enc.dim() == 2, f"Expected state encoding to have 2 dimensions, got {enc.dim()} with sizes {enc.size()}"
            assert enc.size(0) == num_actions, f"Expected state encoding to have size {num_actions} in 1st dimension, got {enc.size()}"
            assert enc.size(1) == self.state_dim, f"Expected state encoding to have size {self.state_dim} in 2nd dimension, got {enc.size()}"

        return enc

    def batch_state_vector(self, state: Dict[str, List[Any]], batch_size: int) -> List[torch.FloatTensor]:
        new_state = {key: state[key] for key in state if torch.is_tensor(state[key][0])}
        
        # re-calculate attention
        for attn in self.attentioninput:
            if attn.active:
                new_state[f"ENCODED_attn_{attn.name}"] = [None] * batch_size
                for idx in range(batch_size):
                    dialog_node = state[StateEntry.DIALOG_NODE.value][idx]
                    attn_enc = attn.encode(model=self.model, 
                        dialog_node=dialog_node,
                        dialog_node_text=dialog_node.content.text,
                        original_user_utterance=state[StateEntry.ORIGINAL_USER_UTTERANCE.value][idx],
                        current_user_utterance=state[StateEntry.CURRENT_USER_UTTERANCE.value][idx],
                        system_utterance_history=state[StateEntry.SYSTEM_UTTERANCE_HISTORY.value][idx],
                        user_utterance_history=state[StateEntry.USER_UTTERANCE_HISTORY.value][idx],
                        dialog_history=state[StateEntry.DIALOG_HISTORY.value][idx],
                        bst=state[StateEntry.BST.value][idx],
                        last_sysact=state[StateEntry.LAST_SYSACT.value][idx],
                        noise=state[StateEntry.NOISE.value][idx]
                    )
                    new_state[f"ENCODED_attn_{attn.name}"][idx] = attn_enc["attn_" + attn.name]
        # re-calculdate RNNs
        for rnn_encoder in self.stateinput.rnn_encoders:
            if rnn_encoder == "dialog_history":
                new_state[f"ENCODED_{rnn_encoder}"] = [hist.unsqueeze(0) for hist in list(self.stateinput.rnn_encoders[rnn_encoder].batch_encode(model=self.model, 
                                                                        system_utterance_history=state[StateEntry.SYSTEM_UTTERANCE_HISTORY.value],
                                                                        user_utterance_history=state[StateEntry.USER_UTTERANCE_HISTORY.value],
                                                                        dialog_history=state[StateEntry.DIALOG_HISTORY.value],
                                                                        noise=state[StateEntry.NOISE.value]
                                                                        ))]
            else:
                for idx in range(len(batch_size)):
                    new_state[f"ENCODED_{rnn_encoder}"] = self.stateinput.rnn_encoders[rnn_encoder].encode(model=self.model, dialog_node=state[StateEntry.DIALOG_NODE.value][idx],
                                                                            dialog_node_text=state[StateEntry.DIALOG_NODE.value][idx].content.text,
                                                                            original_user_utterance=state[StateEntry.ORIGINAL_USER_UTTERANCE.value][idx],
                                                                            current_user_utterance=state[StateEntry.CURRENT_USER_UTTERANCE.value][idx],
                                                                            system_utterance_history=state[StateEntry.SYSTEM_UTTERANCE_HISTORY.value][idx],
                                                                            user_utterance_history=state[StateEntry.USER_UTTERANCE_HISTORY.value][idx],
                                                                            dialog_history=state[StateEntry.DIALOG_HISTORY.value][idx],
                                                                            bst=state[StateEntry.BST.value][idx],
                                                                            last_sysact=state[StateEntry.LAST_SYSACT.value][idx],
                                                                            noise=state[StateEntry.NOISE.value][idx])
        return [self._concat_single_state_dict(new_state, i) for i in range(batch_size)]

    def batch_state_vector_from_obs(self, state: List[Dict[str, Any]], batch_size: int) -> List[torch.FloatTensor]:
        new_state = {key: [state[idx][key] for idx in range(batch_size)] for key in state[0] if torch.is_tensor(state[0][key])}

        # re-calculate attention
        for attn in self.attentioninput:
            if attn.active:
                new_state[f"ENCODED_attn_{attn.name}"] = [None] * batch_size
                for idx in range(batch_size):
                    dialog_node = state[idx][StateEntry.DIALOG_NODE.value]
                    attn_enc = attn.encode(model=self.model, 
                        dialog_node=dialog_node,
                        dialog_node_text=dialog_node.content.text,
                        original_user_utterance=state[idx][StateEntry.ORIGINAL_USER_UTTERANCE.value],
                        current_user_utterance=state[idx][StateEntry.CURRENT_USER_UTTERANCE.value],
                        system_utterance_history=state[idx][StateEntry.SYSTEM_UTTERANCE_HISTORY.value],
                        user_utterance_history=state[idx][StateEntry.USER_UTTERANCE_HISTORY.value],
                        dialog_history=state[idx][StateEntry.DIALOG_HISTORY.value],
                        bst=state[idx][StateEntry.BST.value],
                        last_sysact=state[idx][StateEntry.LAST_SYSACT.value],
                        noise=state[idx][StateEntry.NOISE.value]
                    )
                    new_state[f"ENCODED_attn_{attn.name}"][idx] = attn_enc["attn_" + attn.name]
        # re-calculdate RNNs
        for rnn_encoder in self.stateinput.rnn_encoders:
            if rnn_encoder == "dialog_history":
                new_state[f"ENCODED_{rnn_encoder}"] = [hist.unsqueeze(0) for hist in list(self.stateinput.rnn_encoders[rnn_encoder].batch_encode(model=self.model, 
                                                                        system_utterance_history=[state[idx][StateEntry.SYSTEM_UTTERANCE_HISTORY.value] for idx in range(batch_size)],
                                                                        user_utterance_history=[state[idx][StateEntry.USER_UTTERANCE_HISTORY.value] for idx in range(batch_size)],
                                                                        dialog_history=[state[idx][StateEntry.DIALOG_HISTORY.value] for idx in range(batch_size)],
                                                                        noise=state[0][StateEntry.NOISE.value]
                                                                        ))]
            else:
                for idx in range(batch_size):
                    new_state[f"ENCODED_{rnn_encoder}"] = self.stateinput.rnn_encoders[rnn_encoder].encode(model=self.model, dialog_node=state[idx][StateEntry.DIALOG_NODE.value],
                                                                            dialog_node_text=state[idx][StateEntry.DIALOG_NODE.value].content.text,
                                                                            original_user_utterance=state[idx][StateEntry.ORIGINAL_USER_UTTERANCE.value],
                                                                            current_user_utterance=state[idx][StateEntry.CURRENT_USER_UTTERANCE.value],
                                                                            system_utterance_history=state[idx][StateEntry.SYSTEM_UTTERANCE_HISTORY.value],
                                                                            user_utterance_history=state[idx][StateEntry.USER_UTTERANCE_HISTORY.value],
                                                                            dialog_history=state[idx][StateEntry.DIALOG_HISTORY.value],
                                                                            bst=state[idx][StateEntry.BST.value],
                                                                            last_sysact=state[idx][StateEntry.LAST_SYSACT.value],
                                                                            noise=state[idx][StateEntry.NOISE.value])
        return [self._concat_single_state_dict(new_state, i) for i in range(batch_size)]


    def state_vector(self, state: Dict[str, Any]) -> torch.FloatTensor:
        # re-calculate attention
        new_state = {key: state[key] for key in state if torch.is_tensor(state[key])}
        dialog_node = state[StateEntry.DIALOG_NODE.value]
        for attn in self.attentioninput:
            if attn.active:
                attn_enc = attn.encode(model=self.model, 
                    dialog_node=dialog_node,
                    dialog_node_text=dialog_node.content.text,
                    original_user_utterance=state[StateEntry.ORIGINAL_USER_UTTERANCE.value],
                    current_user_utterance=state[StateEntry.CURRENT_USER_UTTERANCE.value],
                    system_utterance_history=state[StateEntry.SYSTEM_UTTERANCE_HISTORY.value],
                    user_utterance_history=state[StateEntry.USER_UTTERANCE_HISTORY.value],
                    dialog_history=state[StateEntry.DIALOG_HISTORY.value],
                    bst=state[StateEntry.BST.value],
                    last_sysact=state[StateEntry.LAST_SYSACT.value],
                    noise=state[StateEntry.NOISE.value]
                )
                new_state[f"ENCODED_attn_{attn.name}"] = attn_enc["attn_" + attn.name]
        # re-calculdate RNNs
        for rnn_encoder in self.stateinput.rnn_encoders:
            new_state[f"ENCODED_{rnn_encoder}"] = self.stateinput.rnn_encoders[rnn_encoder].encode(model=self.model, dialog_node=dialog_node,
                                                                        dialog_node_text=dialog_node.content.text,
                                                                        original_user_utterance=state[StateEntry.ORIGINAL_USER_UTTERANCE.value],
                                                                        current_user_utterance=state[StateEntry.CURRENT_USER_UTTERANCE.value],
                                                                        system_utterance_history=state[StateEntry.SYSTEM_UTTERANCE_HISTORY.value],
                                                                        user_utterance_history=state[StateEntry.USER_UTTERANCE_HISTORY.value],
                                                                        dialog_history=state[StateEntry.DIALOG_HISTORY.value],
                                                                        bst=state[StateEntry.BST.value],
                                                                        last_sysact=state[StateEntry.LAST_SYSACT.value],
                                                                        noise=state[StateEntry.NOISE.value])
            

        # concatenate all non-action encodings
        enc = torch.cat([new_state[key] for key in new_state if not "action_" in key], -1) # 1 x (state_dim - action_state_subvec_dim)
        assert enc.dim() == 2, f"Expected state encoding to have 2 dimensions, got {enc.dim()} with sizes {enc.size()}"
        assert enc.size(0) == 1, f"Expected state encoding to have size 1 in 1st dimension, got {enc.size()}"
        assert enc.size(1) == self.state_dim - self.action_state_subvec_dim, f"Expected state encoding to have size {self.state_dim - self.action_state_subvec_dim} in 2nd dimension, got {enc.size()}"

        if "ENCODED_action_enc" in new_state:
            action_encodings = new_state["ENCODED_action_enc"]
            num_actions = action_encodings.size(0)
            enc = enc.repeat(num_actions, 1)
            enc = torch.cat((enc, action_encodings), -1) # num_actions x state_dim
            assert enc.dim() == 2, f"Expected state encoding to have 2 dimensions, got {enc.dim()} with sizes {enc.size()}"
            assert enc.size(0) == num_actions, f"Expected state encoding to have size {num_actions} in 1st dimension, got {enc.size()}"
            assert enc.size(1) == self.state_dim, f"Expected state encoding to have size {self.state_dim} in 2nd dimension, got {enc.size()}"

        return enc


    def get_action_masks(self, node_keys: List[int]) -> torch.FloatTensor:
        """
        Mask off impossible / bad actions

        Args:
            node_keys: (batch,)

        Returns:
            MASK for q values: (batch, num_actions), use this to calculate as  Q(s, a) + MASK(s, a)
                                                   where MASK(s, a) = 0 or -Inf
            If action masking is turned off, will return 0 matrix s.t. this function can be used for all scenarios
        """
        mask = torch.zeros(len(node_keys), self.num_actions, dtype=torch.float, device=self.device)
        if self.configuration.action_masking:
            for batch_idx, node_key in enumerate(node_keys):
                mask[batch_idx] += self.action_masks[node_key]
        return mask.bool()

