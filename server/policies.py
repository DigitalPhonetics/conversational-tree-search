

import os
from typing import Any, Dict, Tuple, Union
from copy import deepcopy

import torch

from data.dataset import GraphDataset, NodeType
from encoding.state import StateEncoding
from server.nlu import NLU
from server.webenv import RealUserEnvironmentWeb
from utils.utils import AutoSkipMode, EnvInfo, State
from config import ActionType

from sentence_transformers import util

def load_env(data: GraphDataset, nlu: NLU, sysParser, answer_parser, logic_parser, value_backend) -> RealUserEnvironmentWeb:
    # setup env
    env = RealUserEnvironmentWeb(dataset=data, nlu=nlu,
                        sys_token="SYSTEM", usr_token="USER", sep_token="",
                        max_steps=50, max_reward=150, user_patience=2,
                        system_parser=sysParser, answer_parser=answer_parser, logic_parser=logic_parser, value_backend=value_backend,
                        auto_skip=AutoSkipMode.NONE, stop_on_invalid_skip=False)
    return env


## TODO: 
## - Conversation history
## - GC
## - Logging?
## - Survey processing?
class ChatEngine:
    def __init__(self, user_id: str, socket, data: GraphDataset, state_encoding: StateEncoding, nlu: NLU, sysParser, answerParser, logicParser, valueBackend) -> None:
        self.user_id = user_id
        self.state_encoding = state_encoding
        self.user_env = load_env(data, nlu, sysParser, answerParser, logicParser, valueBackend) # contains .bst, .reset()
        self.socket = socket

    def next_action(self, obs: dict) -> Tuple[int, bool]:
        raise NotImplementedError        

    def start_dialog(self):
        self.user_env.reset()
        self.socket.write_message({"EVENT": "MSG", "VALUE": self.user_env.get_current_node_markup(), "CANDIDATES": self.user_env.get_current_node_answer_candidates()  })
        # wait for initial user utterance

    def user_reply(self, msg):
        if self.user_env.first_turn:
            self.set_initial_user_utterance(msg)
        else:
            # set n-th turn utternace and step
            if self.user_env.current_node.node_type == NodeType.VARIABLE:
                # check if user input is valid using NLU
                error_msg = self.user_env.check_variable_input(msg)
                if error_msg:
                    # unrecognized user input -> forward error message to UI
                    self.socket.write_message({"EVENT": "MSG", "VALUE": error_msg})
                else:
                    # valid user input -> step
                    self.step(action=self.action, utterance=msg)
            else:
                self.step(action=self.action, utterance=msg)

    def set_initial_user_utterance(self, msg):
        self.user_env.set_initial_user_utterance(msg)
        self.step()

    def step(self, action: Union[int, None] = None, utterance: Union[str, None] = None):
        done = False
        while not done:
            # get next action
            if isinstance(action, type(None)):
                self.action, self.intent = self.next_action(self.user_env.get_obs())
                print("ACTION:", self.action, "INTENT:", "FREE" if self.intent == True else "GUIDED")
            # perform next action
            if self.action == ActionType.ASK:
                # output current node
                if self.user_env.current_node.node_type in [NodeType.QUESTION, NodeType.VARIABLE]:
                    # wait for user input
                    if isinstance(utterance, type(None)):
                        self.socket.write_message({"EVENT": "MSG", "VALUE": self.user_env.get_current_node_markup(), "CANDIDATES": self.user_env.get_current_node_answer_candidates() })
                        return
                    else:
                       obs, reward, done = self.user_env.step(self.action, replayed_user_utterance=deepcopy(utterance))
                       action = None
                       utterance = None
                       continue 
                else:
                    self.socket.write_message({"EVENT": "MSG", "VALUE": self.user_env.get_current_node_markup(), "CANDIDATES": self.user_env.get_current_node_answer_candidates()  })
            # SKIP
            obs, reward, done = self.user_env.step(self.action, replayed_user_utterance="")


        


class FAQBaselinePolicy(ChatEngine):
    # TODO 
    # - load similarity model (could pass it via cache argument from server)
    # - encode dataset (could pre-cache the enocded node texts in a seperate file, only encode first user utterance live)
    # - return top-1 result only
    # - logging + metrics, since we don't need and enivronment here ?
    # - also ask Variables in case the response template contains any
    def __init__(self, user_id: str, socket, data: GraphDataset, state_encoding: StateEncoding, nlu: NLU, sysParser, answerParser, logicParser, valueBackend) -> None:
        super().__init__(user_id, socket, data, state_encoding, nlu, sysParser, answerParser, logicParser, valueBackend)
        self.node_idx_mapping, self.node_embeddings = self._embed_node_texts(data)

    def _embed_node_texts(self, data: GraphDataset) -> Tuple[Dict[int, int], torch.FloatTensor]:
        if not os.path.exists("./server/node_embeddings.pt"):
            # embed all info nodes
            print("Node embedding not found, creating...")
            embeddings = []
            node_idx_mapping = {} # mapping from node id -> embedding index
            for node_idx, node in enumerate(data.nodes_by_type[NodeType.INFO]):
                embeddings.append(self.state_encoding.cache.encode_text(state_input_key=State.NODE_TEXT, text=node.text, noise=0.0).cpu().view(1,-1))
                node_idx_mapping[node_idx] = node.key
            # save to file
            embeddings = torch.cat(embeddings, 0)
            torch.save({"node_idx_mapping": node_idx_mapping, "embeddings": embeddings}, "./server/node_embeddings.pt")
            print("Done")
            return node_idx_mapping, embeddings
        else:
            # load node embedding from file
            print("Node embedding found, loading...")
            data = torch.load("./server/node_embeddings.pt")
            print("Done")
            return data["node_idx_mapping"], data['embeddings']

    def step(self, action: Union[int, None] = None, utterance: Union[str, None] = None):
        # top-1 step
        # compare initial user utterance to all info node texts, select best match and return (ending dialog)
        user_enc = self.state_encoding.cache.encode_text(state_input_key=State.INITIAL_USER_UTTERANCE, text=self.user_env.initial_user_utterance, noise=0.0)
        cosine_scores = util.cos_sim(user_enc, self.node_embeddings)
        most_similar_idx = cosine_scores.view(-1).argmax(-1).item()
        # map back from embedding index -> node
        most_similar_node_id = self.node_idx_mapping[most_similar_idx]
        most_similar_node = self.user_env.data.nodes_by_key[most_similar_node_id]
        self.user_env.current_node = most_similar_node

        # output node 
        self.socket.write_message({"EVENT": "MSG", "VALUE": self.user_env.get_current_node_markup(), "CANDIDATES": [] })
        # stop dialog here


class GuidedBaselinePolicy(ChatEngine):
    def next_action(self, obs) -> Tuple[int, bool]:
        if self.user_env.last_action_idx == ActionType.ASK:
            # next action should be skip!
            if self.user_env.current_node.node_type == NodeType.QUESTION:
                # compare user input to answer candidates, and choose followup node based on similarity
                answer_enc = self.state_encoding.cache.encode_answer_text(self.user_env.current_node).view(len(self.user_env.current_node.answers), -1)
                user_enc = self.state_encoding.cache.encode_text(state_input_key=State.CURRENT_USER_UTTERANCE, text=self.user_env.current_user_utterance, noise=0.0).view(1, -1)
                print("ANS ENC", answer_enc.size())
                print("USER ENC", user_enc.size())
                cosine_scores = util.cos_sim(user_enc, answer_enc)
                print("COS SCORES", cosine_scores.size())
                most_similar_idx = cosine_scores.view(-1).argmax(-1).item()
                print("-> MOST SIMILAR IDX", most_similar_idx)
                return ActionType.SKIP.value + most_similar_idx, False # offset by 1, because answer 0 would be ASK
            elif self.user_env.current_node.node_type == NodeType.INFO:
                # no user input required - skip to connected node
                return ActionType.SKIP.value, False
            elif self.user_env.current_node.node_type == NodeType.VARIABLE:
                # should have exactly 1 answer - skip to that
                if len(self.user_env.current_node.answers) > 0:
                    return ActionType.SKIP.value, False # # jump to first (and only) answer: offset by 1, because answer 0 would be ASK
                else:
                    # TODO signal dialog end?
                    print("REACHED DIALOG END")
            else:
                raise Exception("Unexpected node type for asking:", self.user_env.current_node.node_type) 
        else:
            # last action was skip - should ask now!
            if self.user_env.current_node.node_type in [NodeType.INFO, NodeType.QUESTION, NodeType.VARIABLE]:
                return ActionType.ASK.value, False
            raise Exception("Unexpected node type for asking:", self.user_env.current_node.node_type)


class CTSPolicy(ChatEngine):
    def __init__(self, user_id: str, socket, data: GraphDataset, state_encoding: StateEncoding, nlu: NLU, sysParser, answerParser, logicParser, valueBackend, model) -> None:
        super().__init__(user_id, socket, data, state_encoding, nlu, sysParser, answerParser, logicParser, valueBackend)
        self.model = model

    def next_action(self, obs: Dict[EnvInfo, Any]) -> Tuple[int, bool]:
        # encode observation
        s = self.state_encoding.batch_encode(observation=[obs], sys_token="SYSTEM", usr_token="USER", sep_token="", noise=0.0) 
        # predict action & intent
        action, intent = self.model.predict(observation=s, deterministic=True)
        action = int(action)
        intent = intent.item()

        return action, intent