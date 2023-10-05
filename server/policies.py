

from collections import defaultdict
import os
from typing import Any, Dict, Tuple, Union
from copy import deepcopy
import pandas as pd
import torch

from data.dataset import GraphDataset, NodeType
from encoding.state import StateEncoding
from server.nlu import NLU
from server.webenv import RealUserEnvironmentWeb
from utils.utils import AutoSkipMode, EnvInfo, State
from config import ActionType

from sentence_transformers import util

import re
url_pattern = re.compile(r'(<a\s+[^>]*href=")([^"]*)(")([^>]*>)')

def load_env(user_id: int, data: GraphDataset, nlu: NLU, sysParser, answer_parser, logic_parser, value_backend) -> RealUserEnvironmentWeb:
    # setup env
    env = RealUserEnvironmentWeb(user_id=user_id, dataset=data, nlu=nlu,
                        sys_token="SYSTEM", usr_token="USER", sep_token="",
                        max_steps=100, max_reward=150, user_patience=15,
                        system_parser=sysParser, answer_parser=answer_parser, logic_parser=logic_parser, value_backend=value_backend,
                        auto_skip=AutoSkipMode.NONE, stop_on_invalid_skip=False)
    return env


## TODO: 
## - GC
class ChatEngine:
    def __init__(self, user_id: str, socket, data: GraphDataset, state_encoding: StateEncoding, nlu: NLU, sysParser, answerParser, logicParser, valueBackend) -> None:
        self.user_id = user_id
        self.state_encoding = state_encoding
        self.user_env = load_env(user_id, data, nlu, sysParser, answerParser, logicParser, valueBackend) # contains .bst, .reset()
        self.socket = socket
        self.last_sys_nodes = set();

    def next_action(self, obs: dict) -> Tuple[int, bool]:
        raise NotImplementedError        

    def start_dialog(self, goal_node_id: int):
        self.user_env.reset(goal_node_id=goal_node_id)
        self.socket.write_message({"EVENT": "MSG", "VALUE": self.user_env.get_current_node_markup(), "CANDIDATES": self.user_env.get_current_node_answer_candidates(), "NODE_TYPE": self.user_env.current_node.node_type.value  })
        # wait for initial user utterance

    def user_reply(self, msg):
        self.last_sys_nodes = set();
        if self.user_env.first_turn:
            self.set_initial_user_utterance(msg)
        else:
            # set n-th turn utternace and step
            if self.user_env.current_node.node_type == NodeType.VARIABLE:
                # check if user input is valid using NLU
                error_msg = self.user_env.check_variable_input(msg)
                if error_msg:
                    # unrecognized user input -> forward error message to UI
                    self.socket.write_message({"EVENT": "MSG", "VALUE": error_msg, "NODE_TYPE": self.user_env.current_node.node_type.value})
                else:
                    # valid user input -> step
                    self.step(action=self.action, utterance=msg)
            else:
                self.step(action=self.action, utterance=msg)

    def set_initial_user_utterance(self, msg):
        self.last_sys_nodes = set();
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
                if (not isinstance(utterance, type(None))) and self.user_env.current_node.key in self.last_sys_nodes:
                    # we are about to repeat - stop dialog
                    done = True
                    self.user_env.episode_log.append(f'{self.user_id}-{self.user_env.current_episode}$ => STOPPED POLICY BEFORE REPEATING')
                    break
                self.last_sys_nodes.add(self.user_env.current_node.key)
                # output current node
                if self.user_env.current_node.node_type in [NodeType.QUESTION, NodeType.VARIABLE]:
                    # wait for user input
                    if isinstance(utterance, type(None)):
                        self.socket.write_message({"EVENT": "MSG", "VALUE": self.user_env.get_current_node_markup(), "CANDIDATES": self.user_env.get_current_node_answer_candidates(), "NODE_TYPE": self.user_env.current_node.node_type.value })
                        return
                    obs, reward, done = self.user_env.step(self.action, replayed_user_utterance=deepcopy(utterance))
                    action = None
                    utterance = None
                    continue
                else:
                    self.socket.write_message({"EVENT": "MSG", "VALUE": self.user_env.get_current_node_markup(), "CANDIDATES": self.user_env.get_current_node_answer_candidates(), "NODE_TYPE": self.user_env.current_node.node_type.value  })
            # SKIP
            obs, reward, done = self.user_env.step(self.action, replayed_user_utterance="")
        if done:
            self.socket.write_message({"EVENT": "DIALOG_ENDED", "VALUE": True})


class FAQBaselinePolicy(ChatEngine):
    def __init__(self, user_id: str, socket, data: GraphDataset, state_encoding: StateEncoding, nlu: NLU, sysParser, answerParser, logicParser, valueBackend,
                    node_idx_mapping, node_embeddings, node_markup, country_list, country_city_list) -> None:
        super().__init__(user_id, socket, data, state_encoding, nlu, sysParser, answerParser, logicParser, valueBackend)
        self.country_list = country_list
        self.country_city_list = country_city_list
        self.node_idx_mapping = node_idx_mapping
        self.node_embeddings = node_embeddings
        self.node_markup = node_markup

    @staticmethod
    def get_country_city_map(data: GraphDataset):
        hotel_costs = defaultdict(lambda: dict())
        country_list = set()
        country_city_list = []

        content = pd.read_excel(os.path.join(data.resource_dir, "en/reimburse/TAGEGELD_AUSLAND.xlsx"))
        for idx, row in content.iterrows():
            country = row['Land']
            city = row['Stadt']
            country_list.add(country)
            country_city_list.append((country, city))
        return list(country_list), country_city_list

    @staticmethod
    def embed_node_texts(data: GraphDataset, state_encoding, system_parser, country_list, country_city_list, value_backend) -> Tuple[Dict[int, int], torch.FloatTensor]:
        if not os.path.exists("./server/node_embeddings.pt"):
            # embed all info nodes
            print("Node embedding not found, creating...")
            embeddings = []
            node_idx_mapping = {} # mapping from node id -> embedding index
            node_text_idx = 0
            node_markup = []
            for node in data.nodes_by_type[NodeType.INFO]:
                variables = system_parser.find_variables(node.text)
                if "CITY" in variables:
                    # replace country and city
                    for country, city in country_city_list:
                        text = system_parser.parse_template(node.text, value_backend, {"COUNTRY": country, "CITY": city})
                        embeddings.append(state_encoding.cache.encode_text(state_input_key=State.NODE_TEXT, text=text, noise=0.0).cpu().view(1,-1))
                        node_idx_mapping[node_text_idx] = node.key
                        node_markup.append(system_parser.parse_template(node.markup, value_backend, {"COUNTRY": country, "CITY": city}))
                        node_text_idx += 1
                elif "COUNTRY" in variables:
                    # replace country only
                    for country in country_list:
                        text = system_parser.parse_template(node.text, value_backend, {"COUNTRY": country})
                        embeddings.append(state_encoding.cache.encode_text(state_input_key=State.NODE_TEXT, text=text, noise=0.0).cpu().view(1,-1))
                        node_idx_mapping[node_text_idx] = node.key
                        node_markup.append(system_parser.parse_template(node.markup, value_backend, {"COUNTRY": country}))
                        node_text_idx += 1
                else:
                    # normal text, don't replace anything
                    embeddings.append(state_encoding.cache.encode_text(state_input_key=State.NODE_TEXT, text=node.text, noise=0.0).cpu().view(1,-1))
                    node_idx_mapping[node_text_idx] = node.key
                    node_markup.append(node.markup)
                    node_text_idx += 1
            # save to file
            embeddings = torch.cat(embeddings, 0)
            torch.save({"node_idx_mapping": node_idx_mapping, "embeddings": embeddings, "node_markup": node_markup}, "./server/node_embeddings.pt")
            print("Done")
            return node_idx_mapping, embeddings, node_markup
        else:
            # load node embedding from file
            print("Node embedding found, loading...")
            data = torch.load("./server/node_embeddings.pt")
            print("Done")
            return data["node_idx_mapping"], data['embeddings'], data['node_markup']
        
    def set_initial_user_utterance(self, msg):
        self.user_env.set_initial_user_utterance(msg, check_variables=False)
        self.step()

    def get_node_markup(self, node_idx: int) -> str:
        # replace links with alert
        markup = url_pattern.sub(r"""\1#\3 onclick="open_link_info()"\4""", self.node_markup[node_idx])
        return markup


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

        self.user_env.episode_log.append(f'{self.user_id}-{self.user_env.current_episode}$: SKIP + ASK: Node {most_similar_node_id}, {most_similar_node.text[:50]}')
        if self.user_env.goal.goal_node_key == most_similar_node_id:
            self.user_env.episode_log.append(f'{self.user_id}-{self.user_env.current_episode}$=> REACHED GOAL ONCE: {self.user_env.reached_goal_once}')

        # output node 
        self.socket.write_message({"EVENT": "MSG", "VALUE": self.get_node_markup(most_similar_idx), "CANDIDATES": [], "NODE_TYPE": self.user_env.current_node.node_type.value })
        # stop dialog here
        self.socket.write_message({"EVENT": "DIALOG_ENDED", "VALUE": True})


class GuidedBaselinePolicy(ChatEngine):
    def next_action(self, obs) -> Tuple[int, bool]:
        if self.user_env.variable_already_known():
            # SKIP this node, because we already know the variable's value
            return ActionType.SKIP, False # jump to first (and only) answer: offset by 1, because answer 0 would be ASK
        
        if self.user_env.last_action_idx == ActionType.ASK:
            # next action should be skip!
            if self.user_env.current_node.node_type == NodeType.QUESTION:
                # compare user input to answer candidates, and choose followup node based on similarity
                answer_enc = self.state_encoding.cache.encode_answer_text(self.user_env.current_node).view(len(self.user_env.current_node.answers), -1)
                user_enc = self.state_encoding.cache.encode_text(state_input_key=State.CURRENT_USER_UTTERANCE, text=self.user_env.current_user_utterance, noise=0.0).view(1, -1)
                # print("ANS ENC", answer_enc.size())
                # print("USER ENC", user_enc.size())
                cosine_scores = util.cos_sim(user_enc, answer_enc)
                # print("COS SCORES", cosine_scores.size())
                most_similar_idx = cosine_scores.view(-1).argmax(-1).item()
                print("-> MOST SIMILAR ANSWER IDX", most_similar_idx)
                return ActionType.SKIP.value + most_similar_idx, False # offset by 1, because answer 0 would be ASK
            elif self.user_env.current_node.node_type == NodeType.INFO:
                # no user input required - skip to connected node
                return ActionType.SKIP.value, False
            elif self.user_env.current_node.node_type == NodeType.VARIABLE:
                # should have exactly 1 answer - skip to that
                if len(self.user_env.current_node.answers) > 0:
                    return ActionType.SKIP.value, False # jump to first (and only) answer: offset by 1, because answer 0 would be ASK
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

        if action == ActionType.ASK and self.user_env.variable_already_known():
            # SKIP this node, because we already know the variable's value
            print("Hard-coded variable node skip")
            action = ActionType.SKIP.value # jump to first (and only) answer: offset by 1, because answer 0 would be ASK

        return action, intent