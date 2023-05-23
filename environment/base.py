from collections import defaultdict
from copy import deepcopy
from typing import Any, Tuple, Union, Dict

import torch

from chatbot.adviser.app.rl.utils import EnvInfo
from config import ActionType

from data.dataset import DialogNode, GraphDataset, NodeType

from chatbot.adviser.app.answerTemplateParser import AnswerTemplateParser
from chatbot.adviser.app.logicParser import LogicTemplateParser
from chatbot.adviser.app.parserValueProvider import RealValueBackend
from chatbot.adviser.app.rl.utils import AutoSkipMode

import random

class BaseEnv:
    def __init__(self, 
            dataset: GraphDataset, 
            sys_token: str, usr_token: str, sep_token: str,
            max_steps: int, max_reward: float, user_patience: int,
            answer_parser: AnswerTemplateParser, logic_parser: LogicTemplateParser,
            value_backend: RealValueBackend,
            auto_skip: AutoSkipMode) -> None:

        self.env_id = random.randint(0, 99999999)
        self.data = dataset

        self.max_steps = max_steps
        self.max_reward = max_reward

        self.user_patience = user_patience
        self.auto_skip_mode = auto_skip

        self.sys_token = sys_token
        self.usr_token = usr_token
        self.sep_token = sep_token
        
        self.answerParser = answer_parser
        self.logicParser = logic_parser
        self.value_backend = value_backend

        # stats
        self.reset_stats()

    def reset_stats(self):
        # success stats
        self.reached_goals = []
        self.asked_goals = []

        # coverage stats
        self.goal_node_coverage = defaultdict(int)
        self.node_coverage = defaultdict(int)
        self.coverage_answer_synonyms = defaultdict(int)
        self.coverage_variables = defaultdict(lambda: defaultdict(int))
        # self.reached_dialog_tree_end = 0 # TODO add
        self.current_episode = 0

        # node stats
        self.node_count = {node_type: 0 for node_type in NodeType}

        # action stats
        self.actioncount = {action_type: 0 for action_type in ActionType} # counts all ask- and skip-events
        self.actioncount_skips = {node_type: 0 for node_type in NodeType} # counts skip events per node type
        self.actioncount_skip_invalid = 0
        self.actioncount_asks = {node_type: 0 for node_type in NodeType}  # counts ask events per node type
        self.actioncount_ask_variable_irrelevant = 0
        self.actioncount_ask_question_irrelevant = 0
        self.actioncount_missingvariable = 0

    def pre_reset(self):
        self.current_step = 0
        self.user_answer_keys = defaultdict(int)
        self.visited_node_keys = defaultdict(int)
        self.episode_log = []
        self.bst = {}
       
        self.initial_user_utterance = "" 
        self.current_node = self.data.start_node.connected_node
        self.prev_node = None
        self.last_action_idx = ActionType.ASK.value # start by asking start node

    def post_reset(self):
        """
        Requires:
            self.goal: UserGoal or RealUserGoal
        """
        self.initial_user_utterance = deepcopy(self.goal.initial_user_utterance)

        self.goal_node_coverage[self.goal.goal_node.key] += 1
        self.current_user_utterance = deepcopy(self.initial_user_utterance)
        self.user_utterances_history = [deepcopy(self.initial_user_utterance)]
        self.system_utterances_history = [deepcopy(self.current_node.text)]
        self.visited_node_keys[self.current_node.key] = 1

        # coverage stats
        self.node_coverage[self.current_node.key] += 1

        # dialog stats
        self.current_episode += 1
        self.current_step = 0
        self.episode_reward = 0.0
        self.skipped_nodes = 0

        # task stats
        self.reached_goal_once = False
        self.asked_goal_once = False

        # Logging
        self.episode_log.append(f'{self.env_id}-{self.current_episode}$ ======== RESET =========')
        self.episode_log.append(f'{self.env_id}-{self.current_episode}$ GOAL: {self.goal.goal_node.key} {self.goal.goal_node.text[:100]}') 
        self.episode_log.append(f'{self.env_id}-{self.current_episode}$ CONSTRAINTS: {self.goal.constraints}')
        self.episode_log.append(f'{self.env_id}-{self.current_episode}$ INITIAL UTTERANCE: {self.initial_user_utterance}') 

        return self.get_obs()
    
    def check_user_patience_reached(self) -> bool:
        return self.visited_node_keys[self.current_node.key] > self.user_patience # +1 because  skip action to node already counts as 1 visit

    def reached_max_length(self) -> bool:
        return self.current_step == self.max_steps - 1 # -1 because first turn is counted as step 0

    def get_node_coverage(self):
        return len(self.node_coverage) / len(self.data.node_list)

    def get_coverage_faqs(self):
        return len(self.coverage_faqs) / len(self.data.question_list)

    def get_coverage_answer_synonyms(self):
        return len(self.coverage_answer_synonyms) / self.data.num_answer_synonyms
    
    def get_coverage_variables(self):
        return {
            "STADT": len(self.coverage_variables["STADT"]) / len(self.data.city_list),
            "LAND": len(self.coverage_variables["LAND"]) / len(self.data.country_list)
        }

    def get_obs(self) -> Dict[EnvInfo, Any]:
        return {
                EnvInfo.DIALOG_NODE: self.current_node,
                EnvInfo.PREV_NODE: self.prev_node,
                EnvInfo.LAST_SYSTEM_ACT: self.last_action_idx,
                EnvInfo.BELIEFSTATE: deepcopy(self.bst),
                EnvInfo.EPISODE_REWARD: self.episode_reward,
                EnvInfo.EPISODE_LENGTH: self.current_step,
                EnvInfo.EPISODE: self.current_episode,
                EnvInfo.REACHED_GOAL_ONCE: self.reached_goal(),
                EnvInfo.ASKED_GOAL: self.asked_goal(),
                EnvInfo.INITIAL_USER_UTTERANCE: deepcopy(self.initial_user_utterance),
                EnvInfo.CURRENT_USER_UTTERANCE: deepcopy(self.current_user_utterance),
                EnvInfo.USER_UTTERANCE_HISTORY: deepcopy(self.user_utterances_history),
                EnvInfo.SYSTEM_UTTERANCE_HISTORY: deepcopy(self.system_utterances_history)
        }

    def get_transition(self, answer_index: int) -> Union[DialogNode, None]:
        # num_answers = self.current_node.answers.count()
        num_answers = len(self.current_node.answers)
        if self.current_node.connected_node:
            assert num_answers == 0
            # we have a directly connected node instead of answers -> only skip index = 0 is valid
            if answer_index == 0:
                return self.current_node.connected_node
            else:
                # invalid skip index: > 0
                self.episode_log.append(f'{self.env_id}-{self.current_episode}$ -> INVALID TRANSITION TO CONNECTED_NODE {answer_index}')
                return None
        # we dont' have a directly connected node, but maybe answers
        if answer_index >= num_answers:
            # invalid answer index (we have less answers than chosen index)
            self.episode_log.append(f'{self.env_id}-{self.current_episode}$ -> INVALID TRANSITION TO ANSWER {answer_index}, HAS ONLY {num_answers}')
            return None # invalid action: stay at current node
        
        next_node = self.current_node.answer_by_index(answer_index).connected_node
        self.node_coverage[next_node.key] += 1
        return next_node

    def auto_skip(self):
        # TODO
        raise NotImplementedError
        # auto-skipping
        # # auto-skipping after asking
        # if self.current_node.node_type in [NodeType.QUESTION.value, NodeType.INFO.value, NodeType.VARIABLE.value]:
        #     if self.current_node.answer_count() > 0:
        #         # check that current node has answers!
        #         # if so, choose best fitting one (w.r.t. user text)
        #         # similarities = self.similarity_model.encode(current_user_utterance=self.current_user_utterance, dialog_node=self.current_node) # 1 x answers
        #         # skip_action_idx = similarities.view(-1).argmax()
        #         # skip_action = self.current_node.answer_by_index(skip_action_idx)
                
        #         if self.is_faq_mode:
        #             if self.current_node.node_type == NodeType.QUESTION.value:
        #                 if self.auto_skip == AutoSkipMode.ORACLE:
        #                     # semantic level: choose correct answer to jump to (assume perfect NLU)
        #                     response: UserResponse = self.user_answer_keys[self.current_node.key]
        #                     skip_action = Data.objects[self.version].answer_by_key(response.answer_key)
        #                 else:
        #                     # utterance level: choose answer to jump to by similarity
        #                     similarities = self.similarity_model.encode(current_user_utterance=self.current_user_utterance, dialog_node=self.current_node, noise=self.noise) # 1 x actions x 1
        #                     skip_action_idx = similarities.view(-1).argmax(-1).item()
        #                     skip_action = self.current_node.answer_by_index(skip_action_idx)
        #             else:
        #                 assert self.current_node.answer_count() == 1
        #                 skip_action = self.current_node.answers[0]
        #         else:
        #             if self.auto_skip == AutoSkipMode.ORACLE:
        #                 # semantic level: choose correct answer to jump to (assume perfect NLU)
        #                 skip_action = self.current_node.answer_by_goalnode_key(self.goal.goal_node.key)
        #             else:
        #                 # utterance level: choose answer to jump to by similarity
        #                 if self.current_node.node_type == NodeType.QUESTION.value:
        #                     similarities = self.similarity_model.encode(current_user_utterance=self.current_user_utterance, dialog_node=self.current_node, noise=self.noise) # 1 x actions x 1
        #                     skip_action_idx = similarities.view(-1).argmax(-1).item()
        #                     skip_action = self.current_node.answer_by_index(skip_action_idx)
        #                 else:
        #                     assert self.current_node.answer_count() == 1
        #                     skip_action = self.current_node.answers[0]


        #         # jump to connected node
        #         self.current_node = Data.objects[self.version].node_by_key(skip_action.connected_node_key)
        #         self.user_utterances_history.append("")
        #         self.system_utterances_history.append(deepcopy(self.current_node.content.text))
        #     elif self.current_node.connected_node_key:
        #         # no answers, but connected node -> skip to that node
        #         self.current_node = Data.objects[self.version].node_by_key(self.current_node.connected_node_key)
        #         self.user_utterances_history.append("")
        #         self.system_utterances_history.append(deepcopy(self.current_node.content.text))
        #     if self.current_node.key == self.goal.goal_node.key:
        #         # check if skipped-to node is goal node
        #         if self.is_faq_mode:
        #             self.reached_goal_once = True
        #         else:
        #             self.reached_goals.append(1.0)

        #     # update history
        #     self.episode_log.append(f'{self.env_id}-{self.current_episode}$ AUTOSKIP TO NODE: {self.current_node.node_type} - {self.current_node.key} - {self.current_node.content.text[:75]}')
        #     self.last_action_idx = real_action
        #     # update counters
        #     self._update_node_counters()
        #     self.visited_node_keys[self.current_node.key] += 1
        
    def reached_goal(self) -> Union[bool, float]:
        raise NotImplementedError

    def asked_goal(self) -> Union[bool, float]:
        raise NotImplementedError

    def ask(self, replayed_user_utterance: Tuple[str, None]) -> Tuple[bool, float]:
        raise NotImplementedError

    def skip(self, answer_index: int) -> Tuple[bool, float]:
        raise NotImplementedError 

    def step(self, action: int, replayed_user_utterance: Tuple[str, None] = None) -> Tuple[torch.FloatTensor, float, bool, Dict[EnvInfo, Any]]:
        reward = 0.0
        done = False 
        self.prev_node = self.current_node

        # check if dialog should end
        if self.check_user_patience_reached(): 
            reward -= self.max_reward  # bad
            done = True
            self.episode_log.append(f'{self.env_id}-{self.current_episode}$ REACHED MAX USER PATIENCE')
        elif self.reached_max_length():
            done = True # already should have large negtative reward (expect if in guided mode, where max length could even be desired)
            self.episode_log.append(f'{self.env_id}-{self.current_episode}$ REACHED MAX LENGTH')
        else:
            assert self.current_node.node_type != NodeType.LOGIC
            if action == ActionType.ASK:
                done, reward = self.ask(replayed_user_utterance)
            else:
                done, reward = self.skip(action-1) # get answer index by shifting actions to the left
                
            self.episode_log.append(f'{self.env_id}-{self.current_episode}$ -> USER UTTERANCE: {self.current_user_utterance}')
            self.episode_log.append(f'{self.env_id}-{self.current_episode}$ TO NODE: {self.current_node.node_type.value} - {self.current_node.key} - {self.current_node.text[:100]}')

            # update history
            self.last_action_idx = action
            self.user_utterances_history.append(str(deepcopy(self.current_user_utterance)))
            self.system_utterances_history.append(deepcopy(self.current_node.text))

            # update counters
            self.visited_node_keys[self.current_node.key] += 1
            self.current_step += 1
            self.update_node_counters()
            self.update_action_counters(action)

            if (not done) and self.goal.goal_node and self.auto_skip_mode != AutoSkipMode.NONE and self.last_action_idx == ActionType.ASK:
                self.auto_skip()

            # handle logic node auto-transitioning here
            if not done:
                logic_reward, logic_done, did_handle_logic_node = self.handle_logic_nodes()
                if did_handle_logic_node:
                    reward += logic_reward
                    done = logic_done
                self.episode_log.append(f'{self.env_id}-{self.current_episode}$ -> TURN REWARD: {reward}')

                if not self.goal.goal_node:
                    done = True # check if we reached end of dialog tree
                    self.episode_log.append(f'{self.env_id}-{self.current_episode}$ -> REACHED TREE END')

        self.episode_reward += reward
        if done:
            self.reached_goals.append(float(self.reached_goal_once))
            self.asked_goals.append(float(self.asked_goal_once))
            self.episode_log.append(f'{self.env_id}-{self.current_episode}$ -> TURN REWARD: {reward}')
            self.episode_log.append(f'{self.env_id}-{self.current_episode}$ -> FINAL REWARD: {self.episode_reward}')

        obs = self.get_obs()
        return obs, reward/self.max_reward, done


    def update_action_counters(self, action: int):
        action_type = ActionType.ASK if action == ActionType.ASK else ActionType.SKIP
        self.actioncount[action_type] += 1

        if action_type == ActionType.SKIP and self.last_action_idx >= ActionType.SKIP: # it's only a REAL skip if we didn't ask before
            self.actioncount_skips[self.current_node.node_type] += 1
            self.skipped_nodes += 1

    def update_node_counters(self):
        if self.current_node:
            self.node_count[self.current_node.node_type] += 1

    def handle_logic_nodes(self) -> Tuple[float, bool, bool]:
        reward = 0
        done = False
        did_handle_logic_node = False
        while self.current_node and self.current_node.node_type == NodeType.LOGIC:
            did_handle_logic_node = True
            self.node_count[NodeType.LOGIC] += 1
            lhs = self.current_node.text
            var_name = lhs.lstrip("{{").strip() # form: {{ VAR_NAME
            if var_name in self.bst:
                # don't assign reward, is automatic transition without agent interaction
                # evaluate statement, choose correct branch and skip to connected node
                default_answer = None
                # for idx, answer in enumerate(self.current_node.answers.all()):
                for answer in self.current_node.answers:
                    # check if full statement {{{lhs rhs}}} evaluates to True
                    rhs = answer.text
                    if not "DEFAULT" in rhs: # handle DEFAULT case last!
                        if self._fillLogicTemplate(f"{lhs} {rhs}"):
                            # evaluates to True, follow this path!
                            default_answer = answer
                            break
                    else:
                        default_answer = answer
                self.current_node = default_answer.connected_node
                if self.current_node:
                    self.node_coverage[self.current_node.key] += 1
                    self.episode_log.append(f"{self.env_id}-{self.current_episode}$ -> AUTO SKIP LOGIC NODE: SUCCESS {self.current_node.key}")
                else:
                    self.episode_log.append(f"{self.env_id}-{self.current_episode}$ -> AUTO SKIP LOGIC NODE: SUCCESS, but current node NULL {self.current_node}")
                    done = True
            else:
                # we don't know variable (yet?) -> punish and stop
                reward -= self.max_reward
                self.actioncount_missingvariable += 1
                self.episode_log.append(f"{self.env_id}-{self.current_episode}$ -> AUTO SKIP LOGIC NODE: FAIL, VAR {var_name} not in BST -> {self.current_node.key}")
                done = True
                break

        # if did_handle_logic_node and not done:
        #     self.post_handle_logic_nodes()
        
        return reward, done, did_handle_logic_node

    def _fillLogicTemplate(self, delexicalized_utterance: str):
        return self.logicParser.parse_template(delexicalized_utterance, self.value_backend, self.bst)