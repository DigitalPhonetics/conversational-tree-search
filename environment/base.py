from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
import itertools
import random
from statistics import mean
from typing import Any, Optional, Tuple, Type, Union, Dict, List
from enum import IntEnum

import gym
from gym import spaces
import logging
from chatbot.adviser.app.rl.goal import ImpossibleGoalError, UserResponse, UserGoalGenerator
from chatbot.adviser.app.rl.utils import EnvInfo, StateEntry, rand_remove_questionmark

from data.dataset import Answer, DialogNode, GraphDataset, NodeType

from chatbot.adviser.app.answerTemplateParser import AnswerTemplateParser
from chatbot.adviser.app.logicParser import LogicTemplateParser
from chatbot.adviser.app.systemTemplateParser import SystemTemplateParser
from chatbot.adviser.app.parserValueProvider import RealValueBackend
# from chatbot.adviser.app.rl.dialogtree import DialogTree
from chatbot.adviser.app.rl.goal import VariableValue
# from chatbot.adviser.app.rl.goal import DummyGoal, ImpossibleGoalError, UserGoalGenerator, UserResponse, VariableValue
# from chatbot.adviser.app.rl.spaceAdapter import SpaceAdapter
from chatbot.adviser.app.rl.utils import AutoSkipMode #, EnvironmentMode, NodeType, StateEntry, EnvInfo, _load_a1_laenderliste, _load_answer_synonyms, rand_remove_questionmark
# from chatbot.adviser.app.rl.dataset import DialogAnswer, DialogNode
# from chatbot.adviser.app.rl.dataset import Dataset


class ActionType(IntEnum):
    ASK = 0
    SKIP = 1
    
@dataclass
class EnvironmentConfig:
    guided_free_ratio: float
    auto_skip: bool
    normalize_rewards: bool
    max_steps: int
    user_patience: int
    stop_when_reaching_goal: bool 
    num_train_envs: int 
    num_val_envs: int
    num_test_envs: int
    sys_token: Optional[str] = ""
    usr_token: Optional[str] = ""
    sep_token: Optional[str] = ""


# TODO restructure env
# TODO restructure space adapter
# TODO restructure models

# TODO add new config options:
# - chain_dialog_history -> sys_token, usr_token, sep_token

# TODO update function usage:
# - chain_dialog_history -> now returns 5 instead of 2 entries per list item
# - handle_logic_node -> doesn't return reward anymore, doesn't handle chained logic nodes anymore, doesn't log, doesn't update last_node, throws MissingBSTValue

# TODO how do we get a real user in ?

class MissingBSTValue(Exception):
    def __init__(self, var_name: str, bst: dict) -> None:
        super().__init__(var_name, bst)


def chain_dialog_history(sys_utterances: List[str], usr_utterances: List[str], sys_token: str = "", usr_token: str = "", sep_token: str = "") -> List[Tuple[str,str,str,str,str]]:
    """
    Interleave system and user utterances to a combined history.

    Args:
        sys_utterances: List of all system utterances (one turn per list entry)
        usr_utterances: List of all user utterances (one turn per list entry)
        sys_token: Token appended to each system utterance, e.g. "[SYS]" -> [SYS] sys turn 1 [USR] usr turn 1 [SYS] sys turn 2 ...
        usr_token: Token appended to each user utterance, e.g. "[SYS]" -> [SYS] sys turn 1 [USR] usr turn 1 [SYS] sys turn 2 ...
        sep_token Seperator token added between each system and user utterance, e.g. "[SEP]" -> sys 1 [SEP] usr turn 1 [SEP] sys turn 2 ...

    Returns: 
        List[Tuple(sys_token, sys_turn, sep_token, usr_token, usr_turn)]
    """
    turns = len(sys_utterances)
    assert len(usr_utterances) == turns
    return list(itertools.chain(zip([sys_token] * turns, [utterance for utterance in sys_utterances], [sep_token] * turns, [usr_token] * turns, [utterance for utterance in usr_utterances])))



class BaseEnv:
    def __init__(self, env_id: int, dataset: GraphDataset, 
            sys_token: str, usr_token: str, sep_token: str,
            max_steps: int, max_reward: float, user_patience: int,
            answer_parser: AnswerTemplateParser, logic_parser: LogicTemplateParser,
            value_backend: RealValueBackend,
            auto_skip: AutoSkipMode) -> None:
        self.env_id = env_id
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

        # coverage stats
        self.goal_node_coverage = defaultdict(int)
        self.node_coverage = defaultdict(int)
        self.coverage_synonyms = defaultdict(int)
        self.coverage_variables = defaultdict(lambda: defaultdict(int))
        self.reached_dialog_tree_end = 0
        self.current_episode = 0

    def pre_reset(self):
        self.current_step = 0
        self.user_answer_keys = defaultdict(int)
        self.visited_node_keys = defaultdict(int)
        self.episode_log = []
        self.bst = {}
        
        self.current_node = self.data.start_node.connected_node
        self.last_action_idx = ActionType.ASK.value # start by asking start node

    def post_reset(self):
        """
        Requires:
            self.goal_node: DialogNode
            self.initial_utterance: str
        """
        self.goal_node_coverage[self.goal_node.key] += 1
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

        # Logging
        self.episode_log.append(f'{self.env_id}-{self.current_episode}$ ======== RESET =========')
        self.episode_log.append(f'{self.env_id}-{self.current_episode}$ GOAL: {self.goal_node.key} {self.goal_node.text[:100]}') 
        self.episode_log.append(f'{self.env_id}-{self.current_episode}$ CONSTRAINTS: {self.constraints}')
        self.episode_log.append(f'{self.env_id}-{self.current_episode}$ INITIAL UTTERANCE: {self.initial_user_utterance}') 
    
    def check_user_patience_reached(self) -> bool:
        return self.visited_node_keys[self.current_node.key] > self.user_patience # +1 because  skip action to node already counts as 1 visit

    def reached_max_length(self) -> bool:
        return self.current_step == self.max_steps - 1 # -1 because first turn is counted as step 0

    def get_node_coverage(self):
        return len(self.node_coverage) / len(self.data.node_list)

    def get_coverage_faqs(self):
        return len(self.coverage_faqs) / len(self.data.question_list)

    def get_coverage_synonyms(self):
        return len(self.coverage_synonyms) / self.data.num_answer_synonyms
    
    def get_coverage_variables(self):
        return {
            "STADT": len(self.coverage_variables["STADT"]) / len(self.data.city_list),
            "LAND": len(self.coverage_variables["LAND"]) / len(self.data.country_list)
        }

    def get_obs(self) -> Dict[StateEntry, Any]:
        return {
            StateEntry.DIALOG_NODE.value: self.current_node,
            StateEntry.DIALOG_NODE_KEY.value: self.current_node.key,
            StateEntry.ORIGINAL_USER_UTTERANCE.value: deepcopy(self.initial_user_utterance),
            StateEntry.CURRENT_USER_UTTERANCE.value: deepcopy(self.current_user_utterance),
            StateEntry.SYSTEM_UTTERANCE_HISTORY.value: deepcopy(self.system_utterances_history),
            StateEntry.USER_UTTERANCE_HISTORY.value: deepcopy(self.user_utterances_history),
            StateEntry.DIALOG_HISTORY.value: chain_dialog_history(sys_utterances=self.system_utterances_history, usr_utterances=self.user_utterances_history,
                                                            sys_token=self.sys_token, usr_token=self.usr_token, sep_token=self.sep_token),
            StateEntry.BST.value: deepcopy(self.bst),
            StateEntry.LAST_SYSACT.value: self.last_action_idx,
            # StateEntry.NOISE.value: self.noise
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
        #                 skip_action = self.current_node.answer_by_goalnode_key(self.goal_node.key)
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
        #     if self.current_node.key == self.goal_node.key:
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

    def step(self, action: int, replayed_user_utterance: Tuple[str, None] = None) -> Tuple[float, bool]:
        reward = 0.0
        done = False 
        prev_node_key = self.current_node.key

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

            if (not done) and self.goal_node and self.auto_skip_mode != AutoSkipMode.NONE and self.last_action_idx == ActionType.ASK:
                self.auto_skip()

            # handle logic node auto-transitioning here
            if not done:
                logic_reward, logic_done, did_handle_logic_node = self.handle_logic_nodes()
                if did_handle_logic_node:
                    reward += logic_reward
                    done = logic_done
                self.episode_log.append(f'{self.env_id}-{self.current_episode}$ -> TURN REWARD: {reward}')

                if not self.goal_node:
                    done = True # check if we reached end of dialog tree
                    self.episode_log.append(f'{self.env_id}-{self.current_episode}$ -> REACHED TREE END')

        self.episode_reward += reward
        if done:
            self.episode_log.append(f'{self.env_id}-{self.current_episode}$ -> TURN REWARD: {reward}')
            self.episode_log.append(f'{self.env_id}-{self.current_episode}$ -> FINAL REWARD: {self.episode_reward}')

        info = {EnvInfo.NODE_KEY: self.current_node.key,
                EnvInfo.PREV_NODE_KEY: prev_node_key,
                EnvInfo.EPISODE_REWARD: self.episode_reward,
                EnvInfo.EPISODE_LENGTH: self.current_step,
                EnvInfo.EPISODE: self.current_episode,
                EnvInfo.REACHED_GOAL_ONCE: self.reached_goal(),
                EnvInfo.ASKED_GOAL: self.asked_goal()
        }
        return self.get_obs(), reward/self.max_reward, done, info

    def update_action_counters(self, action: int):
        action_type = ActionType.ASK if action == ActionType.ASK else ActionType.SKIP
        self.actioncount[action_type] += 1

        if action_type == ActionType.SKIP and self.last_action_idx >= ActionType.SKIP: # it's only a REAL skip if we didn't ask before
            self.actioncount_skips[self.current_node.node_type] += 1
            self.skipped_nodes += 1

    def update_node_counters(self):
        if self.current_node:
            self.node_count[self.current_node.node_type] += 1

    def post_handle_logic_nodes(self, did_handle_logic_nodes):
        pass

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

        if did_handle_logic_node and not done:
            self.post_handle_logic_nodes()
        
        return reward, done, did_handle_logic_node

    def _fillLogicTemplate(self, delexicalized_utterance: str):
        return self.logicParser.parse_template(delexicalized_utterance, self.value_backend, self.bst)


class _GuidedEnvironment(BaseEnv):
    def __init__(self, env_id: int, dataset: GraphDataset, sys_token: str, usr_token: str, sep_token: str,
            max_steps: int, max_reward: float, user_patience: int,
            answer_parser: AnswerTemplateParser, logic_parser: LogicTemplateParser,
            value_backend: RealValueBackend,
            auto_skip: AutoSkipMode) -> None:
        super().__init__(env_id=env_id, dataset=dataset,
            sys_token=sys_token, usr_token=usr_token, sep_token=sep_token,
            max_steps=max_steps, max_reward=max_reward, user_patience=user_patience,
            answer_parser=answer_parser, logic_parser=logic_parser, value_backend=value_backend,
            auto_skip=auto_skip)

    def draw_random_answer(self, node: DialogNode) -> Tuple[Answer, str]:
        if not node.key in self.user_answer_keys:
            # answer for this node is requested for the 1st time -> draw new goal for next turn
            self.user_answer_keys[node.key] = UserResponse(relevant=True, answer_key=node.random_answer().key)
        # answer = DialogAnswer.objects.get(version=self.version, key=self.user_answer_keys[node.key])
        answer = self.data.answers_by_key[self.user_answer_keys[node.key].answer_key]
        user_utterance = rand_remove_questionmark(random.choice(self.data.answer_synonyms[answer.text.lower()]))
        return answer, user_utterance
        
    def reset(self, current_episode: int):
        self.pre_reset()

        first_answer, self.initial_user_utterance = self.draw_random_answer(self.current_node)
        self.goal_node = first_answer.connected_node

        self.reached_goals = []
        self.asked_goals = []
        self.constraints = {}

        self.coverage_synonyms[self.initial_user_utterance.replace("?", "")] += 1

        self.post_reset()
        self.episode_log.append(f'{self.env_id}-{self.current_episode}$ MODE: GUIDED') 

    def choose_next_goal_node_guided(self) -> bool:
        done = False
        if self.current_node.node_type in [NodeType.QUESTION, NodeType.VARIABLE]:
            # draw next-turn goal
            # if self.current_node.answers.count() > 0:
            if len(self.current_node.answers) > 0:
                # make list of all follow up nodes
                #  - remove all visited nodes
                #  - draw random follow up node from remaining candidates
                #   - if none available, return None -> end dialog
                goal_candidates = set([answer.connected_node.key for answer in self.current_node.answers if answer.connected_node])
                goal_candidates = goal_candidates.difference(self.visited_node_keys.keys())
                if len(goal_candidates) == 0:
                    # no viable followup node (all already visited) -> break dialog
                    self.goal_node = None
                else:
                    # draw radnom followup node from list of viable nodes
                    self.goal_node = self.data.nodes_by_key[random.choice(list(goal_candidates))]
                    # answer, _ = self._set_answer(self.current_node, self.goal_node) # can't set user utterance here, because we're not ASKing here
            else:
                self.goal_node = None # reached end of dialog tree
        elif self.current_node.node_type == NodeType.INFO:
            # adapt goal to next node
            self.goal_node = self.current_node.connected_node
        else:
            # logic node: handle and skip to next viable node
            _, done, _ = self.handle_logic_nodes()
        self.episode_log.append(f'{self.env_id}-{self.current_episode}$ -> NEXT GOAL: ({self.goal_node.key if self.goal_node else "NONE"}) {self.goal_node.text[:100] if self.goal_node else ""} ')
        if done or isinstance(self.goal_node, type(None)):
            self.episode_log.append(f'{self.env_id}-{self.current_episode} NO NEXT GOAL - LOGIC NODE WITHOUT REQUIRED BST VARIABLE$')
            return False
        self.goal_node_coverage[self.goal_node.key] += 1
        return True
   
    def ask(self, replayed_user_utterance: Tuple[str, None]) -> Tuple[bool, float]:
        done = False
        reward = 0.0

        if self.last_action_idx == ActionType.ASK:
            if self.auto_skip_mode != AutoSkipMode.NONE:
                reward += 2 # ask is also skip
                # last ask brought us to correct goal
                if not self.choose_next_goal_node_guided():
                    done = True
            else:
                reward -= 1 # don't ask multiple times in a row!
        else:
            reward += 2 # important to ask each node
            self.asked_goals.append(1.0 * (self.reached_goals and self.reached_goals[-1] == 1)) # only asked correct goal if we jumped to the correct node in the previous transition
            if self.reached_goals and self.reached_goals[-1] == 1:
                # update goal: previous action was skip to correct node (STOP not possible)
                # -> draw new goal
                if not self.choose_next_goal_node_guided():
                    done = True

        if self.current_node.node_type == NodeType.VARIABLE:
            # get variable name and value
            answer = self.current_node.answers[0]
            var = self.answer_template_parser.find_variable(answer.text)

            # check if variable was already asked
            if var.name in self.bst:
                reward -= 4 # variable value already known
                self.episode_log.append(f'{self.env_id}-{self.current_episode}$ -> VARIABLE ALREADY KNOWN')
            else:
                # draw random variable
                self.bst[var.name] = VariableValue(var_name=var.name, var_type=var.type).draw_value() # variable is asked for the 1st time
            self.current_user_utterance = str(deepcopy(self.bst[var.name]))
            self.coverage_variables[var.name][self.bst[var.name]] += 1
            self.episode_log.append(f'{self.env_id}-{self.current_episode}$ -> VAR NAME: {var.name}, VALUE: {self.bst[var.name]}')
        elif self.current_node.node_type == NodeType.QUESTION:
            # get user reply
            if not self.goal_node:
                # there are no more options to visit new nodes from current node -> stop dialog
                done = True
            else:
                answer = self.current_node.answer_by_connected_node(self.goal_node)
                self.current_user_utterance = rand_remove_questionmark(random.choice(self.data.answer_synonyms[answer.text.lower()]))
                self.coverage_synonyms[self.current_user_utterance.replace("?", "")] += 1
        return done, reward

    def skip(self, answer_index: int) -> Tuple[bool, float]:
        done = False
        reward = 0.0

        next_node = self.get_transition(answer_index)
        if (not next_node) or self.goal_node.key != next_node.key:
            reward -= 4 # skipping is good after ask, but followup-node is wrong!
            self.reached_goals.append(0)
            self.actioncount_skip_invalid += 1
            self.episode_log.append(f'{self.env_id}-{self.current_episode}$ -> INVALID SKIP OR WRONG FOLLOWUP NODE')
        else:
            if self.last_action_idx >= ActionType.SKIP:
                if self.current_node.node_type == NodeType.VARIABLE:
                    # double skip: check if we are skipping a known variable
                    answer = self.current_node.answer_by_index(0)
                    var = self.answerParser.find_variable(answer.text)
                    # check if variable was already asked
                    if var.name in self.bst:    
                        # it is good to skip this node since variable is already known!
                        reward += 3
                        self.episode_log.append(f'{self.env_id}-{self.current_episode}$ -> SKIPPED ALREADY KNOWN VARIABLE')
                    else:
                        reward -= 2
                        self.episode_log.append(f'{self.env_id}-{self.current_episode}$ -> SKIPPED VARIABLE NODE W/O ASKING')
                else:
                    reward -= 2  # last action was skip: punish, should have asked this turn
                    self.episode_log.append(f'{self.env_id}-{self.current_episode}$ -> SKIPPED TO CORRECT NODE, BUT W/O ASKING')
                self.asked_goals.append(0)
            else: 
                reward += 3 # skipping is good after ask, and we chose next node correctly
                self.episode_log.append(f'{self.env_id}-{self.current_episode}$ -> SKIPPED TO CORRECT NODE')
            self.reached_goals.append(1)
        # progress to next node, if possible, and update goal
        if next_node:
            self.current_node = next_node
            if self.reached_goals[-1] == 0:
                if not self.choose_next_goal_node_guided():
                    done = True
                if not self.goal_node:
                    # there are no more options to visit new nodes from current node -> stop dialog
                    done = True
        return done, reward

    def reached_goal(self) -> float:
        return mean(self.reached_goals if self.reached_goals else [0.0])
    
    def asked_goal(self) -> float:
        mean(self.asked_goals if self.asked_goals else [0.0])

    def post_handle_logic_nodes(self, did_handle_logic_nodes):
        # update goal node selection in guided dialog after transitioning from logic nodes
        self.choose_next_goal_node_guided()


class _FreeEnvironment(BaseEnv):
    def __init__(self, env_id: int, dataset: GraphDataset, sys_token: str, usr_token: str, sep_token: str,
            max_steps: int, max_reward: float, user_patience: int,
            answer_parser: AnswerTemplateParser, system_parser: SystemTemplateParser, logic_parser: LogicTemplateParser,
            value_backend: RealValueBackend,
            auto_skip: AutoSkipMode) -> None:
        super().__init__(env_id=env_id, dataset=dataset,
            sys_token=sys_token, usr_token=usr_token, sep_token=sep_token, 
            max_steps=max_steps, max_reward=max_reward, user_patience=user_patience,
            answer_parser=answer_parser, logic_parser=logic_parser, value_backend=value_backend,
            auto_skip=auto_skip)
        self.goal_gen = UserGoalGenerator(graph=dataset, answer_parser=answer_parser,
            system_parser=system_parser, value_backend=value_backend,
            paraphrase_fraction=0.0, generate_fraction=0.0)

    def reset(self, current_episode: int):
        self.pre_reset()

        self.goal = None
        while not self.goal:
            try:
                self.goal = self.goal_gen.draw_goal()
            except ImpossibleGoalError:
                print("IMPOSSIBLE GOAL")
                continue
            except ValueError:
                continue
        self.goal_node = self.goal.goal_node
        self.initial_user_utterance = deepcopy(self.goal.initial_user_utterance)
        self.reached_goal_once = False
        self.asked_goal_once = False
        self.constraints = self.goal.variables 
        
        self.coverage_synonyms[self.goal.faq_key] += 1
        self.post_reset()
        self.episode_log.append(f'{self.env_id}-{self.current_episode}$ MODE: Free') 

    def ask(self, replayed_user_utterance: Tuple[str, None]) -> Tuple[bool, float]:
        reward = 0.0
        done = False 

        if not self.asked_goal_once and self.goal.has_reached_goal_node(self.current_node):
            # we ask goal node for the first time
            reward += self.max_reward
            self.asked_goal_once = 1
            self.episode_log.append(f'{self.env_id}-{self.current_episode}$ ASK REACHED GOAL')

            if self.stop_when_reaching_goal:
                # we asked goal: auto-stop
                self.episode_log.append(f'{self.env_id}-{self.current_episode}$ AUTO-STOP REACHED GOAL')
                done = True
        else:
            reward -= 1

        if not done:
            if self.auto_skip_mode != AutoSkipMode.NONE:
                reward -= 1 # because it is 2 actions

            if self.current_node.node_type == NodeType.VARIABLE:
                # get variable name and value
                var = self.answer_template_parser.find_variable(self.current_node.answer_by_index(0).text)

                # check if variable was already asked
                if var.name in self.bst:
                    reward -= 1 # variable value already known
                
                # get user reply and save to bst
                var_instance = self.goal.get_user_input(self.current_node, self.bst, self.data)
                self.bst[var.name] = var_instance.var_value
                self.current_user_utterance = str(deepcopy(var_instance.var_value))

                if not var_instance.relevant:
                    # asking for irrelevant variable is bad
                    reward -= 2
                    self.actioncount_ask_variable_irrelevant += 1
                    self.episode_log.append(f'{self.env_id}-{self.current_episode}$ -> IRRELEVANT VAR: {var.name} ')
                self.coverage_variables[var.name][self.bst[var.name]] += 1
                self.episode_log.append(f'{self.env_id}-{self.current_episode}$ -> VAR NAME: {var.name}, VALUE: {self.bst[var.name]}')
            elif self.current_node.node_type == NodeType.QUESTION:
                response = None
                if self.current_node.key in self.user_answer_keys:
                    response = self.user_answer_keys[self.current_node.key]
                else:
                    response = self.goal.get_user_response(self.current_node)
                    self.user_answer_keys[self.current_node.key] = response
                if not response:
                    # reached end of dialog tree
                    done = True
                    self.episode_log.append(f'{self.env_id}-{self.current_episode}$ -> REACHED TREE END')
                else:
                    # get user reply
                    if not response.relevant:
                        reward -= 2 # chose different path than goal path]
                        self.actioncount_ask_question_irrelevant += 1
                        self.episode_log.append(f'{self.env_id}-{self.current_episode}$ -> IRRELEVANT QUESTION')
                    # answer = self.current_node.answers.get(key=response.answer_key)
                    if replayed_user_utterance:
                        self.current_user_utterance = replayed_user_utterance
                    else:
                        answer = self.current_node.answer_by_key(response.answer_key)
                        self.current_user_utterance = rand_remove_questionmark(random.choice(self.data.answer_synonyms[answer.text.lower()]))
                    self.coverage_synonyms[self.current_user_utterance.replace("?", "")] += 1
            # info nodes don't require special handling

        return done, reward

    def skip(self, answer_index: int) -> Tuple[bool, float]:
        reward = -1.0
        done = False 

        next_node = self.get_transition(answer_index)

        if next_node:
            # valid transition
            self.current_node = next_node
            if self.goal.has_reached_goal_node(self.current_node):
                reward += 5 # assign a reward for reaching the goal (but not asked yet, because this was a skip)
                self.reached_goal_once = True
                self.episode_log.append(f'{self.env_id}-{self.current_episode}$ -> REACHED GOAL')
        else:
            # invalid transition -> punish
            reward -= 3
            self.actioncount_skip_invalid += 1
        return done, reward

    def reached_goal(self) -> bool:
        return self.reached_goal_once

    def asked_goal(self) -> Union[bool, float]:
        return self.asked_goal_once



class RealUserEnvironment(BaseEnv):
    def __init__(self, env_id: int, dataset: GraphDataset, sys_token: str, usr_token: str, sep_token: str,
            max_steps: int, max_reward: float,
            answer_parser: AnswerTemplateParser, logic_parser: LogicTemplateParser,
            value_backend: RealValueBackend,
            auto_skip: AutoSkipMode) -> None:
        super().__init__(env_id=env_id, dataset=dataset,
            sys_token=sys_token, usr_token=usr_token, sep_token=sep_token, 
            max_steps=max_steps, max_reward=max_reward,
            answer_parser=answer_parser, logic_parser=logic_parser, value_backend=value_backend,
            auto_skip=auto_skip)

    def reset(self):
        self.pre_reset()

        # Mock a goal node that we can never reach to keep the conversation alive
        self.goal_node = DialogNode(key="syntheticGoalNode", text="Synthetic Goal Node", node_type=NodeType.INFO, answers=[], questions=[], connected_node=None)

        # Output first node
        print(self.current_node.text)
        # Ask for initial user input
        self.initial_user_utterance = deepcopy(input(">>"))

        self.reached_goal_once = False
        self.asked_goal_once = False
        self.constraints = {}
        
        self.post_reset()

    def check_user_patience_reached(self) -> bool:
        return False # should never quit dialog automatically

    def ask(self, replayed_user_utterance: Tuple[str, None]) -> Tuple[bool, float]:
        reward = 0.0
        # output system text
        print("ASKING", self.current_node.text)

        if self.auto_skip_mode != AutoSkipMode.NONE:
            reward -= 1 # because it is 2 actions

        if self.current_node.node_type == NodeType.VARIABLE:
            # get variable name
            var = self.answer_template_parser.find_variable(self.current_node.answer_by_index(0).text)

            # check if variable was already asked
            if var.name in self.bst:
                reward -= 1 # variable value already known
            
            # get user reply and save to bst
            var_value = input(">>")
            self.bst[var.name] = var_value
            self.current_user_utterance = str(deepcopy(var_value))

            self.coverage_variables[var.name][self.bst[var.name]] += 1
        elif self.current_node.node_type == NodeType.QUESTION:
            response = input(">>")
            self.current_user_utterance = deepcopy(response)

        return False, reward

    def skip(self, answer_index: int) -> Tuple[bool, float]:
        done = False
        reward = -1.0
        
        print("SKIPPING", self.current_node.text[:100])
        next_node = self.get_transition(answer_index)

        if next_node:
            # valid transition
            self.current_node = next_node
        else:
            done = True
            print("REACHED END OF DIALOG TREE")
        return done, reward

    def reached_goal(self) -> Union[bool, float]:
        return False

    def asked_goal(self) -> Union[bool, float]:
        return False



class CTSEnvironment(gym.Env):
    def __init__(self, mode: str,
                dataset: GraphDataset,
                guided_free_ratio: float,
                auto_skip: AutoSkipMode,
                normalize_rewards: bool,
                max_steps: int,
                user_patience: int,
                stop_when_reaching_goal: bool,
                num_train_envs: int,
                num_val_envs: int,
                num_test_envs: int,
                sys_token: str, usr_token: str, sep_token: str):
        print("ENV!!", mode, "TOKENS:", sys_token, usr_token, sep_token)

        answer_parser = AnswerTemplateParser()
        logic_parser = LogicTemplateParser()
        system_parser = SystemTemplateParser()
        value_backend = RealValueBackend(dataset.a1_countries)

        self.guided_free_ratio = guided_free_ratio
        self.max_reward = 4 * dataset.get_max_tree_depth() if normalize_rewards else 1.0
        
        
        if guided_free_ratio > 0.0:
            self.guided_env = _GuidedEnvironment(dataset=dataset, 
                sys_token=sys_token, usr_token=usr_token, sep_token=sep_token,
                max_steps=max_steps, max_reward=self.max_reward, user_patience=user_patience,
                answer_parser=answer_parser, logic_parser=logic_parser,
                value_backend=value_backend,
                auto_skip=auto_skip)
        if guided_free_ratio < 1.0:
            self.free_env = _FreeEnvironment(dataset=dataset, 
                sys_token=sys_token, usr_token=usr_token, sep_token=sep_token,
                max_steps=max_steps, max_reward=self.max_reward, user_patience=user_patience,
                answer_parser=answer_parser, system_parser=system_parser, logic_parser=logic_parser, 
                value_backend=value_backend,
                auto_skip=auto_skip)

        # TODO add logger
        # TODO forward coverage stats
    
    @property
    def current_episode(self):
        return self.guided_env.current_episode + self.free_env.current_episode

    def reset(self):
        # choose uniformely at random between guided and free env according to ratio
        self.active_env = self.guided_env if random.random() < self.guided_free_ratio else self.free_env
        return self.active_env.reset(current_episode=self.current_episode)

    def step(self, action: int, replayed_user_utterance: Tuple[str, None] = None) -> Tuple[dict, float, bool, dict]:
        obs, reward, done, info = self.active_env.step(action, replayed_user_utterance)
        info[EnvInfo.IS_FAQ] = self.active_env == self.free_env
        return obs, reward, done, info

    def get_goal_node_coverage_free(self):
        return len(self.free_env.goal_node_coverage) / self.data.count_question_nodes()

    def get_goal_node_coverage_guided(self):
        return len(self.guided_env.goal_node_coverage) / self.data.num_guided_goal_nodes

