from collections import defaultdict
from functools import reduce
import itertools
from statistics import mean
from typing import Any, Callable, DefaultDict, Dict, List, Optional, Sequence, Type, Union, Tuple
import logging
import random
from copy import deepcopy

import gymnasium as gym
from config import ActionType
from data.dataset import Answer, DialogNode, GraphDataset, NodeType
from data.parsers.answerTemplateParser import AnswerTemplateParser
from data.parsers.logicParser import LogicTemplateParser
from gymnasium import spaces
import torch
from data.parsers.parserValueProvider import RealValueBackend
from data.parsers.systemTemplateParser import SystemTemplateParser
from environment.goal import DummyGoal, UserGoal, UserGoalGenerator, UserResponse, VariableValue
from utils.envutils import GoalDistanceMode

from utils.utils import AutoSkipMode, EnvInfo, rand_remove_questionmark


import itertools
import warnings

import gymnasium as gym
import numpy as np

from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvIndices, VecEnvObs, VecEnvStepReturn
from stable_baselines3.common.vec_env.patch_gym import _patch_env
from stable_baselines3.common.vec_env.util import obs_space_info

from encoding.state import StateEncoding



class OldCTSEnv(gym.Env):
    def __init__(self, 
                # env_id: int,
                mode: str,
                dataset: GraphDataset,
                guided_free_ratio: float,
                auto_skip: AutoSkipMode,
                normalize_rewards: bool,
                max_steps: int,
                user_patience: int,
                stop_when_reaching_goal: bool,
                stop_on_invalid_skip: bool,
                sys_token: str, usr_token: str, sep_token: str,
                goal_distance_mode: GoalDistanceMode,
                goal_distance_increment: int,
                **kwargs):
        
        assert isinstance(auto_skip, AutoSkipMode)
        assert isinstance(goal_distance_mode, GoalDistanceMode) 
        self.mode = mode
     
        self.max_steps = max_steps if max_steps else 2 * dataset.get_max_tree_depth() # if no max step is set, choose automatically 2*tree depth
        self.user_patience = user_patience
        self.stop_when_reaching_goal = stop_when_reaching_goal
        self.normalize_rewards = normalize_rewards
        self.auto_skip = auto_skip
        self.guided_free_ratio = guided_free_ratio
        self.dialog_faq_ratio = guided_free_ratio

        # load dialog tree info
        self.data = dataset
        # setup user goal generator
        self.systemParser = SystemTemplateParser()
        self.logicParser = LogicTemplateParser()
        self.answer_template_parser = AnswerTemplateParser()
        self.value_backend = RealValueBackend(a1_laender=dataset.a1_countries, data=dataset)
        self.goal_gen = UserGoalGenerator(graph=dataset, answer_parser=self.answer_template_parser, system_parser=self.systemParser,
                                            value_backend=self.value_backend)
        
        # reward normalization
        self.max_reward = 4*dataset.get_max_tree_depth() if normalize_rewards else 1.0

        self.episode_log = []
        self.current_episode = 0
        self.max_distance = 100

        # stats
        self.reached_goals_guided = []
        self.reached_goals_free = []
        self.asked_goals_guided = []
        self.asked_goals_free = []
        self.turn_counter = 0

        # coverage stats
        self.goal_node_coverage_free = defaultdict(int)
        self.goal_node_coverage_guided = defaultdict(int)
        self.node_coverage = defaultdict(int)
        self.coverage_faqs = defaultdict(int)
        self.coverage_synonyms = defaultdict(int)
        self.coverage_variables = defaultdict(lambda: defaultdict(int))

        self.is_faq_mode = False


    def _transform_dialog_history(self):
        # interleave system and user utterances
        # Returns: List[Tuple(sys: str, usr: str)]
        usr_history = self.user_utterances_history
        if len(usr_history) < len(self.system_utterances_history):
            usr_history = usr_history + [""]
        assert len(usr_history) == len(self.system_utterances_history)
        return list(itertools.chain(zip([utterance for utterance in self.system_utterances_history], [utterance for utterance in self.user_utterances_history])))
        
    def reached_goal(self):
        if self.is_faq_mode:
            return self.reached_goal_once
        else:
            if len(self.reached_goals) == 0:
                return 0.0
            else:
                return mean(self.reached_goals)
    

    def _get_obs(self) -> Dict[EnvInfo, Any]:
        return {
                EnvInfo.DIALOG_NODE_KEY: self.current_node.key,
                EnvInfo.LAST_SYSTEM_ACT: self.last_action_idx,
                EnvInfo.BELIEFSTATE: deepcopy(self.bst),
                EnvInfo.EPISODE_REWARD: self.episode_reward,
                EnvInfo.EPISODE_LENGTH: self.current_step,
                EnvInfo.EPISODE: self.current_episode,
                EnvInfo.REACHED_GOAL_ONCE: self.reached_goal(),
                EnvInfo.PERCIEVED_LENGTH: self.percieved_length,
                EnvInfo.ASKED_GOAL: self.asked_goal,
                EnvInfo.INITIAL_USER_UTTERANCE: deepcopy(self.initial_user_utterance),
                EnvInfo.CURRENT_USER_UTTERANCE: deepcopy(self.current_user_utterance),
                EnvInfo.USER_UTTERANCE_HISTORY: deepcopy(self.user_utterances_history),
                EnvInfo.SYSTEM_UTTERANCE_HISTORY: deepcopy(self.system_utterances_history),
                EnvInfo.LAST_VALID_SKIP_TRANSITION_IDX: self.current_step,
                EnvInfo.GOAL: DummyGoal(goal_node_key=self.goal.goal_node_key,
                                        initial_user_utterance=deepcopy(self.goal.initial_user_utterance),
                                        delexicalised_initial_user_utterance=deepcopy(self.goal.delexicalised_initial_user_utterance),
                                        constraints=deepcopy(self.goal.constraints),
                                        answer_pks=deepcopy(self.user_answer_keys),
                                        visited_ids=deepcopy(self.goal.visited_ids)
                                    ),
                EnvInfo.IS_FAQ: self.is_faq_mode
        }
    

    def seed(self, seed: int):
        pass

    def _increment_episode_count(self):
        self.current_episode += 1
        
    def _her_faq_reset(self, goal: DummyGoal) -> Dict[EnvInfo, Any]:
        self.user_answer_keys = goal.answer_pks
        self.visited_node_keys = defaultdict(int)
        self.episode_log = []
        self.bst = {}
        self._increment_episode_count()
        
        self.current_node = self.data.start_node.connected_node
        self.last_action_idx = 1 # start by asking start node
       
        self.is_faq_mode = True
        self.goal = goal
        self.goal_node = self.data.nodes_by_key[self.goal.goal_node_key]
        self.initial_user_utterance = deepcopy(self.goal.initial_user_utterance)
        self.reached_goal_once = False
        self.asked_goal = False
        self.constraints = self.goal.constraints 
        
        self.current_user_utterance = deepcopy(self.initial_user_utterance)
        self.user_utterances_history = [deepcopy(self.initial_user_utterance)]
        self.system_utterances_history = [deepcopy(self.current_node.text)]
        self.visited_node_keys[self.current_node.key] = 1

        # dialog stats
        self.current_step = 0
        self.episode_reward = 0.0
        self.skipped_nodes = 0

        # node stats
        self.nodecount_info = 0
        self.nodecount_variable = 0
        self.nodecount_question = 0
        self.nodecount_logic = 0

        # action stats
        self.actioncount_stop = 0
        self.actioncount_ask = 0
        self.actioncount_skip = 0
        self.actioncount_stop_prematurely = 0
        self.actioncount_ask_variable = 0
        self.actioncount_ask_variable_irrelevant = 0
        self.actioncount_ask_question = 0
        self.actioncount_ask_question_irrelevant = 0
        self.actioncount_skip_info = 0
        self.actioncount_skip_question = 0
        self.actioncount_skip_variable = 0
        self.actioncount_skip_invalid = 0
        self.actioncount_missingvariable = 0

        self.coverage_faqs[goal.goal_node_key] += 1
        self.goal_node_coverage_free[self.goal_node.key] += 1

        return self._get_obs()

    def reset(self) -> Dict[EnvInfo, Any]:
        if self.current_episode > 0:
            if self.is_faq_mode:
                self.reached_goals_free.append(self.reached_goal_once)
                self.asked_goals_guided.append(self.asked_goal)
            else:
                self.reached_goals_guided.append(mean(self.reached_goals) if len(self.reached_goals) > 0 else 0.0)
                self.asked_goals_guided.append(mean(self.asked_goals) if len(self.asked_goals) > 0 else 0.0)


        self.user_answer_keys = defaultdict(int)
        self.visited_node_keys = defaultdict(int)
        self.episode_log = []
        self.bst = {}
        self._increment_episode_count()
        self.reached_goal_once = False
        self.asked_goal = False
        self.reached_goals = []
        self.asked_goals = []
        
        self.current_node = self.data.start_node.connected_node
        self.last_action_idx = 1 # start by asking start node
        self.goal = None

        if random.random() < self.dialog_faq_ratio:
            # draw guided goal (turn by turn)
            self.is_faq_mode = False
            first_answer, self.initial_user_utterance = self._draw_random_answer(self.current_node)
            self.goal_node = first_answer.connected_node
            self.constraints = {}
            self.goal_node_coverage_guided[self.goal_node.key] += 1
            self.coverage_synonyms[self.initial_user_utterance.replace("?", "")] += 1
            self.goal = DummyGoal(goal_node_key=self.goal_node.key, initial_user_utterance=self.initial_user_utterance, 
                                        delexicalised_initial_user_utterance=self.initial_user_utterance, constraints={},
                                        answer_pks={}, visited_ids=set())
        else:
            # draw free goal
            self.is_faq_mode = True
            self.goal: UserGoal = self.goal_gen.draw_goal_free(max_distance=100)
            self.goal_node = self.data.nodes_by_key[self.goal.goal_node_key]
            self.initial_user_utterance = deepcopy(self.goal.initial_user_utterance)
            self.constraints = self.goal.variables 
            
            self.goal_node_coverage_free[self.goal_node.key] += 1
            self.coverage_faqs[self.goal.goal_node_key] += 1
        
        self.current_user_utterance = deepcopy(self.initial_user_utterance)
        self.user_utterances_history = [deepcopy(self.initial_user_utterance)]
        self.system_utterances_history = [deepcopy(self.current_node.text)]
        self.visited_node_keys[self.current_node.key] = 1

        # coverage stats
        self.node_coverage[self.current_node.key] += 1

        # dialog stats
        self.current_step = 0
        self.episode_reward = 0.0
        self.skipped_nodes = 0
        self.percieved_length = 0

        # node stats
        self.nodecount_info = 0
        self.nodecount_variable = 0
        self.nodecount_question = 0
        self.nodecount_logic = 0

        # action stats
        self.actioncount_stop = 0
        self.actioncount_ask = 0
        self.actioncount_skip = 0
        self.actioncount_stop_prematurely = 0
        self.actioncount_ask_variable = 0
        self.actioncount_ask_variable_irrelevant = 0
        self.actioncount_ask_question = 0
        self.actioncount_ask_question_irrelevant = 0
        self.actioncount_skip_info = 0
        self.actioncount_skip_question = 0
        self.actioncount_skip_variable = 0
        self.actioncount_skip_invalid = 0
        self.actioncount_missingvariable = 0
        self.env_id = 0

        # Logging
        self.episode_log.append(f'{self.env_id}-{self.current_episode}$ ======== RESET =========')
        self.episode_log.append(f'{self.env_id}-{self.current_episode}$ GOAL: ({"FREE" if self.is_faq_mode else "GUIDED"}): {self.goal_node.key} {self.goal_node.text[:75]}') 
        self.episode_log.append(f'{self.env_id}-{self.current_episode}$ CONSTRAINTS: {self.constraints}')
        self.episode_log.append(f'{self.env_id}-{self.current_episode}$ INITIAL UTTERANCE: {self.initial_user_utterance}') 
        return self._get_obs()
    
    def reset_episode_log(self):
        self.episode_log = []

    def reset_stats(self):
        self.reached_goals = []
        self.asked_goals = []
        self.percieved_length = 0

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


    def _draw_random_answer(self, node: DialogNode) -> Tuple[Answer, str]:
        if not node.key in self.user_answer_keys:
            # answer for this node is requested for the 1st time -> draw new goal for next turn
            # answer_keys = node.answers.values_list("key", flat=True)
            # self.user_answer_keys[node.key] = random.choice(answer_keys)
            self.user_answer_keys[node.key] = UserResponse(relevant=True, answer_key=node.random_answer().key)
        # answer = DialogAnswer.objects.get(version=self.version, key=self.user_answer_keys[node.key])
        answer = self.data.answers_by_key[self.user_answer_keys[node.key].answer_key]
        user_utterance = rand_remove_questionmark(random.choice(self.data.answer_synonyms[answer.text.lower()]))
        return answer, user_utterance

    def _check_user_patience_reached(self) -> bool:
        return self.visited_node_keys[self.current_node.key] > self.user_patience # +1 because  skip action to node already counts as 1 visit

    def _reached_max_length(self) -> bool:
        return self.current_step == self.max_steps - 1 # -1 because first turn is counted as step 0

    def _update_action_counters(self, action: int):
        if action >= 2 and self.last_action_idx >= 2: # it's only a REAL skip if we didn't ask before
            # update counters
            if self.current_node.node_type == NodeType.INFO:
                self.actioncount_skip_info += 1
            elif self.current_node.node_type == NodeType.VARIABLE:
                self.actioncount_skip_variable += 1
            elif self.current_node.node_type == NodeType.QUESTION:
                self.actioncount_skip_question += 1
            self.skipped_nodes += 1

    def _update_node_counters(self):
        if self.current_node:
            if self.current_node.node_type == NodeType.INFO:
                self.nodecount_info += 1
            elif self.current_node.node_type == NodeType.VARIABLE:
                self.nodecount_variable += 1
            elif self.current_node.node_type == NodeType.QUESTION:
                self.nodecount_question += 1

    def _handle_logic_nodes(self) -> Tuple[float, bool]:
        reward = 0
        done = False

        did_handle_logic_node = False
        while self.current_node and self.current_node.node_type == NodeType.LOGIC:
            did_handle_logic_node = True
            self.nodecount_logic += 1
            lhs = self.current_node.text
            var_name = lhs.lstrip("{{").strip() # form: {{ VAR_NAME
            if var_name in self.bst:
                # don't assign reward, is automatic transition without agent interaction
                # evaluate statement, choose correct branch and skip to connected node
                default_answer = None
                # for idx, answer in enumerate(self.current_node.answers.all()):
                for idx, answer in enumerate(self.current_node.answers):
                    # check if full statement {{{lhs rhs}}} evaluates to True
                    rhs = answer.text
                    if not "DEFAULT" in rhs: # handle DEFAULT case last!
                        if self._fillLogicTemplate(f"{lhs} {rhs}"):
                            # evaluates to True, follow this path!
                            default_answer = answer
                            break
                    else:
                        default_answer = answer
                self.prev_node = self.current_node
                self.current_node = default_answer.connected_node
                if self.goal_node.key == self.current_node.key:
                    if self.is_faq_mode:
                        self.reached_goal_once = True
                    else:
                        self.reached_goals.append(1.0)
                    self.episode_log.append(f'{self.env_id}-{self.current_episode}$ -> REACHED GOAL (via logic node skip)')
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

        if did_handle_logic_node and (not self.is_faq_mode) and not done:
            # update goal node selection in guided dialog after transitioning from logic nodes
            self._choose_next_goal_node_guided()
        return reward, done

    def _choose_next_goal_node_guided(self) -> bool:
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
            _, done = self._handle_logic_nodes() 
        self.episode_log.append(f'{self.env_id}-{self.current_episode}$ -> NEXT GOAL: ({self.goal_node.key if self.goal_node else "NONE"}) {self.goal_node.text[:50] if self.goal_node else ""} ')
        if done or isinstance(self.goal_node, type(None)):
            self.episode_log.append(f'{self.env_id}-{self.current_episode} NO NEXT GOAL - LOGIC NODE WITHOUT REQUIRED BST VARIABLE$')
            return False
        self.goal_node_coverage_guided[self.goal_node.key] += 1
        return True

    def step(self, action: int, _replayed_user_utterance: Union[str, None] = "") -> Tuple[Dict[EnvInfo, Any], float, bool, Dict[EnvInfo, Any]]:
        self.env_id = 0
        self.episode_log.append(f'{self.env_id}-{self.current_episode}$ - --- Turn {self.current_step}')
        if self.current_node:
            self.episode_log.append(f'{self.env_id}-{self.current_episode}$ FROM NODE: {self.current_node.node_type} - {self.current_node.key} - {self.current_node.text[:75]}')
        prev_node_key = self.current_node.key
        done = False
        reward = 0.0
        real_action = action + 1 # 1st action is ASK instead of STOP because we have no STOP action - adding 1 here doesn't change replay buffer correctness, but allows code for step() to remain unchanged
        self.episode_log.append(f'{self.env_id}-{self.current_episode}$ -> ACTION: {real_action}')
        self.current_step += 1
        self.current_user_utterance = "" # reset each turn, will be assigned in _step_X method if action is ASK
        self._update_action_counters(real_action)
        self.actioncount_stop += real_action == 0
        self.actioncount_ask += real_action == 1
        self.actioncount_skip += real_action >= 2
        self._update_node_counters()


        # check if dialog should end
        if self._check_user_patience_reached(): 
            reward -= self.max_reward  # bad
            done = True
            self.episode_log.append(f'{self.env_id}-{self.current_episode}$ REACHED MAX USER PATIENCE')
        elif self._reached_max_length():
            done = True # already should have large negtative reward (expect if in guided mode, where max length could even be desired)
            self.episode_log.append(f'{self.env_id}-{self.current_episode}$ REACHED MAX LENGTH')
        else:
            # step
            if action == 0: # ASK
                self.percieved_length += 1

            reward, done = self._step_free(real_action, _replayed_user_utterance) if self.is_faq_mode else self._step_guided(real_action)
            
            self.episode_log.append(f'{self.env_id}-{self.current_episode}$ -> USER UTTERANCE: {self.current_user_utterance}')
            self.episode_log.append(f'{self.env_id}-{self.current_episode}$ TO NODE: {self.current_node.node_type} - {self.current_node.key} - {self.current_node.text[:75]}')

            # update history
            self.last_action_idx = real_action
            self.user_utterances_history.append(str(deepcopy(self.current_user_utterance)))
            self.system_utterances_history.append(deepcopy(self.current_node.text))

            # update counters
            self.visited_node_keys[self.current_node.key] += 1

            # auto-skipping
            if (not done) and self.goal_node and self.auto_skip != AutoSkipMode.NONE and self.last_action_idx == 1:
                # auto-skipping after asking
                if self.current_node.node_type in [NodeType.QUESTION, NodeType.INFO, NodeType.VARIABLE]:
                    if len(self.current_node.answers) > 0:
                        # check that current node has answers!
                        # if so, choose best fitting one (w.r.t. user text)
                        # similarities = self.similarity_model.encode(current_user_utterance=self.current_user_utterance, dialog_node=self.current_node) # 1 x answers
                        # skip_action_idx = similarities.view(-1).argmax()
                        # skip_action = self.current_node.answer_by_index(skip_action_idx)
                        
                        if self.is_faq_mode:
                            if self.current_node.node_type == NodeType.QUESTION:
                                if self.auto_skip == AutoSkipMode.ORACLE:
                                    # semantic level: choose correct answer to jump to (assume perfect NLU)
                                    response: UserResponse = self.user_answer_keys[self.current_node.key]
                                    skip_action = self.data.answers_by_key[response.answer_key]
                                else:
                                    # utterance level: choose answer to jump to by similarity
                                    similarities = self.similarity_model.encode(current_user_utterance=self.current_user_utterance, dialog_node=self.current_node, noise=self.noise) # 1 x actions x 1
                                    skip_action_idx = similarities.view(-1).argmax(-1).item()
                                    skip_action = self.current_node.answer_by_index(skip_action_idx)
                            else:
                                assert len(self.current_node.answers) == 1
                                skip_action = self.current_node.answers[0]
                        else:
                            if self.auto_skip == AutoSkipMode.ORACLE:
                                # semantic level: choose correct answer to jump to (assume perfect NLU)
                                skip_action = self.current_node.answer_by_goalnode_key(self.goal_node.key)
                            else:
                                # utterance level: choose answer to jump to by similarity
                                if self.current_node.node_type == NodeType.QUESTION:
                                    similarities = self.similarity_model.encode(current_user_utterance=self.current_user_utterance, dialog_node=self.current_node, noise=self.noise) # 1 x actions x 1
                                    skip_action_idx = similarities.view(-1).argmax(-1).item()
                                    skip_action = self.current_node.answer_by_index(skip_action_idx)
                                else:
                                    assert len(self.current_node.answers) == 1
                                    skip_action = self.current_node.answers[0]


                        # jump to connected node
                        self.current_node = skip_action.connected_node
                        self.user_utterances_history.append("")
                        self.system_utterances_history.append(deepcopy(self.current_node.text))
                    elif self.current_node.connected_node_key:
                        # no answers, but connected node -> skip to that node
                        self.current_node = self.current_node.connected_node
                        self.user_utterances_history.append("")
                        self.system_utterances_history.append(deepcopy(self.current_node.text))
                    if self.current_node.key == self.goal_node.key:
                        # check if skipped-to node is goal node
                        if self.is_faq_mode:
                            self.reached_goal_once = True
                        else:
                            self.reached_goals.append(1.0)

                    # update history
                    self.episode_log.append(f'{self.env_id}-{self.current_episode}$ AUTOSKIP TO NODE: {self.current_node.node_type} - {self.current_node.key} - {self.current_node.text[:75]}')
                    self.last_action_idx = real_action
                    # update counters
                    self._update_node_counters()
                    self.visited_node_keys[self.current_node.key] += 1

            # handle logic node auto-transitioning here
            if not done:
                logic_reward, logic_done = self._handle_logic_nodes()
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
           
        obs = self._get_obs()
        self.turn_counter += 1
        return obs, reward/self.max_reward, done, False, obs

    def _get_node_answers(self, node: DialogNode) -> List[Answer]:
        return node.answers.order_by('answer_index')

    def _get_transition(self, action: int) -> Tuple[DialogNode, None]:
        # num_answers = self.current_node.answers.count()
        num_answers = len(self.current_node.answers)
        answer_idx = action - 2 # subtract STOP and ASK actions
        if num_answers == 0 and self.current_node.connected_node:
            if answer_idx == 0:
                return self.current_node.connected_node
            else:
                self.episode_log.append(f'{self.env_id}-{self.current_episode}$ -> INVALID TRANSITION TO CONNECTED_NODE {answer_idx}')
                return None
        if answer_idx >= num_answers:
            self.episode_log.append(f'{self.env_id}-{self.current_episode}$ -> INVALID TRANSITION TO ANSWER {answer_idx}, HAS ONLY {num_answers}')
            return None # invalid action: stay at current node
        # return self.current_node.answers.get(answer_index=answer_idx).connected_node
        
        next_node = self.current_node.answer_by_index(answer_idx).connected_node
        self.node_coverage[next_node.key] += 1
        return next_node

    def _step_guided(self, action: int) -> Tuple[float, bool]:     
        reward = 0.0 
        done = False

        if action == 0:
            # STOP
            reward -= self.max_steps
            done = True
            self.actioncount_stop_prematurely += 1
            self.episode_log.append(f'{self.env_id}-{self.current_episode}$ -> STOP ACTION')
        elif action == 1:
            assert self.current_node.node_type != NodeType.LOGIC

            # ASK
            if self.last_action_idx == 1:
                if self.auto_skip != AutoSkipMode.NONE:
                    reward += 2 # ask is also skip
                else:
                    reward -= 1 # don't ask multiple times in a row!
                if self.auto_skip != AutoSkipMode.NONE:
                    # last ask brought us to correct goal
                    if not self._choose_next_goal_node_guided():
                        done = True
                        self.episode_log.append(f'{self.env_id}-{self.current_episode}$ -> FAILED CHOOSING NEW GUIDED GOAL NODE')
            else:
                reward += 2 # important to ask each node
                self.asked_goals.append(1.0 * (self.reached_goals and self.reached_goals[-1] == 1)) # only asked correct goal if we jumped to the correct node in the previous transition
                if self.reached_goals and self.reached_goals[-1] == 1:
                    # update goal: previous action was skip to correct node (STOP not possible)
                    # -> draw new goal
                    if not self._choose_next_goal_node_guided():
                        done = True
                        self.episode_log.append(f'{self.env_id}-{self.current_episode}$ -> FAILED CHOOSING NEW GUIDED GOAL NODE')

            if self.current_node.node_type == NodeType.VARIABLE:
                # get variable name and value
                # answer = self.current_node.answers.first()
                answer = self.current_node.answers[0]
                var = self.answer_template_parser.find_variable(answer.text)

                # check if variable was already asked
                if var.name in self.bst:
                    reward -= 4 # variable value already known
                    self.episode_log.append(f'{self.env_id}-{self.current_episode}$ -> VARIABLE ALREADY KNOWN')
                else:
                    # draw random variable
                    self.bst[var.name] = VariableValue(var_name=var.name, var_type=var.type).draw_value(self.data) # variable is asked for the 1st time
                self.current_user_utterance = str(deepcopy(self.bst[var.name]))
                self.actioncount_ask_variable += 1
                self.coverage_variables[var.name][self.bst[var.name]] += 1
                self.episode_log.append(f'{self.env_id}-{self.current_episode}$ -> VAR NAME: {var.name}, VALUE: {self.bst[var.name]}')
            elif self.current_node.node_type == NodeType.QUESTION:
                # get user reply
                if not self.goal_node:
                    # there are no more options to visit new nodes from current node -> stop dialog
                    done = True
                    self.episode_log.append(f'{self.env_id}-{self.current_episode}$ -> GOAL NODE = NONE')
                else:
                    answer = self.current_node.answer_by_connected_node(self.goal_node)
                    self.current_user_utterance = rand_remove_questionmark(random.choice(self.data.answer_synonyms[answer.text.lower()]))
                    self.coverage_synonyms[self.current_user_utterance.replace("?", "")] += 1
        else:
            # SKIP
            next_node = self._get_transition(action)
            if (not next_node) or self.goal_node.key != next_node.key:
                reward -= 4 # skipping is good after ask, but followup-node is wrong!
                self.reached_goals.append(0)
                self.actioncount_skip_invalid += 1
                self.episode_log.append(f'{self.env_id}-{self.current_episode}$ -> INVALID SKIP OR WRONG FOLLOWUP NODE')
            else:
                if self.last_action_idx >= 2:
                    if self.current_node.node_type == NodeType.VARIABLE:
                        # double skip: check if we are skipping a known variable
                        answer = self.current_node.answers[0]
                        var = self.answer_template_parser.find_variable(answer.text)
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
                    if not self._choose_next_goal_node_guided():
                        done = True
                        self.episode_log.append(f'{self.env_id}-{self.current_episode}$ -> FAILED CHOOSING NEW GUIDED GOAL NODE')
                    if not self.goal_node:
                        # there are no more options to visit new nodes from current node -> stop dialog
                        done = True
                        self.episode_log.append(f'{self.env_id}-{self.current_episode}$ -> FAILED CHOOSING NEW GUIDED GOAL NODE')
        return reward, done

    def _step_free(self, action: int, _replayed_user_utterance: Union[str, None]) -> Tuple[float, bool]:
        reward = 0.0 
        done = False

        if action == 0:
            # STOP
            if self.asked_goal or self.goal.has_reached_goal_node(self.current_node):
                # we are stopping on or after goal node (after having asked)
                reward += self.max_reward * 0.5
                self.episode_log.append(f'{self.env_id}-{self.current_episode}$ STOP REACHED GOAL')
            else:
                reward -= self.max_reward # we are stopping without having asked goal node and stopping on non-goal node
                self.actioncount_stop_prematurely += 1
            done = True
            self.episode_log.append(f'{self.env_id}-{self.current_episode}$ -> STOP ACTION')
        elif action == 1:
            # ASK
            if not self.asked_goal and self.goal.has_reached_goal_node(self.current_node):
                # we ask goal node for the first time
                reward += self.max_reward
                self.asked_goal = 1
                self.episode_log.append(f'{self.env_id}-{self.current_episode}$ ASK REACHED GOAL')

                if self.stop_when_reaching_goal:
                    # we asked goal: auto-stop
                    self.episode_log.append(f'{self.env_id}-{self.current_episode}$ AUTO-STOP REACHED GOAL')
                    done = True
            else:
                reward -= 1
            
            if not done:
                if self.auto_skip != AutoSkipMode.NONE:
                    reward -= 1 # because it is 2 actions

                if self.current_node.node_type == NodeType.VARIABLE:
                    # get variable name and value
                    # var = self.answer_template_parser.find_variable(self.current_node.answers.first().text)
                    var = self.answer_template_parser.find_variable(self.current_node.answers[0].text)

                    # check if variable was already asked
                    if var.name in self.bst:
                        reward -= 1 # variable value already known
                    
                    # get user reply and save to bst
                    var_instance = self.goal.get_user_input(self.current_node, self.bst, self.data, self.answer_template_parser)
                    self.bst[var.name] = var_instance.var_value
                    self.current_user_utterance = str(deepcopy(var_instance.var_value))

                    if not var_instance.relevant:
                        # asking for irrelevant variable is bad
                        reward -= 2
                        self.actioncount_ask_variable_irrelevant += 1
                        self.episode_log.append(f'{self.env_id}-{self.current_episode}$ -> IRRELEVANT VAR: {var.name} ')
                    self.actioncount_ask_variable += 1
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
                        if _replayed_user_utterance:
                            self.current_user_utterance = _replayed_user_utterance
                        else:
                            answer = self.current_node.answer_by_key(response.answer_key)
                            self.current_user_utterance = rand_remove_questionmark(random.choice(self.data.answer_synonyms[answer.text.lower()]))
                        self.coverage_synonyms[self.current_user_utterance.replace("?", "")] += 1
                    self.actioncount_ask_question += 1
                # info nodes don't require special handling
        else:
            # SKIP
            reward -= 1
            next_node = self._get_transition(action)

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
        return reward, done
        
    def _fillLogicTemplate(self, delexicalized_utterance: str):
        return self.logicParser.parse_template(delexicalized_utterance, self.value_backend, self.bst)

    def get_history_word_count(self):
        words = 0
        for hist_item in self.system_utterances_history + self.user_utterances_history:
            words += len(hist_item.split())
        return words

    def render(self, mode='human', close=False):
        # text output
        print(f"Step: {self.current_step}")
        print(f"Episode reward: {self.episode_reward}")
        print(f"Current node: {self.current_node.key}: {self.current_node.text}")
        print(f"Goal: {self.goal_node.goal_node.key}: {self.goal_node.goal_node.text}")



class OldCustomVecEnv(VecEnv):
    """
    Creates a simple vectorized wrapper for multiple environments, calling each environment in sequence on the current
    Python process. This is useful for computationally simple environment such as ``Cartpole-v1``,
    as the overhead of multiprocess or multithread outweighs the environment computation time.
    This can also be used for RL methods that
    require a vectorized environment, but that you want a single environments to train with.

    :param env_fns: a list of functions
        that return environments to vectorize
    :raises ValueError: If the same environment instance is passed as the output of two or more different env_fn.
    """

    actions: np.ndarray

    def __init__(self, env_fns: List[Callable[[], gym.Env]],
                 state_encoding: StateEncoding,
                 sys_token: str,
                 usr_token: str,
                 sep_token: str,
                 ):
        self.envs: List[OldCTSEnv] = [_patch_env(fn()) for fn in env_fns]
        self.sys_token = sys_token
        self.usr_token = usr_token
        self.sep_token = sep_token

        self.state_encoding = state_encoding
        # setup state space info
        self.action_space = gym.spaces.Discrete(state_encoding.space_dims.num_actions)
        if state_encoding.action_config.in_state_space == True:
            # state space: max. node degree (#actions) x state dim
            self.observation_space = gym.spaces.Box(low=float('-inf'), high=float('inf'), shape=(state_encoding.space_dims.num_actions, state_encoding.space_dims.state_vector,)) #, dtype=np.float32)
        else:
            self.observation_space = gym.spaces.Box(low=float('-inf'), high=float('inf'), shape=(state_encoding.space_dims.state_vector,)) #, dtype=np.float32)

        VecEnv.__init__(self, len(env_fns), self.observation_space, self.action_space)
        self.keys, shapes, dtypes = obs_space_info(self.observation_space)

        self.buf_obs: List[Dict[str, Any]] = [{} for _ in range(self.num_envs)]
        self.buf_dones = np.zeros((self.num_envs,), dtype=bool)
        self.buf_rews = np.zeros((self.num_envs,), dtype=np.float32)
        self.buf_infos: List[Dict[str, Any]] = [{} for _ in range(self.num_envs)]
        self.metadata = {}

    def step_async(self, actions: np.ndarray) -> None:
        self.actions = actions

    def step_wait(self) -> VecEnvStepReturn:
        for env_idx in range(self.num_envs):
            obs, self.buf_rews[env_idx], terminated, truncated, info = self.envs[env_idx].step(
                self.actions[env_idx]
            )
            self.buf_infos[env_idx] = obs
            # convert to SB3 VecEnv api
            self.buf_dones[env_idx] = terminated
            # See https://github.com/openai/gym/issues/3102
            # Gym 0.26 introduces a breaking change
            self.buf_infos[env_idx]["TimeLimit.truncated"] = False

            if self.buf_dones[env_idx]:
                # save final observation where user can get it, then reset
                self.buf_infos[env_idx]["terminal_observation"] = obs
                # obs, self.reset_infos[env_idx] = self.envs[env_idx].reset()
                obs = self.envs[env_idx].reset()
            self._save_obs(env_idx, obs)

        # batch encode!
        return (self._obs_from_buf(), np.copy(self.buf_rews), np.copy(self.buf_dones), deepcopy(self.buf_infos))


    def seed(self, seed: Optional[int] = None) -> Sequence[Union[None, int]]:
        # Avoid circular import
        from stable_baselines3.common.utils import compat_gym_seed

        if seed is None:
            seed = np.random.randint(0, 2**32 - 1)
        seeds = []
        for idx, env in enumerate(self.envs):
            seeds.append(compat_gym_seed(env, seed=seed + idx))  # type: ignore[func-returns-value]
        return seeds


    def reset(self) -> VecEnvObs:
        for env_idx in range(self.num_envs):
            # TODO what did they write about reset infos and next state after termination?? READ
            obs = self.envs[env_idx].reset()
            self._save_obs(env_idx, obs)
        return self._obs_from_buf()


    def close(self) -> None:
        for env in self.envs:
            env.close()


    def get_images(self) -> Sequence[Optional[np.ndarray]]:
        if self.render_mode != "rgb_array":
            warnings.warn(
                f"The render mode is {self.render_mode}, but this method assumes it is `rgb_array` to obtain images."
            )
            return [None for _ in self.envs]
        return [env.render() for env in self.envs]  # type: ignore[misc]


    def render(self, mode: Optional[str] = None) -> Optional[np.ndarray]:
        """
        Gym environment rendering. If there are multiple environments then
        they are tiled together in one image via ``BaseVecEnv.render()``.

        :param mode: The rendering type.
        """
        return super().render(mode=mode)


    def _save_obs(self, env_idx: int, obs: VecEnvObs) -> None:
        self.buf_obs[env_idx] = obs

    def _obs_from_buf(self) -> VecEnvObs:
        # TODO convert self.buf_infos[env_idx]["terminal_observation"] into vectors for all buf elements that contain terminal_observation
        #
        # TODO better IDEA: buf_obs encoding contains all terminal_observations as well - just have to pick them by index and concatenate!
        # 
        batch_encoding = self.state_encoding.batch_encode(self.buf_obs, sys_token=self.sys_token, usr_token=self.usr_token, sep_token=self.sep_token)
        terminal_observations = [env_idx for env_idx, info in enumerate(self.buf_infos) if "terminal_observation" in info]
        for env_idx in terminal_observations:
            self.buf_infos[env_idx]['terminal_observation'] = batch_encoding[env_idx].detach().clone()
        return batch_encoding
    
    @property 
    def current_episode(self) -> int:
        return sum([env.current_episode for env in self.envs])
    
    @property
    def turn_counter(self) -> int:
        return sum([env.turn_counter for env in self.envs])


    def get_attr(self, attr_name: str, indices: VecEnvIndices = None) -> List[Any]:
        """Return attribute from vectorized environment (see base class)."""
        target_envs = self._get_target_envs(indices)
        return [getattr(env_i, attr_name) for env_i in target_envs]


    def set_attr(self, attr_name: str, value: Any, indices: VecEnvIndices = None) -> None:
        """Set attribute inside vectorized environments (see base class)."""
        target_envs = self._get_target_envs(indices)
        for env_i in target_envs:
            setattr(env_i, attr_name, value)


    def env_method(self, method_name: str, *method_args, indices: VecEnvIndices = None, **method_kwargs) -> List[Any]:
        """Call instance methods of vectorized environments."""
        target_envs = self._get_target_envs(indices)
        return [getattr(env_i, method_name)(*method_args, **method_kwargs) for env_i in target_envs]


    def env_is_wrapped(self, wrapper_class: Type[gym.Wrapper], indices: VecEnvIndices = None) -> List[bool]:
        """Check if worker environments are wrapped with a given wrapper"""
        target_envs = self._get_target_envs(indices)
        # Import here to avoid a circular import
        from stable_baselines3.common import env_util

        return [env_util.is_wrapped(env_i, wrapper_class) for env_i in target_envs]


    def _get_target_envs(self, indices: VecEnvIndices) -> List[gym.Env]:
        indices = self._get_indices(indices)
        return [self.envs[i] for i in indices]

    def stats_asked_goals_free(self):
        values = list(itertools.chain(*[env.asked_goals_free for env in self.envs]))
        if len(values) == 0:
            return 0.0
        return mean(values)

    def stats_asked_goals_guided(self):
        values = list(itertools.chain(*[env.asked_goals_guided for env in self.envs]))
        if len(values) == 0:
            return 0.0
        return mean(values)

    def stats_reached_goals_free(self):
        values = list(itertools.chain(*[env.reached_goals_free for env in self.envs]))
        if len(values) == 0:
            return 0.0
        return mean(values)

    def stats_reached_goals_guided(self):
        values = list(itertools.chain(*[env.reached_goals_guided for env in self.envs]))
        if len(values) == 0:
            return 0.0
        return mean(values)
    
    def stats_goal_node_coverage_free(self):
        coverage_dicts = [env.free_env.goal_node_coverage for env in self.envs if hasattr(env, "free_env")]
        coverage_dict = self._join_dicts(coverage_dicts)
        return len(coverage_dict) / self.envs[0].data.num_free_goal_nodes

    def stats_goal_node_coverage_guided(self):
        coverage_dicts = [env.guided_env.goal_node_coverage for env in self.envs if hasattr(env, "guided_env")]
        coverage_dict = self._join_dicts(coverage_dicts)
        return len(coverage_dict) / self.envs[0].data.num_guided_goal_nodes
    
    def _join_dicts(self, dicts: List[DefaultDict[str, float]]) -> Dict[str, float]:
        """
        Sums the values from each dict by key
        """
        result = {}
        keys = set(itertools.chain(*[d.keys() for d in dicts]))
        for key in keys:
            result[key] = sum([d[key] for d in dicts])
        return result

    def stats_node_coverage(self):
        coverage_dicts = []
        for env in self.envs:
            if hasattr(env, "free_env"):
                coverage_dicts.append(env.free_env.node_coverage)
            if hasattr(env, "guided_env"): 
                coverage_dicts.append(env.guided_env.node_coverage)
        coverage_dict = self._join_dicts(coverage_dicts)
        return len(coverage_dict) / len(self.envs[0].data.node_list)
    
    def stats_synonym_coverage_questions(self):
        coverage_dict = self._join_dicts([env.free_env.coverage_question_synonyms for env in self.envs if hasattr(env, "free_env")])
        return len(coverage_dict) / len(self.envs[0].data.question_list)

    def stats_synonym_coverage_answers(self):
        coverage_dicts = []
        for env in self.envs:
            if hasattr(env, "guided_env"):
                coverage_dicts.append(env.guided_env.coverage_answer_synonyms)
            if hasattr(env, "free_env"):
                coverage_dicts.append(env.free_env.coverage_answer_synonyms)
        coverage_dict = self._join_dicts(coverage_dicts)
        return len(coverage_dict) / self.envs[0].data.num_answer_synonyms
    
    def stats_actioncount_skips_indvalid(self):
        count = 0
        for env in self.envs:
            if hasattr(env, "free_env"):
                count += env.free_env.actioncount_skip_invalid
            if hasattr(env, "guided_env"): 
                count += env.guided_env.actioncount_skip_invalid
        return count

    def stats_actioncount_ask_variable_irrelevant(self):
        count = 0
        for env in self.envs:
            if hasattr(env, "free_env"):
                count += env.free_env.actioncount_ask_variable_irrelevant
            if hasattr(env, "guided_env"): 
                count += env.guided_env.actioncount_ask_variable_irrelevant
        return count

    def stats_actioncount_ask_question_irrelevant(self):
        count = 0
        for env in self.envs:
            if hasattr(env, "free_env"):
                count += env.free_env.actioncount_ask_question_irrelevant
            if hasattr(env, "guided_env"): 
                count += env.guided_env.actioncount_ask_question_irrelevant
        return count

    def stats_actioncount_missingvariable(self):
        count = 0
        for env in self.envs:
            if hasattr(env, "free_env"):
                count += env.free_env.actioncount_missingvariable
            if hasattr(env, "guided_env"): 
                count += env.guided_env.actioncount_missingvariable
        return count
