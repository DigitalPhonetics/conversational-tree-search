from collections import defaultdict
from functools import reduce
import itertools
from statistics import mean
from typing import Any, Dict, List, Tuple, Union
import logging
import random
from copy import deepcopy

import gym
from gym import spaces
import torch
from config import ActionType
from data.parsers.answerTemplateParser import AnswerTemplateParser
from data.parsers.logicParser import LogicTemplateParser
from data.parsers.parserValueProvider import RealValueBackend

from encoding.state import StateEncoding
from data.dataset import GraphDataset, Answer, DialogNode, NodeType
from rl.utils import AutoSkipMode, EnvInfo, EnvironmentMode, State, rand_remove_questionmark
from simulation.goal import UserGoalGenerator, UserResponse, DummyGoal, ImpossibleGoalError, VariableValue



class DialogEnvironment(gym.Env):
    def __init__(self, dialog_tree: GraphDataset, state_enc: StateEncoding,
                    mode: EnvironmentMode,
                    max_steps: int = None, user_patience: int = 3,
                    normalize_rewards: bool = True,
                    stop_when_reaching_goal: bool = False,  # ignored in guided mode
                    dialog_faq_ratio: float = 0.0,
                    log_to_file: Union[None, str, logging.Logger] = None,
                    env_id: int = 1,
                    goal_gen: UserGoalGenerator = None,
                    logic_parser: LogicTemplateParser = None,
                    answer_template_parser: AnswerTemplateParser = None,
                    return_obs: bool = True,
                    auto_skip: AutoSkipMode = AutoSkipMode.NONE,
                    similarity_model = None):
        """
        Args:
            stop_when_reaching_goal: If True, environment sets 'done' Flag once goal is reached.
            dialog_faq_ratio: how many percent of the generated goals should be step-by-step dialogs vs. faq dialogs
        """
        self.mode = mode
        self.return_obs = return_obs
        self.env_id = env_id
        self.state_enc = state_enc
        self.max_steps = max_steps if max_steps else 2 * dialog_tree.get_max_tree_depth() # if no max step is set, choose automatically 2*tree depth
        self.user_patience = user_patience
        self.stop_when_reaching_goal = stop_when_reaching_goal
        self.normalize_rewards = normalize_rewards
        self.auto_skip = auto_skip
        self.similarity_model = similarity_model

        self.dialog_faq_ratio = dialog_faq_ratio
        self.num_faqbased_dialogs = 0
        self.num_turnbased_dialogs = 0

        # load dialog tree info
        self.dialogTree = dialog_tree
        # setup user goal generator
        self.logicParser = LogicTemplateParser() if not logic_parser else logic_parser
        self.answer_template_parser = AnswerTemplateParser() if not answer_template_parser else answer_template_parser
        self.value_backend = RealValueBackend(dialog_tree)
        self.goal_gen = UserGoalGenerator(dialog_tree=dialog_tree, value_backend=self.value_backend) if not goal_gen else goal_gen

        # setup state space
        self.observation_space = spaces.Box(low=-9999999., high=9999999., shape=(self.state_enc.space_dims.state_vector,))
        self.action_space = spaces.Discrete(self.state_enc.space_dims.action_vector)

        # reward normalization
        self.max_reward = 4*dialog_tree.get_max_tree_depth() if normalize_rewards else 1.0

        # auto-skipping (text level)
        # if auto_skip:
        #   self.similarity_model = AnswerSimilarityEncoding(device=adapter.device, model_name='distiluse-base-multilingual-cased-v2', dialog_tree=dialog_tree)

        self.episode_log = []
        self.log_to_file = log_to_file
        if not isinstance(log_to_file, type(None)):
            print("Single Env: Logging to file", log_to_file)
            if not isinstance(log_to_file, logging.Logger):
                print("No log handler - Creating one ")
                self.logger = logging.getLogger("env" + mode.name)
                file_handler = logging.FileHandler(log_to_file, mode='w')
                file_handler.setLevel(logging.INFO)
                self.logger.addHandler(file_handler)
            else:
                self.logger = log_to_file
        else:
            print("Single env - no logger - logging to console for mode: ", mode.name)
            self.logger = logging.getLogger("env" + mode.name)

        # logging.basicConfig(filename=log_to_file, encoding='utf-8', filemode='w', level=logging.DEBUG)
        self.logger.info(f"{self.env_id}$ ENVIRONMENT INFO")
        self.logger.info(f"{self.env_id}$ - state {self.state_enc.space_dims.state_vector}")
        self.logger.info(f"{self.env_id}$ - actions {self.state_enc.space_dims.num_actions}")
        self.logger.info(f"{self.env_id}$ - normalization {self.max_reward}")
            
        self.current_episode = 0

        # coverage stats
        self.goal_node_coverage_free = defaultdict(int)
        self.goal_node_coverage_guided = defaultdict(int)
        self.node_coverage = defaultdict(int)
        self.coverage_faqs = defaultdict(int)
        self.coverage_synonyms = defaultdict(int)
        self.coverage_variables = defaultdict(lambda: defaultdict(int))

    def get_goal_node_coverage_free(self):
        return len(self.goal_node_coverage_free) / self.dialogTree.num_free_goal_nodes
    
    def get_goal_node_coverage_guided(self):
        return len(self.goal_node_coverage_guided) / self.dialogTree.num_guided_goal_nodes

    def get_node_coverage(self):
        return len(self.node_coverage) / len(self.dialogTree.node_list)
    
    def get_coverage_faqs(self):
        return len(self.coverage_faqs) / len(self.dialogTree.question_list)

    def get_coverage_synonyms(self):
        return len(self.coverage_synonyms) / self.dialogTree.num_answer_synonyms
    
    def get_coverage_variables(self):
        return {
            "CITY": len(self.coverage_variables["CITY"]) / len(self.dialogTree.city_keys),
            "COUNTRY": len(self.coverage_variables["COUNTRY"]) / len(self.dialogTree.country_keys)
        }

    def _transform_dialog_history(self):
        # interleave system and user utterances
        # Returns: List[Tuple(sys: str, usr: str)]
        usr_history = self.user_utterances_history
        if len(usr_history) < len(self.system_utterances_history):
            usr_history = usr_history + [""]
        assert len(usr_history) == len(self.system_utterances_history)
        return list(itertools.chain(zip([utterance for utterance in self.system_utterances_history], [utterance for utterance in self.user_utterances_history])))
        
    def _get_obs(self) -> Dict[EnvInfo, Any]:
        """ convert current dialog node to state vector, based on configuration 
        
        for each answer in node, we have 1 action: 
            * follow answer
        we have 1 additional action:
            * ASK (independent of answer)

        Returns:
            state-action embedding: [1, num_answers + 2, state_size] (torch.FloatTensor) if self.config.ACTION_CONFIG.ACTIONS_IN_STATE_SPACE
                              else: [1, state_size]

        NOTE: the policy has to deal with the state-action return matrix instead of a single state return vector
              and forward each of the actions individually
        """
        if self.return_obs:
            obs = {
                EnvInfo.ASKED_GOAL: self.asked_goal if self.is_faq_mode else mean(self.asked_goals if self.asked_goals else [0.0]),
                EnvInfo.BELIEFSTATE: deepcopy(self.bst),
                EnvInfo.CURRENT_USER_UTTERANCE: deepcopy(self.current_user_utterance),
                EnvInfo.DIALOG_NODE_KEY: self.current_node.key,
                EnvInfo.ENV_MODE: self.mode,
                EnvInfo.EPISODE: self.current_episode,
                EnvInfo.EPISODE_LENGTH: self.current_step,
                EnvInfo.EPISODE_REWARD: self.episode_reward,
                EnvInfo.GOAL: deepcopy(self.goal),
                EnvInfo.INITIAL_USER_UTTERANCE: deepcopy(self.initial_user_utterance),
                EnvInfo.IS_FAQ: self.is_faq_mode,
                EnvInfo.LAST_SYSTEM_ACT: self.last_action_idx,
                EnvInfo.REACHED_GOAL_ONCE: self.reached_goal_once if self.is_faq_mode else mean(self.reached_goals if self.reached_goals else [0.0]),
                EnvInfo.SYSTEM_UTTERANCE_HISTORY: deepcopy(self.system_utterances_history),
                EnvInfo.USER_UTTERANCE_HISTORY: deepcopy(self.user_utterances_history),
            }
            return obs
        return None

    def _increment_episode_count(self):
        self.current_episode += 1

    def _her_guided_reset(self, goal_node: DialogNode, initial_user_utterance: str) -> Dict[State, Any]:
        self.user_answer_keys = defaultdict(int)
        self.visited_node_keys = defaultdict(int)
        self.episode_log = []
        self.bst = {}
        self._increment_episode_count()
        
        self.initial_user_utterance = deepcopy(initial_user_utterance)
        self.current_node = self.dialogTree.start_node.connected_node
        self.last_action_idx = ActionType.ASK.value # start by asking start node
       
        self.is_faq_mode = False
        self.goal_node = goal_node
        self.reached_goals = []
        self.asked_goals = []
        self.constraints = {}
        
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
        self.actioncount_ask = 0
        self.actioncount_skip = 0
        self.actioncount_ask_variable = 0
        self.actioncount_ask_variable_irrelevant = 0
        self.actioncount_ask_question = 0
        self.actioncount_ask_question_irrelevant = 0
        self.actioncount_skip_info = 0
        self.actioncount_skip_question = 0
        self.actioncount_skip_variable = 0
        self.actioncount_skip_invalid = 0
        self.actioncount_missingvariable = 0

        return self._get_obs()
        
    def _her_faq_reset(self, goal: DummyGoal) -> Dict[State, Any]:
        self.user_answer_keys = goal.answer_pks
        self.visited_node_keys = defaultdict(int)
        self.episode_log = []
        self.bst = {}
        self._increment_episode_count()
        
        self.current_node = self.dialogTree.start_node.connected_node
        self.last_action_idx = ActionType.ASK.value # start by asking start node
       
        self.is_faq_mode = True
        self.goal = goal
        self.goal_node = self.goal.goal_node
        self.initial_user_utterance = deepcopy(self.goal.initial_user_utterance)
        self.reached_goal_once = False
        self.asked_goal = False
        self.constraints = self.goal.variables 
        
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
        self.actioncount_ask = 0
        self.actioncount_skip = 0
        self.actioncount_ask_variable = 0
        self.actioncount_ask_variable_irrelevant = 0
        self.actioncount_ask_question = 0
        self.actioncount_ask_question_irrelevant = 0
        self.actioncount_skip_info = 0
        self.actioncount_skip_question = 0
        self.actioncount_skip_variable = 0
        self.actioncount_skip_invalid = 0
        self.actioncount_missingvariable = 0

        self.coverage_faqs[goal.faq_key] += 1
        self.goal_node_coverage_free[self.goal_node.key] += 1

        return self._get_obs()

    def reset(self) -> Dict[State, Any]:
        self.user_answer_keys = defaultdict(int)
        self.visited_node_keys = defaultdict(int)
        self.episode_log = []
        self.bst = {}
        self._increment_episode_count()
        
        self.current_node = self.dialogTree.start_node.connected_node
        self.last_action_idx = ActionType.ASK.value # start by asking start node
        self.goal = None

        if random.random() < self.dialog_faq_ratio:
            # draw guided goal (turn by turn)
            self.is_faq_mode = False
            first_answer, self.initial_user_utterance = self._draw_random_answer(self.current_node)
            self.goal_node = first_answer.connected_node
            self.reached_goals = []
            self.asked_goals = []
            self.constraints = {}

            self.goal_node_coverage_guided[self.goal_node.key] += 1
            self.coverage_synonyms[self.initial_user_utterance.replace("?", "")] += 1
        else:
            # draw free goal
            self.is_faq_mode = True
            while not self.goal:
                try:
                    self.goal = self.goal_gen.draw_goal()
                except ImpossibleGoalError:
                    # print("IMPOSSIBLE GOAL")
                    continue
                except ValueError:
                    continue
            self.goal_node = self.goal.goal_node
            self.initial_user_utterance = deepcopy(self.goal.initial_user_utterance)
            self.reached_goal_once = False
            self.asked_goal = False
            self.constraints = self.goal.variables 
            
            self.goal_node_coverage_free[self.goal_node.key] += 1
            self.coverage_faqs[self.goal.faq_key] += 1
        
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

        # node stats
        self.nodecount_info = 0
        self.nodecount_variable = 0
        self.nodecount_question = 0
        self.nodecount_logic = 0

        # action stats
        self.actioncount_ask = 0
        self.actioncount_skip = 0
        self.actioncount_ask_variable = 0
        self.actioncount_ask_variable_irrelevant = 0
        self.actioncount_ask_question = 0
        self.actioncount_ask_question_irrelevant = 0
        self.actioncount_skip_info = 0
        self.actioncount_skip_question = 0
        self.actioncount_skip_variable = 0
        self.actioncount_skip_invalid = 0
        self.actioncount_missingvariable = 0

        # Logging
        self.episode_log.append(f'{self.env_id}-{self.current_episode}$ ======== RESET =========')
        self.episode_log.append(f'{self.env_id}-{self.current_episode}$ GOAL: ({"FREE" if self.is_faq_mode else "GUIDED"}): {self.goal_node.key} {self.goal_node.text[:75]}') 
        self.episode_log.append(f'{self.env_id}-{self.current_episode}$ CONSTRAINTS: {self.constraints}')
        self.episode_log.append(f'{self.env_id}-{self.current_episode}$ INITIAL UTTERANCE: {self.initial_user_utterance}') 
        return self._get_obs()

    def _draw_random_answer(self, node: DialogNode) -> Tuple[Answer, str]:
        if not node.key in self.user_answer_keys:
            # answer for this node is requested for the 1st time -> draw new goal for next turn
            self.user_answer_keys[node.key] = UserResponse(relevant=True, answer_key=node.random_answer().key)
        answer = self.dialogTree.answers_by_key[self.user_answer_keys[node.key].answer_key]
        user_utterance = rand_remove_questionmark(random.choice(self.dialogTree.answer_synonyms[answer.text.lower()]))
        return answer, user_utterance

    def _check_user_patience_reached(self) -> bool:
        return self.visited_node_keys[self.current_node.key] > self.user_patience # +1 because  skip action to node already counts as 1 visit

    def _reached_max_length(self) -> bool:
        return self.current_step == self.max_steps - 1 # -1 because first turn is counted as step 0

    def _update_action_counters(self, action: int):
        if action >= ActionType.SKIP and self.last_action_idx >= ActionType.SKIP: # it's only a REAL skip if we didn't ask before
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
                    self.reached_goal_once = True
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
                    self.goal_node = self.dialogTree.nodes_by_key[random.choice(list(goal_candidates))]
            else:
                self.goal_node = None # reached end of dialog tree
        elif self.current_node.node_type == NodeType.INFO:
            # adapt goal to next node
            if self.current_node.connected_node:
                self.goal_node = self.current_node.connected_node
            else:
                return False # reached end of tree
        else:
            # logic node: handle and skip to next viable node
            _, done = self._handle_logic_nodes() 
        self.episode_log.append(f'{self.env_id}-{self.current_episode}$ -> NEXT GOAL: ({self.goal_node.key if self.goal_node else "NONE"}) {self.goal_node.text[:50] if self.goal_node else ""} ')
        if done or isinstance(self.goal_node, type(None)):
            self.episode_log.append(f'{self.env_id}-{self.current_episode} NO NEXT GOAL - LOGIC NODE WITHOUT REQUIRED BST VARIABLE$')
            return False
        self.goal_node_coverage_guided[self.goal_node.key] += 1
        return True

    def step(self, action: int, _replayed_user_utterance: Union[str, None] = "") -> Tuple[Dict[State, Any], float, bool, Dict[EnvInfo, Any]]:
        self.episode_log.append(f'{self.env_id}-{self.current_episode}$ - --- Turn {self.current_step}')
        if self.current_node:
            self.episode_log.append(f'{self.env_id}-{self.current_episode}$ FROM NODE: {self.current_node.node_type} - {self.current_node.key} - {self.current_node.text[:75]}')
        prev_node_key = self.current_node.key
        done = False
        reward = 0.0
        self.episode_log.append(f'{self.env_id}-{self.current_episode}$ -> ACTION: {action}')
        self.current_step += 1
        self.current_user_utterance = "" # reset each turn, will be assigned in _step_X method if action is ASK
        self._update_action_counters(action)
        self.actioncount_ask += action == ActionType.ASK
        self.actioncount_skip += action >= ActionType.SKIP
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
            reward, done = self._step_free(action, _replayed_user_utterance) if self.is_faq_mode else self._step_guided(action)
            
            self.episode_log.append(f'{self.env_id}-{self.current_episode}$ -> USER UTTERANCE: {self.current_user_utterance}')
            self.episode_log.append(f'{self.env_id}-{self.current_episode}$ TO NODE: {self.current_node.node_type} - {self.current_node.key} - {self.current_node.text[:75]}')

            # update history
            self.last_action_idx = action
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
                        if self.is_faq_mode:
                            if self.current_node.node_type == NodeType.QUESTION:
                                if self.auto_skip == AutoSkipMode.ORACLE:
                                    # semantic level: choose correct answer to jump to (assume perfect NLU)
                                    response: UserResponse = self.user_answer_keys[self.current_node.key]
                                    skip_action = self.dialogTree.answers_by_key[response.answer_key]
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
                    elif self.current_node.connected_node:
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
                    self.last_action_idx = action
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
        return obs, reward/self.max_reward, done, obs

    def _get_transition(self, action: int) -> Tuple[DialogNode, None]:
        num_answers = len(self.current_node.answers)
        answer_idx = action - 1 # subtract ASK action
        if num_answers == 0 and self.current_node.connected_node:
            if answer_idx == 0:
                return self.current_node.connected_node
            else:
                self.episode_log.append(f'{self.env_id}-{self.current_episode}$ -> INVALID TRANSITION TO CONNECTED_NODE {answer_idx}')
                return None
        if answer_idx >= num_answers:
            self.episode_log.append(f'{self.env_id}-{self.current_episode}$ -> INVALID TRANSITION TO ANSWER {answer_idx}, HAS ONLY {num_answers}')
            return None # invalid action: stay at current node
        
        next_node = self.current_node.answer_by_index(answer_idx).connected_node
        self.node_coverage[next_node.key] += 1
        return next_node

    def _step_guided(self, action: int) -> Tuple[float, bool]:     
        reward = 0.0 
        done = False

        if action == ActionType.ASK:
            assert self.current_node.node_type != NodeType.LOGIC

            # ASK
            if self.last_action_idx == ActionType.ASK:
                if self.auto_skip != AutoSkipMode.NONE:
                    reward += 2 # ask is also skip
                else:
                    reward -= 1 # don't ask multiple times in a row!
                if self.auto_skip != AutoSkipMode.NONE:
                    # last ask brought us to correct goal
                    if not self._choose_next_goal_node_guided():
                        done = True
            else:
                reward += 2 # important to ask each node
                self.asked_goals.append(1.0 * (self.reached_goals and self.reached_goals[-1] == 1)) # only asked correct goal if we jumped to the correct node in the previous transition
                if self.reached_goals and self.reached_goals[-1] == 1:
                    # update goal: previous action was skip to correct node (STOP not possible)
                    # -> draw new goal
                    if not self._choose_next_goal_node_guided():
                        done = True

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
                    self.bst[var.name] = VariableValue(var_name=var.name, var_type=var.type).draw_value(self.dialogTree) # variable is asked for the 1st time
                self.current_user_utterance = str(deepcopy(self.bst[var.name]))
                self.actioncount_ask_variable += 1
                self.coverage_variables[var.name][self.bst[var.name]] += 1
                self.episode_log.append(f'{self.env_id}-{self.current_episode}$ -> VAR NAME: {var.name}, VALUE: {self.bst[var.name]}')
            elif self.current_node.node_type == NodeType.QUESTION:
                # get user reply
                if not self.goal_node:
                    # there are no more options to visit new nodes from current node -> stop dialog
                    done = True
                else:
                    answer = self.current_node.answer_by_connected_node(self.goal_node)
                    self.current_user_utterance = rand_remove_questionmark(random.choice(self.dialogTree.answer_synonyms[answer.text.lower()]))
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
                if self.last_action_idx >= ActionType.SKIP:
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
                    if not self.goal_node:
                        # there are no more options to visit new nodes from current node -> stop dialog
                        done = True
        return reward, done

    def _step_free(self, action: int, _replayed_user_utterance: Union[str, None]) -> Tuple[float, bool]:
        reward = 0.0 
        done = False

        if action == ActionType.ASK:
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
                    var = self.answer_template_parser.find_variable(self.current_node.answers[0].text)

                    # check if variable was already asked
                    if var.name in self.bst:
                        reward -= 1 # variable value already known
                    
                    # get user reply and save to bst
                    var_instance = self.goal.get_user_input(self.current_node, self.bst)
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
                            self.current_user_utterance = rand_remove_questionmark(random.choice(self.dialogTree.answer_synonyms[answer.text.lower()]))
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


    @classmethod 
    def make_env(cls, state_enc: StateEncoding, max_episode_steps: int = 100):
        gym.envs.register(
            id='Reisekosten-v0',
            entry_point='simulation.dialogenv:DialogEnvironment',
            max_episode_steps=max_episode_steps,
            kwargs={"state_enc": state_enc}
        )
        env = gym.make('Reisekosten-v0')
        return env




class ParallelDialogEnvironment(gym.Env):
    def __init__(self, dialog_tree: GraphDataset, state_enc: StateEncoding,
                    mode: EnvironmentMode,
                    max_steps: int = None, user_patience: int = 3,
                    normalize_rewards: bool = True,
                    stop_when_reaching_goal: bool = False,
                    dialog_faq_ratio: float = 0.0,
                    log_to_file: str = None,
                    n_envs: int = 1,
                    auto_skip: AutoSkipMode = AutoSkipMode.NONE,
                    similarity_model=None) -> None:
        
        self.state_enc = state_enc
        self.log_to_file = log_to_file

        # logging
        if log_to_file:
            self.logger = logging.getLogger("env" + mode.name)
            self.logger.setLevel(logging.INFO)
            print("ParallelEnv: Logging to file", log_to_file)
            print("ParallelEnv: No log handler - creating one")
            file_handler = logging.FileHandler(log_to_file, mode='w')
            file_handler.setLevel(logging.INFO)
            self.logger.addHandler(file_handler)
        else:
            self.logger = None

        # # load dialog tree info
        self.dialogTree = dialog_tree
        self.logicParser = LogicTemplateParser()
        self.value_backend = RealValueBackend(dialog_tree)
        self.goal_gen = UserGoalGenerator(dialog_tree=self.dialogTree, value_backend=self.value_backend)
        self.answer_template_parser = AnswerTemplateParser()
        self.envs = [DialogEnvironment(dialog_tree=dialog_tree, state_enc=state_enc, mode=mode, 
                                        max_steps=max_steps, user_patience=user_patience,
                                        normalize_rewards=normalize_rewards, stop_when_reaching_goal=stop_when_reaching_goal,
                                        dialog_faq_ratio=dialog_faq_ratio, log_to_file=self.logger,
                                        env_id=env_id, goal_gen=self.goal_gen,
                                        logic_parser=self.logicParser, answer_template_parser=self.answer_template_parser,
                                        auto_skip=auto_skip, similarity_model=similarity_model) for env_id in range(n_envs)
                        ]
        self.max_steps = max_steps if max_steps else 2 * dialog_tree.get_max_tree_depth() # if no max step is set, choose automatically 2*tree depth
        self.user_patience = user_patience
        self.stop_when_reaching_goal = stop_when_reaching_goal
        self.dialog_faq_ratio = dialog_faq_ratio

        # setup state space
        self.observation_space = spaces.Box(low=-9999999., high=9999999., shape=(self.state_enc.space_dims.state_vector,))
        self.action_space = spaces.Discrete(self.state_enc.space_dims.action_vector)


    def __len__(self):
        return len(self.envs)
    
    def step(self, actions: List[int]):
        # 
        observations = []
        rewards = []
        dones = []
        infos = []

        for a, env in zip(actions, self.envs):
            obs, reward, done, info = env.step(a)
            observations.append(obs)
            rewards.append(reward)
            dones.append(done)
            infos.append(info)

        return observations, rewards, dones, infos

    def reset(self) -> List[torch.FloatTensor]:
        return [env.reset() for env in self.envs]

    def reset_single(self, index: int) -> torch.FloatTensor:
        return self.envs[index].reset()

    @property
    def current_nodes(self) -> List[DialogNode]:
        return [env.current_node for env in self.envs]

    @property
    def current_nodes_keys(self) -> List[int]:
        return [env.current_node.key for env in self.envs]

    @property
    def current_episode(self) -> int:
        return sum([env.current_episode for env in self.envs])

    @property
    def actioncount_ask(self) -> int:
        return sum([env.actioncount_ask for env in self.envs])

    @property
    def actioncount_skip(self) -> int:
        return sum([env.actioncount_skip for env in self.envs])

    @property
    def actioncount_ask_variable_irrelevant(self) -> int:
        return sum([env.actioncount_ask_variable_irrelevant for env in self.envs])

    @property
    def actioncount_ask_question_irrelevant(self) -> int:
        return sum([env.actioncount_ask_question_irrelevant for env in self.envs])

    @property
    def actioncount_missingvariable(self) -> int:
        return sum([env.actioncount_missingvariable for env in self.envs])

    @property
    def num_faqbased_dialogs(self) -> int:
        return sum([env.num_faqbased_dialogs for env in self.envs])

    @property
    def num_turnbased_dialogs(self) -> int:
        return sum([env.num_turnbased_dialogs for env in self.envs])
    
    @property
    def max_reward(self) -> int:
        return self.envs[0].max_reward

    def close(self):
        for env in self.envs:
            env.close()

    def get_goal_node_coverage_free(self):
        return len(reduce(lambda d1, d2: set(d1).union(d2), [env.goal_node_coverage_free.keys() for env in self.envs])) / self.dialogTree.num_free_goal_nodes

    def get_goal_node_coverage_guided(self):
        return len(reduce(lambda d1, d2: set(d1).union(d2), [env.goal_node_coverage_guided.keys() for env in self.envs])) / self.dialogTree.num_guided_goal_nodes

    def get_node_coverage(self):
        return len(reduce(lambda d1, d2: set(d1).union(d2), [env.node_coverage.keys() for env in self.envs])) / len(self.dialogTree.node_list)

    def get_coverage_faqs(self):
        return len(reduce(lambda d1, d2: set(d1).union(d2), [env.coverage_faqs.keys() for env in self.envs])) / len(self.dialogTree.question_list)

    def get_coverage_synonyms(self):
        return len(reduce(lambda d1, d2: set(d1).union(d2), [env.coverage_synonyms.keys() for env in self.envs])) / self.dialogTree.num_answer_synonyms
    
    def get_coverage_variables(self):
        return {
            "CITY": len(reduce(lambda d1, d2: set(d1).union(d2), [env.coverage_variables["CITY"].keys() for env in self.envs])) / len(self.dialogTree.city_keys),
            "COUNTRY": len(reduce(lambda d1, d2: set(d1).union(d2), [env.coverage_variables["COUNTRY"].keys() for env in self.envs])) / len(self.dialogTree.country_keys)
        }