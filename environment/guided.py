from copy import deepcopy
import random
from statistics import mean
from typing import Tuple

from chatbot.adviser.app.rl.goal import UserResponse
from chatbot.adviser.app.rl.utils import rand_remove_questionmark
from config import ActionType

from data.dataset import Answer, DialogNode, GraphDataset, NodeType
from data.cache import Cache

from chatbot.adviser.app.answerTemplateParser import AnswerTemplateParser
from chatbot.adviser.app.logicParser import LogicTemplateParser
from chatbot.adviser.app.parserValueProvider import RealValueBackend
from chatbot.adviser.app.rl.goal import VariableValue
from chatbot.adviser.app.rl.utils import AutoSkipMode
from encoding.state import StateEncoding
from environment.base import BaseEnv



class GuidedEnvironment(BaseEnv):
    def __init__(self, env_id: int, cache: Cache, dataset: GraphDataset, state_encoding: StateEncoding,
            sys_token: str, usr_token: str, sep_token: str,
            max_steps: int, max_reward: float, user_patience: int,
            answer_parser: AnswerTemplateParser, logic_parser: LogicTemplateParser,
            value_backend: RealValueBackend,
            auto_skip: AutoSkipMode) -> None:
        super().__init__(env_id=env_id, cache=cache, dataset=dataset, state_encoding=state_encoding,
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

        self.episode_log.append(f'{self.env_id}-{self.current_episode}$ MODE: GUIDED') 
        return self.post_reset()

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
            var = self.answerParser.find_variable(answer.text)

            # check if variable was already asked
            if var.name in self.bst:
                reward -= 4 # variable value already known
                self.episode_log.append(f'{self.env_id}-{self.current_episode}$ -> VARIABLE ALREADY KNOWN')
            else:
                # draw random variable
                self.bst[var.name] = VariableValue(var_name=var.name, var_type=var.type).draw_value(self.data) # variable is asked for the 1st time
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
