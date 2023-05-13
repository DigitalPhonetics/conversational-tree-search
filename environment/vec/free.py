from copy import deepcopy
import random
from typing import Tuple, Union

from chatbot.adviser.app.rl.goal import ImpossibleGoalError, UserGoalGenerator
from chatbot.adviser.app.rl.utils import rand_remove_questionmark
from data.cache import Cache

from data.dataset import GraphDataset, NodeType

from chatbot.adviser.app.answerTemplateParser import AnswerTemplateParser
from chatbot.adviser.app.logicParser import LogicTemplateParser
from chatbot.adviser.app.systemTemplateParser import SystemTemplateParser
from chatbot.adviser.app.parserValueProvider import RealValueBackend
from chatbot.adviser.app.rl.utils import AutoSkipMode
from environment.vec.base import BaseEnv


class FreeEnvironment(BaseEnv):
    def __init__(self, env_id: int, dataset: GraphDataset,
            sys_token: str, usr_token: str, sep_token: str,
            max_steps: int, max_reward: float, user_patience: int,
            stop_when_reaching_goal: bool,
            answer_parser: AnswerTemplateParser, system_parser: SystemTemplateParser, logic_parser: LogicTemplateParser,
            value_backend: RealValueBackend,
            auto_skip: AutoSkipMode) -> None:
        super().__init__(env_id=env_id, dataset=dataset,
            sys_token=sys_token, usr_token=usr_token, sep_token=sep_token, 
            max_steps=max_steps, max_reward=max_reward, user_patience=user_patience,
            answer_parser=answer_parser, logic_parser=logic_parser, value_backend=value_backend,
            auto_skip=auto_skip)
        self.goal_gen = UserGoalGenerator(graph=dataset, answer_parser=answer_parser,
            system_parser=system_parser, value_backend=value_backend)
        self.stop_when_reaching_goal = stop_when_reaching_goal

    def reset(self, current_episode: int, max_distance: int):
        self.pre_reset()

        self.goal = None
        while not self.goal:
            try:
                self.goal = self.goal_gen.draw_goal_free(max_distance)
            except ImpossibleGoalError:
                print("IMPOSSIBLE GOAL")
                continue
            except ValueError:
                print("VALUE ERROR")
                continue

        self.coverage_synonyms[self.goal.faq_key] += 1
        self.episode_log.append(f'{self.env_id}-{self.current_episode}$ MODE: Free') 
        return self.post_reset()

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
                var = self.answerParser.find_variable(self.current_node.answer_by_index(0).text)

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


