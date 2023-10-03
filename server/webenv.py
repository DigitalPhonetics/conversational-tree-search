from copy import deepcopy
from typing import List, Tuple, Union
from config import ActionType


from data.dataset import GraphDataset, NodeType

from data.parsers.answerTemplateParser import AnswerTemplateParser
from data.parsers.logicParser import LogicTemplateParser
from data.parsers.parserValueProvider import RealValueBackend
from data.parsers.systemTemplateParser import SystemTemplateParser
from environment.realuser import RealUserEnvironment, RealUserGoal
from server.nlu import NLU
from utils.utils import AutoSkipMode

import re
url_pattern = re.compile(r'(<a\s+[^>]*href=")([^"]*)(")([^>]*>)')

class RealUserEnvironmentWeb(RealUserEnvironment):
    def __init__(self, user_id: int,
            dataset: GraphDataset, nlu: NLU,
            sys_token: str, usr_token: str, sep_token: str,
            max_steps: int, max_reward: float, user_patience: int,
            system_parser: SystemTemplateParser, answer_parser: AnswerTemplateParser, logic_parser: LogicTemplateParser,
            value_backend: RealValueBackend,
            auto_skip: AutoSkipMode, stop_on_invalid_skip: bool) -> None:
        super().__init__(user_id=user_id, dataset=dataset, nlu=nlu,
            sys_token=sys_token, usr_token=usr_token, sep_token=sep_token, 
            max_steps=max_steps, max_reward=max_reward, user_patience=user_patience,
            answer_parser=answer_parser, logic_parser=logic_parser, value_backend=value_backend,
            auto_skip=auto_skip, stop_on_invalid_skip=stop_on_invalid_skip)
        self.first_turn = False
        self.system_parser = system_parser

    def variable_already_known(self) -> bool:
        """ Checks whether the current node is a variable node, and if so, if its value is already known """
        if self.current_node.node_type == NodeType.VARIABLE:
            var = self.answerParser.find_variable(self.current_node.answer_by_index(0).text)
            return var.name in self.bst
        return False
    

    def get_current_node_markup(self) -> str:
        # replace links with alert
        markup = url_pattern.sub(r"""\1#\3 onclick="open_link_info()"\4""", self.current_node.markup)
        return self.system_parser.parse_template(markup, self.value_backend, self.bst)

    def reached_tree_end(self) -> bool:
        return not self.current_node or (len(self.current_node.answers) == 0 and not self.current_node.connected_node)

    def get_current_node_answer_candidates(self) -> List[str]:
        if self.current_node.node_type == NodeType.QUESTION:
            return [answer.text for answer in self.current_node.answers]
        elif self.current_node.node_type == NodeType.VARIABLE:
            var = self.answerParser.find_variable(self.current_node.answer_by_index(0).text)
            if var.type == "BOOLEAN":
                return ["yes", "no"]
            elif var.type == "LOCATION":
                if var.name == "COUNTRY":
                    return ["USA", "China", "UK", "Germany", "Egypt"]
            elif var.type == "TIMESPAN":
                return ["2 days", "3 weeks", "1 year"]
        return [] # no answer candidates

    def step(self, action: int, replayed_user_utterance: Tuple[str, None] = None):
        self.first_turn = False
        reward = 0.0
        done = False 
        self.prev_node = self.current_node
        self.current_user_utterance = replayed_user_utterance # reset user utterance for current turn

        if not self.reached_goal_once and self.goal.has_reached_goal_node(self.current_node):
            self.episode_log.append(f'{self.env_id}-{self.current_episode}$ REACHED GOAL')
            self.reached_goal_once = True

        # check if dialog should end
        if self.check_user_patience_reached(): 
            reward = -self.max_reward  # bad
            done = True
            self.episode_log.append(f'{self.env_id}-{self.current_episode}$ REACHED MAX USER PATIENCE')
        elif self.reached_max_length():
            done = True # already should have large negtative reward (expect if in guided mode, where max length could even be desired)
            self.episode_log.append(f'{self.env_id}-{self.current_episode}$ REACHED MAX LENGTH')
        else:
            assert self.current_node.node_type != NodeType.LOGIC
            if action == ActionType.ASK:
                self.percieved_length += 1
                self.actioncount_asks[self.current_node.node_type] += 1
                done, reward = self.ask(replayed_user_utterance)
                self.episode_log.append(f'{self.env_id}-{self.current_episode}$ ASKING NODE: {self.current_node.node_type.value} - {self.current_node.key} - {self.current_node.text[:100]}')
            else:
                self.actioncount_skips[self.current_node.node_type] += 1
                done, reward = self.skip(action-1) # get answer index by shifting actions to the left

                # handle logic node auto-transitioning here
                if not done:
                    reward, logic_done, did_handle_logic_node = self.handle_logic_and_varupdate_nodes(reward)
                    if did_handle_logic_node:
                        done = logic_done
                    self.episode_log.append(f'{self.env_id}-{self.current_episode}$ -> TURN REWARD: {reward}')

                    if isinstance(self.goal.goal_node_key, type(None)):
                        done = True # check if we reached end of dialog tree
                        self.episode_log.append(f'{self.env_id}-{self.current_episode}$ -> REACHED TREE END')

                # check if agent is on correct path
                if (not self.current_node) or (not self.current_node.key in self.goal.visited_ids):
                    self.on_path = False
                    if self.current_node:
                        # check if skip was correct locally (in case it is NOT the first turn of an FAQ-style dialog - in which case the answer will not be in the answer synonyms!)
                        if self.prev_node.node_type == NodeType.QUESTION and self.last_action_idx == ActionType.ASK and (self.env_mode == "guided" or (self.env_mode == "free" and self.user_utterances_history[-1] != self.initial_user_utterance)):
                            # we have user input for previous turn!
                            locally_correct = self.locally_correct_skip(prev_usr_utterance=self.user_utterances_history[-1], origin_node=self.prev_node, followup_node=self.current_node)
                            self.actioncount_skip_accuracy.append(1.0 if locally_correct else 0.0)
                    if self.stop_on_invalid_skip and (not done) and self.current_node:
                        # we're not at the end of the tree, but we took a wrong skip
                        done = True
                        reward = -self.max_reward
                if self.on_path:
                    # transition is on goal path! -> update index
                    self.last_valid_skip_transition_idx = self.current_step
                
            self.episode_log.append(f'{self.env_id}-{self.current_episode}$ -> USER UTTERANCE: {self.current_user_utterance}')
            if self.current_node:
                self.episode_log.append(f'{self.env_id}-{self.current_episode}$ TO NODE: {self.current_node.node_type.value} - {self.current_node.key} - {self.current_node.text[:100]}')

            # update history
            self.last_action_idx = action
            self.user_utterances_history.append(str(deepcopy(self.current_user_utterance)))
            if self.current_node:
                self.system_utterances_history.append(deepcopy(self.current_node.text))

            # update counters
            if self.current_node:
                self.visited_node_keys[self.current_node.key] += 1
            self.current_step += 1
            self.update_node_counters()
            self.update_action_counters(action)

            if (not done) and self.goal.goal_node_key and self.auto_skip_mode != AutoSkipMode.NONE and self.last_action_idx == ActionType.ASK:
                self.auto_skip()

        if not done:
            done = self.reached_tree_end()

        if not self.reached_goal_once and self.goal.has_reached_goal_node(self.current_node):
            self.episode_log.append(f'{self.env_id}-{self.current_episode}$ REACHED GOAL')
            self.reached_goal_once = True
       
        self.episode_reward += reward
        if done:
            self.reached_goals.append(float(self.reached_goal_once))
            self.asked_goals.append(float(self.asked_goal_once))
            # self.episode_log.append(f'{self.env_id}-{self.current_episode}$ -> TURN REWARD: {reward}')
            self.episode_log.append(f'{self.env_id}-{self.current_episode}$=> REACHED GOAL ONCE: {self.reached_goal_once}')
            self.episode_log.append(f'{self.env_id}-{self.current_episode}$=> ASKED GOAL ONCE: {self.asked_goal_once}')
            self.episode_log.append(f'{self.env_id}-{self.current_episode}$=> FINAL REWARD: {self.episode_reward}')
            self.episode_log.append(f'{self.env_id}-{self.current_episode}$=> PERCIEVED LENGTH: {self.percieved_length}')
            self.episode_log.append(f'{self.env_id}-{self.current_episode}$=> TOTAL LENGTH: {self.current_step}')

        obs = self.get_obs()
        reward /= self.max_reward 
        assert -1 <= reward <= 1, f"invalid reward normalization: {reward} not in [-1,1]"

        return obs, reward, done
    
    def check_variable_input(self, utterance: str) -> Union[str, None]:
        # Returns error string if problem
        # else return None
        # in UI, check if error string not None 
        #   -> step, if None
        #   -> don't step, write error msg if not None
        
        # get variable name
        var = self.answerParser.find_variable(self.current_node.answer_by_index(0).text)
        if var.name in ["CITY", "COUNTRY"]:
            nlu_results = self.nlu.extract_places(utterance)
            if var.name in nlu_results:
                if len(nlu_results[var.name]) > 1:
                    return f"Please provide only a single value. Detected multiple values: {', '.join(nlu_results[var.name])}"
                if len(nlu_results[var.name]) == 0:
                    if var.name == "COUNTRY":
                        return f"Sorry, but the {var.name.lower()} you entered is unknown to the system. Please check spelling or try another value."
                    elif var.name == "CITY":
                        return None # unkown cities will default to $REST
        elif var.name == "TRIP_LENGTH":
            nlu_results = self.nlu.extract_time(utterance)
            if len(nlu_results['time_spans']) > 1:
                return f"Please provide only a single value. Detected multiple values: {', '.join(nlu_results['time_spans'])}"
            elif len(nlu_results['time_spans']) == 0:
                return f"Sorry, but the time span you entered was not recognized by the system. Please rephrase."
        elif var.name == "PRIVATE_EXTENSION":
            # boolean
            nlu_results = self.nlu.extract_boolean(utterance)
            if len(nlu_results) > 1:
                return f"Please provide only a single value. Detected multiple values: {', '.join(nlu_results)}" 
            elif len(nlu_results) == 0:
                return f"Sorry, but the value you provided could not be interpreted as confirmation nor the opposite. Please try to rephrase."
        else:
            return "ERROR: Found unknown variable. Please report this problem."
        
        return None

    def ask(self, replayed_user_utterance: Tuple[str, None]) -> Tuple[bool, float]:
        reward = 0.0
        # output system text
        print("ASKING", self.current_node.text)

        if self.auto_skip_mode != AutoSkipMode.NONE:
            reward -= 1 # because it is 2 actions

        if self.current_node.node_type == NodeType.VARIABLE:
            # get variable name
            var = self.answerParser.find_variable(self.current_node.answer_by_index(0).text)

            # check if variable was already asked
            if var.name in self.bst:
                reward -= 1 # variable value already known
            
            # get user reply and save to bst
            if var.name in ["CITY", "COUNTRY"]:
                nlu_results = self.nlu.extract_places(replayed_user_utterance)
                if var.name in nlu_results and len(nlu_results[var.name]) > 0:
                    self.bst[var.name] = nlu_results[var.name][0]
                elif var.name == "CITY":
                    self.bst[var.name] = "$REST"
            elif var.name == "TRIP_LENGTH":
                nlu_results = self.nlu.extract_time(replayed_user_utterance)
                self.bst[var.name] = nlu_results['time_spans'][0]
            elif var.name == "PRIVATE_EXTENSION":
                # boolean
                nlu_results = self.nlu.extract_boolean(replayed_user_utterance)
                self.bst[var.name] = nlu_results[0]
            else:
                return "ERROR: Found unknown variable. Please report this problem."

            self.current_user_utterance = str(deepcopy(self.bst[var.name]))
            self.coverage_variables[var.name][self.bst[var.name]] += 1
        elif self.current_node.node_type == NodeType.QUESTION:
            self.current_user_utterance = deepcopy(replayed_user_utterance)

        return False, reward

    def reset(self, goal_node_id: int):
        self.first_turn = True
        self.pre_reset()
        if not isinstance(goal_node_id, type(None)):
            self.goal_node_id = goal_node_id
    
    def set_initial_user_utterance(self, initial_user_utterance: str, check_variables: bool = True):
        # TODO check for bst values in first utterance
        # (we don't know variable type / name here, so just have to see if anything matches)
        self.goal = RealUserGoal(initial_user_utterance=deepcopy(initial_user_utterance), delexicalised_initial_user_utterance=deepcopy(initial_user_utterance),
                                 goal_node_key=self.goal_node_id, constraints=dict(), visited_ids=set())

        # check locations
        if check_variables:
            nlu_results = self.nlu.extract_places(initial_user_utterance)
            if "COUNTRY" in nlu_results and len(nlu_results["COUNTRY"]) == 1:
                # exactly 1 match -> set
                self.bst["COUNTRY"] = nlu_results["COUNTRY"][0]
            if "CITY" in nlu_results and len(nlu_results["CITY"]) == 1:
                self.bst['CITY'] = nlu_results["CITY"][0]
        
        return self.post_reset()
    