from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, Set, Tuple, Union
import warnings
from config import ActionType


from data.dataset import DialogNode, GraphDataset, NodeType

from data.parsers.answerTemplateParser import AnswerTemplateParser
from data.parsers.logicParser import LogicTemplateParser
from data.parsers.parserValueProvider import RealValueBackend
from server.nlu import NLU
from utils.utils import AutoSkipMode
from encoding.state import StateEncoding
from environment.base import BaseEnv


@dataclass
class RealUserGoal:
    initial_user_utterance: str
    delexicalised_initial_user_utterance: str
    goal_node_key: str
    constraints: Dict[str, Any]
    visited_ids: Set[int]

    def has_reached_goal_node(self, candidate: DialogNode) -> bool:
        # TODO return true if chosen goal was reached
        warnings.warn("HAS REACHED GOAL NODE IS NOT YET IMPLEMENTED")
        return False




class RealUserEnvironment(BaseEnv):
    def __init__(self,
            dataset: GraphDataset, nlu: NLU,
            sys_token: str, usr_token: str, sep_token: str,
            max_steps: int, max_reward: float, user_patience: int,
            answer_parser: AnswerTemplateParser, logic_parser: LogicTemplateParser,
            value_backend: RealValueBackend,
            auto_skip: AutoSkipMode, stop_on_invalid_skip: bool) -> None:
        assert isinstance(auto_skip, AutoSkipMode)
        super().__init__(dataset=dataset,
            sys_token=sys_token, usr_token=usr_token, sep_token=sep_token, 
            max_steps=max_steps, max_reward=max_reward, user_patience=user_patience,
            answer_parser=answer_parser, logic_parser=logic_parser, value_backend=value_backend,
            auto_skip=auto_skip, stop_on_invalid_skip=stop_on_invalid_skip)
        self.nlu = nlu

    def reset(self):
        self.pre_reset()

        # Mock a goal node that we can never reach to keep the conversation alive
        # goal_node = DialogNode(key="syntheticGoalNode", text="Synthetic Goal Node", node_type=NodeType.INFO, answers=[], questions=[], connected_node=None)

        # Output first node
        print(self.current_node.text)
        # Ask for initial user input
        initial_user_utterance = deepcopy(input(">>"))
        self.goal = RealUserGoal(initial_user_utterance=initial_user_utterance, delexicalised_initial_user_utterance=initial_user_utterance,
                                 goal_node_key=self.data.start_node.key, constraints=dict(), visited_ids=set())

        return self.post_reset()

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
            var = self.answerParser.find_variable(self.current_node.answer_by_index(0).text)

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
        print("SKIPPING")
        
        next_node = self.get_transition(answer_index)

        if next_node:
            # valid transition
            self.current_node = next_node
            print("-> TO", self.current_node.text[:100])
        else:
            done = True
            print("REACHED END OF DIALOG TREE")
        return done, reward

    def reached_goal(self) -> Union[bool, float]:
        return False

    def asked_goal(self) -> Union[bool, float]:
        return False



class RealUserEnvironmentWeb(RealUserEnvironment):
    def __init__(self,
            dataset: GraphDataset, nlu: NLU,
            sys_token: str, usr_token: str, sep_token: str,
            max_steps: int, max_reward: float, user_patience: int,
            answer_parser: AnswerTemplateParser, logic_parser: LogicTemplateParser,
            value_backend: RealValueBackend,
            auto_skip: AutoSkipMode, stop_on_invalid_skip: bool) -> None:
        super().__init__(dataset=dataset, nlu=nlu,
            sys_token=sys_token, usr_token=usr_token, sep_token=sep_token, 
            max_steps=max_steps, max_reward=max_reward, user_patience=user_patience,
            answer_parser=answer_parser, logic_parser=logic_parser, value_backend=value_backend,
            auto_skip=auto_skip, stop_on_invalid_skip=stop_on_invalid_skip)
        self.first_turn = False

    def step(self, action: int, replayed_user_utterance: Tuple[str, None] = None):
        self.first_turn = False
        reward = 0.0
        done = False 
        self.prev_node = self.current_node
        self.current_user_utterance = replayed_user_utterance # reset user utterance for current turn

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
                elif len(nlu_results[var.name]) == 0:
                    return f"Sorry, but the {var.name.lower()} you entered is unknown to the system. Please check spelling or try another value."
        elif var.name == "TRIP_LENGTH":
            nlu_results = self.nlu.extract_time(utterance)
            if len(nlu_results['time_spans']) > 1:
                return f"Please provide only a single value. Detected multiple values: {', '.join(nlu_results['time_spans'])}"
            elif len(nlu_results['time_span']) == 0:
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
                if var.name in nlu_results:
                    self.bst[var.name] = nlu_results[var.name][0]
            elif var.name == "TRIP_LENGTH":
                nlu_results = self.nlu._extract_time_span(replayed_user_utterance)
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

    def reset(self):
        self.first_turn = True
        self.pre_reset()
    
    def set_initial_user_utterance(self, initial_user_utterance: str):
        # TODO check for bst values in first utterance
        # (we don't know variable type / name here, so just have to see if anything matches)
        self.goal = RealUserGoal(initial_user_utterance=deepcopy(initial_user_utterance), delexicalised_initial_user_utterance=deepcopy(initial_user_utterance),
                                 goal_node_key=self.data.start_node.key, constraints=dict(), visited_ids=set())

        # check locations
        nlu_results = self.nlu.extract_places(initial_user_utterance)
        if "COUNTRY" in nlu_results and len(nlu_results["COUNTRY"]) == 1:
            # exactly 1 match -> set
            self.bst["COUNTRY"] = nlu_results["COUNTRY"][0]
        if "CITY" in nlu_results and len(nlu_results["CITY"]) == 1:
            self.bst['CITY'] = nlu_results["CITY"][0]
        
        return self.post_reset()
    
        
    # def ask(self, replayed_user_utterance: Tuple[str, None]) -> Tuple[bool, float]:
    #     """ NOTE: ASK action is split into 2 parts: first part for reward, 2nd part for setting the user utterance """
    #     reward = 0.0
    #     # output system text
    #     print("ASKING", self.current_node.text)

    #     if self.auto_skip_mode != AutoSkipMode.NONE:
    #         reward -= 1 # because it is 2 actions

    #     return False, reward
    
    # def needs_user_utterance(self) -> bool:
    #     return self.current_node.node_type in [NodeType.VARIABLE, NodeType.QUESTION]

    # def set_user_utterance(self, user_utterance: str) -> str:
    #     self.first_turn = False
    #     if self.current_node.node_type == NodeType.VARIABLE:
    #         # get variable name
    #         var = self.answerParser.find_variable(self.current_node.answer_by_index(0).text)

    #         # check if variable was already asked
    #         if var.name in self.bst:
    #             reward -= 1 # variable value already known
            
    #         print("=== NLU ===")
    #         print(f"EXPECTING VAR {var.name} OF TYPE {var.type}")
    #         if var.name in ["CITY", "COUNTRY"]:
    #             nlu_results = self.nlu.extract_places(user_utterance)
    #             if var.name in nlu_results:
    #                 if len(nlu_results[var.name]) > 1:
    #                     return f"Please provide only a single value. Detected multiple values: {', '.join(nlu_results[var.name])}"
    #                 elif len(nlu_results[var.name]) == 0:
    #                     return f"Sorry, but the {var.name.lower()} you entered is unknown to the system. Please check spelling or try another value."
    #                 # found single value!
    #                 self.bst[var.name] = nlu_results[var.name][0]
    #         elif var.name == "TRIP_LENGTH":
    #             nlu_results = self.nlu._extract_time_span(user_utterance)
    #             if len(nlu_results['time_spans']) > 1:
    #                 return f"Please provide only a single value. Detected multiple values: {', '.join(nlu_results['time_spans'])}"
    #             elif len(nlu_results['time_span']) == 0:
    #                 return f"Sorry, but the time span you entered was not recognized by the system. Please rephrase."
    #             self.bst[var.name] = nlu_results['time_spans'][0]
    #         elif var.name == "PRIVATE_EXTENSION":
    #             # boolean
    #             nlu_results = self.nlu.extract_boolean(user_utterance)
    #             if len(nlu_results) > 1:
    #                 return f"Please provide only a single value. Detected multiple values: {', '.join(nlu_results)}" 
    #             elif len(nlu_results) == 0:
    #                 return f"Sorry, but the value you provided could not be interpreted as confirmation nor the opposite. Please try to rephrase."
    #             self.bst[var.name] = nlu_results[0]
    #         else:
    #             return "ERROR: Found unknown variable. Please report this problem."
            
    #         self.current_user_utterance = str(deepcopy(self.bst[var.name]))
    #         self.coverage_variables[var.name][self.bst[var.name]] += 1
    #     elif self.current_node.node_type == NodeType.QUESTION:
    #         self.current_user_utterance = deepcopy(user_utterance)

    # def set_initial_user_utterance(self, initial_user_utterance: str):
    #     self.first_turn = False
    #     self.goal = RealUserGoal(initial_user_utterance=initial_user_utterance, delexicalised_initial_user_utterance=initial_user_utterance,
    #                              goal_node_key=self.data.start_node.key, constraints=dict(), visited_ids=set())
    #     return self.post_reset()
    
    # def reset(self):
    #     self.episode_reward = 0.0
    #     self.percieved_length = 0

    #     self.pre_reset()
    #     self.first_turn = True

    # def step(self, action: int, replayed_user_utterance: Tuple[str, None] = None):
    #     reward = 0.0
    #     done = False 
    #     self.prev_node = self.current_node
    #     self.current_user_utterance = replayed_user_utterance # reset user utterance for current turn

    #     # check if dialog should end
    #     if self.check_user_patience_reached(): 
    #         reward = -self.max_reward  # bad
    #         done = True
    #         self.episode_log.append(f'{self.env_id}-{self.current_episode}$ REACHED MAX USER PATIENCE')
    #     elif self.reached_max_length():
    #         done = True # already should have large negtative reward (expect if in guided mode, where max length could even be desired)
    #         self.episode_log.append(f'{self.env_id}-{self.current_episode}$ REACHED MAX LENGTH')
    #     else:
    #         assert self.current_node.node_type != NodeType.LOGIC
    #         if action == ActionType.ASK:
    #             self.percieved_length += 1
    #             self.actioncount_asks[self.current_node.node_type] += 1
    #             done, reward = self.ask(replayed_user_utterance)
    #             self.episode_log.append(f'{self.env_id}-{self.current_episode}$ ASKING NODE: {self.current_node.node_type.value} - {self.current_node.key} - {self.current_node.text[:100]}')
    #         else:
    #             self.actioncount_skips[self.current_node.node_type] += 1
    #             done, reward = self.skip(action-1) # get answer index by shifting actions to the left

    #             # handle logic node auto-transitioning here
    #             if not done:
    #                 reward, logic_done, did_handle_logic_node = self.handle_logic_and_varupdate_nodes(reward)
    #                 if did_handle_logic_node:
    #                     done = logic_done
    #                 self.episode_log.append(f'{self.env_id}-{self.current_episode}$ -> TURN REWARD: {reward}')

    #                 if isinstance(self.goal.goal_node_key, type(None)):
    #                     done = True # check if we reached end of dialog tree
    #                     self.episode_log.append(f'{self.env_id}-{self.current_episode}$ -> REACHED TREE END')

    #             # check if agent is on correct path
    #             if (not self.current_node) or (not self.current_node.key in self.goal.visited_ids):
    #                 self.on_path = False
    #                 if self.current_node:
    #                     # check if skip was correct locally (in case it is NOT the first turn of an FAQ-style dialog - in which case the answer will not be in the answer synonyms!)
    #                     if self.prev_node.node_type == NodeType.QUESTION and self.last_action_idx == ActionType.ASK and (self.env_mode == "guided" or (self.env_mode == "free" and self.user_utterances_history[-1] != self.initial_user_utterance)):
    #                         # we have user input for previous turn!
    #                         locally_correct = self.locally_correct_skip(prev_usr_utterance=self.user_utterances_history[-1], origin_node=self.prev_node, followup_node=self.current_node)
    #                         self.actioncount_skip_accuracy.append(1.0 if locally_correct else 0.0)
    #                 if self.stop_on_invalid_skip and (not done) and self.current_node:
    #                     # we're not at the end of the tree, but we took a wrong skip
    #                     done = True
    #                     reward = -self.max_reward
    #             if self.on_path:
    #                 # transition is on goal path! -> update index
    #                 self.last_valid_skip_transition_idx = self.current_step
                
    #         self.episode_log.append(f'{self.env_id}-{self.current_episode}$ -> USER UTTERANCE: {self.current_user_utterance}')
    #         if self.current_node:
    #             self.episode_log.append(f'{self.env_id}-{self.current_episode}$ TO NODE: {self.current_node.node_type.value} - {self.current_node.key} - {self.current_node.text[:100]}')

    #         # update history
    #         self.last_action_idx = action
    #         self.user_utterances_history.append(str(deepcopy(self.current_user_utterance)))
    #         if self.current_node:
    #             self.system_utterances_history.append(deepcopy(self.current_node.text))

    #         # update counters
    #         if self.current_node:
    #             self.visited_node_keys[self.current_node.key] += 1
    #         self.current_step += 1
    #         self.update_node_counters()
    #         self.update_action_counters(action)

    #         if (not done) and self.goal.goal_node_key and self.auto_skip_mode != AutoSkipMode.NONE and self.last_action_idx == ActionType.ASK:
    #             self.auto_skip()

       
    #     self.episode_reward += reward
    #     if done:
    #         self.reached_goals.append(float(self.reached_goal_once))
    #         self.asked_goals.append(float(self.asked_goal_once))
    #         # self.episode_log.append(f'{self.env_id}-{self.current_episode}$ -> TURN REWARD: {reward}')
    #         self.episode_log.append(f'{self.env_id}-{self.current_episode}$=> REACHED GOAL ONCE: {self.reached_goal_once}')
    #         self.episode_log.append(f'{self.env_id}-{self.current_episode}$=> ASKED GOAL ONCE: {self.asked_goal_once}')
    #         self.episode_log.append(f'{self.env_id}-{self.current_episode}$=> FINAL REWARD: {self.episode_reward}')
    #         self.episode_log.append(f'{self.env_id}-{self.current_episode}$=> PERCIEVED LENGTH: {self.percieved_length}')
    #         self.episode_log.append(f'{self.env_id}-{self.current_episode}$=> TOTAL LENGTH: {self.current_step}')

    #     obs = self.get_obs()
    #     reward /= self.max_reward 
    #     assert -1 <= reward <= 1, f"invalid reward normalization: {reward} not in [-1,1]"

    #     return obs, reward, done


