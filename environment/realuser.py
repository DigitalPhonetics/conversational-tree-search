from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, Set, Tuple, Union
import warnings


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
        
    def ask(self, replayed_user_utterance: Tuple[str, None]) -> Tuple[bool, float]:
        """ NOTE: ASK action is split into 2 parts: first part for reward, 2nd part for setting the user utterance """
        reward = 0.0
        # output system text
        print("ASKING", self.current_node.text)

        if self.auto_skip_mode != AutoSkipMode.NONE:
            reward -= 1 # because it is 2 actions

        return False, reward
    
    def needs_user_utterance(self) -> bool:
        return self.current_node.node_type in [NodeType.VARIABLE, NodeType.QUESTION]

    def set_user_utterance(self, user_utterance: str) -> str:
        self.first_turn = False
        if self.current_node.node_type == NodeType.VARIABLE:
            # get variable name
            var = self.answerParser.find_variable(self.current_node.answer_by_index(0).text)

            # check if variable was already asked
            if var.name in self.bst:
                reward -= 1 # variable value already known
            
            print("=== NLU ===")
            print(f"EXPECTING VAR {var.name} OF TYPE {var.type}")
            if var.name in ["CITY", "COUNTRY"]:
                nlu_results = self.nlu.extract_places(user_utterance)
                if var.name in nlu_results:
                    if len(nlu_results[var.name]) > 1:
                        return f"Please provide only a single value. Detected multiple values: {', '.join(nlu_results[var.name])}"
                    elif len(nlu_results[var.name]) == 0:
                        return f"Sorry, but the {var.name.lower()} you entered is unknown to the system. Please check spelling or try another value."
                    # found single value!
                    self.bst[var.name] = nlu_results[var.name][0]
            elif var.name == "TRIP_LENGTH":
                nlu_results = self.nlu._extract_time_span(user_utterance)
                if len(nlu_results['time_spans']) > 1:
                    return f"Please provide only a single value. Detected multiple values: {', '.join(nlu_results['time_spans'])}"
                elif len(nlu_results['time_span']) == 0:
                    return f"Sorry, but the time span you entered was not recognized by the system. Please rephrase."
                self.bst[var.name] = nlu_results['time_spans'][0]
            elif var.name == "PRIVATE_EXTENSION":
                # boolean
                nlu_results = self.nlu.extract_boolean(user_utterance)
                if len(nlu_results) > 1:
                    return f"Please provide only a single value. Detected multiple values: {', '.join(nlu_results)}" 
                elif len(nlu_results) == 0:
                    return f"Sorry, but the value you provided could not be interpreted as confirmation nor the opposite. Please try to rephrase."
                self.bst[var.name] = nlu_results[0]
            else:
                return "ERROR: Found unknown variable. Please report this problem."
            
            self.current_user_utterance = str(deepcopy(user_utterance))
            self.coverage_variables[var.name][self.bst[var.name]] += 1
        elif self.current_node.node_type == NodeType.QUESTION:
            self.current_user_utterance = deepcopy(user_utterance)

    def set_initial_user_utterance(self, initial_user_utterance: str):
        self.first_turn = False
        self.goal = RealUserGoal(initial_user_utterance=initial_user_utterance, delexicalised_initial_user_utterance=initial_user_utterance,
                                 goal_node_key=self.data.start_node.key, constraints=dict(), visited_ids=set())
        return self.post_reset()
    

    def reset(self):
        self.episode_reward = 0.0
        self.percieved_length = 0

        self.pre_reset()
        self.first_turn = True

        # Output first node
        # print(self.current_node.text)
        # Ask for initial user input
        