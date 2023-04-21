from copy import deepcopy
from typing import Tuple, Union


from data.dataset import DialogNode, GraphDataset, NodeType

from chatbot.adviser.app.answerTemplateParser import AnswerTemplateParser
from chatbot.adviser.app.logicParser import LogicTemplateParser
from chatbot.adviser.app.parserValueProvider import RealValueBackend
from chatbot.adviser.app.rl.utils import AutoSkipMode
from environment.base import BaseEnv


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


