from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
import itertools
import random
from typing import Tuple, Type, Union, Dict, List
import gym
from gym import spaces
import logging

# from chatbot.adviser.app.answerTemplateParser import AnswerTemplateParser
# from chatbot.adviser.app.logicParser import LogicTemplateParser
# from chatbot.adviser.app.parserValueProvider import RealValueBackend
# from chatbot.adviser.app.rl.dialogtree import DialogTree
# from chatbot.adviser.app.rl.goal import DummyGoal, ImpossibleGoalError, UserGoalGenerator, UserResponse, VariableValue
# from chatbot.adviser.app.rl.spaceAdapter import SpaceAdapter
# from chatbot.adviser.app.rl.utils import AutoSkipMode, EnvironmentMode, NodeType, StateEntry, EnvInfo, _load_a1_laenderliste, _load_answer_synonyms, rand_remove_questionmark
# from chatbot.adviser.app.rl.dataset import DialogAnswer, DialogNode
# from chatbot.adviser.app.rl.dataset import Dataset


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


# TODO restructure env
# TODO restructure space adapter
# TODO restructure models

# TODO add new config options:
# - chain_dialog_history -> sys_token, usr_token, sep_token

# TODO update function usage:
# - chain_dialog_history -> now returns 5 instead of 2 entries per list item
# - handle_logic_node -> doesn't return reward anymore, doesn't handle chained logic nodes anymore, doesn't log, doesn't update last_node, throws MissingBSTValue


class MissingBSTValue(Exception):
    def __init__(self, var_name: str, bst: dict) -> None:
        super().__init__(var_name, bst)


def chain_dialog_history(self, sys_utterances: List[str], usr_utterances: List[str], sys_token: str = "", usr_token: str = "", sep_token: str = "") -> List[Tuple[str,str,str,str,str]]:
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


# def handle_logic_node(current_node: DialogNode, bst: dict, logic_parser: LogicTemplateParser, value_backend: RealValueBackend, data: Dataset) -> DialogNode:
#     """
#     Evaluates the logic using the provided bst.
#     Then returns the follow-up node of the matching logic branch.
#     If the required variables could not be found in the bst, raise MissingBSTValue exception.

#     Returns:
#         Next dialog node according to bst and logic
#     """
#     assert current_node.node_type == NodeType.LOGIC.value

#     lhs = current_node.content.text
#     var_name = lhs.lstrip("{{").strip() # form: {{ VAR_NAME
#     if var_name in bst:
#         # don't assign reward, is automatic transition without agent interaction
#         # evaluate statement, choose correct branch and skip to connected node
#         default_answer = None
#         # for idx, answer in enumerate(self.current_node.answers.all()):
#         for answer in current_node.answers:
#             # check if full statement {{{lhs rhs}}} evaluates to True
#             rhs = answer.content.text
#             if not "DEFAULT" in rhs: # handle DEFAULT case last!
#                 if logic_parser.parse_template(f"{lhs} {rhs}", value_backend, bst):
#                     # evaluates to True, follow this path!
#                     default_answer = answer
#                     break
#             else:
#                 default_answer = answer
#         # TODO what if logic node is end node? -> handle
#         return data.node_by_key(default_answer.connected_node_key)
#     else:
#         raise MissingBSTValue(var_name, bst)


# def get_start_node(dialog_tree: DialogTree) -> DialogNode:
#     """
#     First node is "START" node -> follow connected node key to get to the first node with user content and return that
#     """
#     return dialog_tree.data.node_by_key(dialog_tree.get_start_node().connected_node_key)



# class BaseEnvironment(gym.Env):
#     def __init__(self) -> None:
#         super().__init__()

    


# # TODO move a1_laenderliste in dataset
# # TODO move answer_synonyms in dataset
# # TODO add dialog id and eval step to all logging
# # TODO idea: could we implement auto-skipping as a wrapper around the envs?

# class GuidedModeEnvironment(BaseEnvironment):
#     def __init__(self,
#                 dialog_tree: DialogTree,
#                 use_answer_synonyms: bool,
#                 mode: EnvironmentMode,
#                 noise: float,
#                 max_steps: int,
#                 user_patience: int,
#                 reward_normalization: float,
#                 a1_laenderliste: dict = None,
#                 logic_parser: LogicTemplateParser = None,
#                 answer_template_parser: AnswerTemplateParser = None,
#                 answer_synonyms: Dict[str, List[str]] = None,
#                 auto_skip: AutoSkipMode = AutoSkipMode.NONE,
#                 similarity_model = None,
#                 **kwargs) -> None:
#         super().__init__()

#         self.dialog_tree = dialog_tree
#         self.use_answer_synonyms = use_answer_synonyms
#         self.mode = mode
#         self.noise = noise
#         self.max_steps = max_steps
#         self.user_patience = user_patience
#         self.reward_normalization = reward_normalization
#         self.a1_laenderliste = a1_laenderliste
#         self.logic_parser = logic_parser
#         self.answer_template_parser = answer_template_parser
#         self.answer_synonyms = answer_synonyms
#         self.auto_skip = auto_skip
#         self.similarity_model = similarity_model

#         # counters
#         self.num_dialogs = 0

#     def draw_random_answer(self) -> Tuple[DialogAnswer, str]:
#         if not self.current_node.key in self.user_answer_keys:
#             # answer for this node is requested for the 1st time -> draw new goal for next turn
#             self.user_answer_keys[self.current_node.key] = UserResponse(relevant=True, answer_key=self.current_node.random_answer().key)
#         answer = self.dialog_tree.data.answer_by_key(self.user_answer_keys[self.current_node.key].answer_key)
#         user_utterance = rand_remove_questionmark(random.choice(self.answer_synonyms[answer.content.text.lower()]))
#         return answer, user_utterance

#     def reset(self):
#         self.current_node = get_start_node(self.dialog_tree)
#         self.user_answer_keys = defaultdict(int)
#         self.visited_node_keys = defaultdict(int)
#         self.bst = {}

#         first_answer, self.initial_user_utterance = self.draw_random_answer(self.current_node)
#         self.goal_node = self.dialog_tree.data.node_by_key(first_answer.connected_node_key)
#         self.current_user_utterance = deepcopy(self.initial_user_utterance)
#         self.user_utterances_history = [deepcopy(self.initial_user_utterance)]
#         self.system_utterances_history = [deepcopy(self.current_node.content.text)]

#         self.reached_goals = []
#         self.asked_goals = []
#         self.constraints = {}
#         self.last_action_index = 1 # start by ASK start node
#         self.visited_node_keys[self.current_node.key] = 1
#         self.num_dialogs += 1

#         self.goal_node_coverage_guided[self.goal_node.key] += 1
#         self.node_coverage[self.current_node.key] += 1
#         self.coverage_synonyms[self.initial_user_utterance.replace("?", "")] += 1


    
#     def step(self):
#         pass



# class FreeModeEnvironment(BaseEnvironment):
#     def __init__(self,
#                 dialog_tree: DialogTree,
#                 stop_action: bool,
#                 use_answer_synonyms: bool,
#                 mode: EnvironmentMode,
#                 noise: float,
#                 max_steps: int,
#                 user_patience: int,
#                 reward_normalization: float,
#                 stop_when_reaching_goal: bool,
#                 log_file: Union[None, str],
#                 env_id: int,
#                 goal_gen: UserGoalGenerator,
#                 a1_laenderliste: dict = None,
#                 logic_parser: LogicTemplateParser = None,
#                 answer_template_parser: AnswerTemplateParser = None,
#                 answer_synonyms: Dict[str, List[str]] = None,
#                 return_obs: bool = True,
#                 auto_skip: AutoSkipMode = AutoSkipMode.NONE,
#                 similarity_model = None,
#                 **kwargs) -> None:
#         super().__init__()

#     def reset(self):
#         self.current_node = get_start_node(self.dialog_tree)

#     def step(self, action: int):
#         pass



# class DialogEnvironment(gym.Env):
#     def __init__(self, 
#                 dialog_tree: DialogTree,
#                 stop_action: bool,
#                 use_answer_synonyms: bool,
#                 mode: EnvironmentMode,
#                 noise: float,
#                 max_steps: int,
#                 user_patience: int,
#                 normalize_rewards: bool,
#                 stop_when_reaching_goal: bool,
#                 log_file: Union[None, str],
#                 env_id: int,
#                 goal_gen: UserGoalGenerator,
#                 a1_laenderliste: dict = None,
#                 logic_parser: LogicTemplateParser = None,
#                 answer_template_parser: AnswerTemplateParser = None,
#                 answer_synonyms: Dict[str, List[str]] = None,
#                 return_obs: bool = True,
#                 auto_skip: AutoSkipMode = AutoSkipMode.NONE,
#                 similarity_model = None,
#                 use_joint_dataset: bool = False,
#                 env_ratios: Dict[Type, float] = {
#                     GuidedModeEnvironment: 0.5,
#                     FreeModeEnvironment: 0.5
#                 }) -> None:
#         super().__init__()
#         self.env_id = env_id


#         self.max_reward = 4*dialog_tree.get_max_tree_depth() if normalize_rewards else 1.0

#         # TODO create sub-envs, then forward calls to this env for the sub-envs
#         self.envs = {}
#         assert sum(env_ratios.values()) == 1.0, f"Environment ratios have to sum up to 1.0, but are only {sum(env_ratios.values())}"
#         for env_cls in env_ratios:
#             self.envs[env_cls] = {
#                 "instance": env_cls(dialog_tree=dialog_tree, stop_action=stop_action, use_answer_synonyms=use_answer_synonyms, mode=mode,
#                                     noise=noise, max_steps=max_steps, user_patience=user_patience, normalize_rewards=normalize_rewards,
#                                     stop_when_reaching_goal=stop_when_reaching_goal, log_file=log_file, goal_gen=goal_gen, a1_laenderliste=a1_laenderliste,
#                                     logic_parser=logic_parser, answer_template_parser=answer_template_parser, answer_synonyms=answer_synonyms,
#                                     return_obs=return_obs, auto_skip=auto_skip, similarity_model=similarity_model, use_joint_dataset=use_joint_dataset,
#                                     reward_normalization=1.0/self.max_reward) 
#             }


#     def step(self, action: int):
#         pass

#     def reset():
#         pass