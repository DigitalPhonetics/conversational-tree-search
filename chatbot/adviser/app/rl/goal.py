from collections import defaultdict
from dataclasses import dataclass
import os
import random
import string
from typing import Any, Dict, List, Set, Tuple, Union
from chatbot.adviser.app.answerTemplateParser import AnswerTemplateParser
from chatbot.adviser.app.parserValueProvider import RealValueBackend
from chatbot.adviser.app.rl.utils import rand_remove_questionmark
from chatbot.adviser.app.systemTemplateParser import SystemTemplateParser
from copy import deepcopy
from data.dataset import GraphDataset, NodeType, Answer, DialogNode

class VariableValue:
    def __init__(self, var_name: str, var_type: str) -> None:
        self.var_name = var_name
        self.var_type = var_type

        self.lt_condition = None
        self.leq_condition = None
        self.eq_condition = None
        self.neq_condition = set()
        self.gt_condition = None
        self.geq_condition = None

    def add_default_condition(self, other_branch_conditions: List[Tuple[str, Any]]) -> bool:
        # invert conditions in same branch statement (DEFAULT is always == condition)
        # TODO: this is not sufficient, e.g. there could be multiple != statements and DEFAULT could trigger one of them
        for other_cond, other_val in other_branch_conditions:
            if other_cond == "==":
                return self.add_condition("!=", other_val)
            elif other_cond == "!=":
                return self.add_condition("==", other_val)
            elif other_cond == "<":
                return self.add_condition(">=", other_val)
            elif other_val == "<=":
                return self.add_condition(">", other_val)
            elif other_val == ">":
                return self.add_condition("<=", other_val)
            elif other_val == ">=":
                return self.add_condition("<", other_val)

    def add_condition(self, condition: str, value: Any) -> bool:
        """
        Args:
            condition: >,<,==,!=,>=,<=

        Returns:
            True, if no existing conditions are violated, else False
        """
        
        # TODO falls value = default, müssen die andern conditions ins Gegenteil umgewandelt werden
        if condition == "==":
            if self.eq_condition and self.eq_condition != value:
                return False
            if self.neq_condition and value in self.neq_condition:
                return False
            if self.lt_condition and self.lt_condition >= value:
                return False
            if self.leq_condition and self.leq_condition > value:
                return False
            if self.gt_condition and self.gt_condition <= value:
                return False
            if self.geq_condition and self.geq_condition < value:
                return False
            self.eq_condition = value
        elif condition == "!=":
            if self.eq_condition and self.eq_condition == value:
                return False
            self.neq_condition.add(value)
        elif condition == "<":
            if self.eq_condition and self.eq_condition >= value: # a = eq < lt
                return False
            if self.lt_condition:
                if self.lt_condition > value: # change bounds to lower value
                    self.lt_condition = value
                    return True
                else:
                    return True # keep lower bound
            if self.gt_condition and self.gt_condition >= value:  #  geq < a < lt
                return False # lower bound can't be greater than higher bound
            if self.geq_condition and self.geq_condition >= value: #  geq <= a < lt
                return False # lower bound can't be greater than higher bound
            self.lt_condition = value
        elif condition == "<=":
            if self.eq_condition and self.eq_condition > value: # a = eq <= lt
                return False
            if self.leq_condition:
                if self.leq_condition > value: # change bounds to lower value
                    self.leq_condition = value
                    return True
                else:
                    return True # keep lower bound
            if self.gt_condition and self.gt_condition > value:  #  geq < a <= lt
                return False # lower bound can't be greater than higher bound
            if self.geq_condition and self.geq_condition > value: #  gt <= a <= lt
                return False # lower bound can't be greater than higher bound
            self.leq_condition = value
        elif condition == ">":
            if self.eq_condition and self.eq_condition <= value: # gt < a = eq
                return False
            if self.gt_condition:
                if self.gt_condition < value: # change bounds to higher value
                    self.gt_condition = value
                    return True
                else:
                    return True # keep upper bound
            if self.lt_condition and self.lt_condition <= value:  #  gt < a < lt
                return False # lower bound can't be greater than higher bound
            if self.leq_condition and self.leq_condition <= value: #  gt < a <= lt
                return False # lower bound can't be greater than higher bound
            self.gt_condition = value
        elif condition == ">=":
            if self.eq_condition and self.eq_condition < value: # gt <= a
                return False
            if self.geq_condition:
                if self.geq_condition < value: # change bounds to upper value
                    self.geq_condition = value
                    return True
                else:
                    return True # keep upper bound
            if self.lt_condition and self.lt_condition <= value:  #  gt <= a < lt
                return False # lower bound can't be greater than higher bound
            if self.leq_condition and self.leq_condition < value: #  gt <= a <= lt
                return False # lower bound can't be greater than higher bound
            self.geq_condition = value
        return True
    
    def _draw_word(self) -> str:
        length = random.choice(list(range(15))) # max word length: 15
        return "".join([random.choice(list(string.ascii_lowercase + string.ascii_uppercase + string.digits + " ")) for _ in range(length)])
    
    def _draw_number(self) -> int:
        lower_bound = self.gt_condition
        if not lower_bound:
            lower_bound = self.geq_condition
        if self.gt_condition and self.geq_condition:
            lower_bound = max(float(self.gt_condition), float(self.geq_condition) - 1)
        if not lower_bound:
            lower_bound = 0
        
        upper_bound = self.lt_condition
        if not upper_bound:
            upper_bound = self.leq_condition
        if self.lt_condition and self.leq_condition:
            upper_bound = min(float(self.lt_condition), float(self.leq_condition) + 1)
        if not upper_bound:
            upper_bound = 10000 if not lower_bound else 100 * float(lower_bound)

        return random.randint(int(lower_bound), int(upper_bound))
        
    def draw_value(self, data: GraphDataset):
        """ Draw a value respecting the variable type as well as the given (valid) conditions """
        if self.eq_condition:
            return self.eq_condition

        if self.var_type == "TEXT":
            # create random text, only respect eq / neq restraints
            length = random.choice(list(range(15))) # max word length: 15
            word = self._draw_word()
            while word.lower() in self.neq_condition.lower():
                word = self._draw_word()
            # other conditions not applicable
            return word
        elif self.var_type == "NUMBER":
            return self._draw_number()
        elif self.var_type == "LOCATION":
            if "country" in self.var_name.lower():
                country = random.choice(list(data.countries.keys()))
                while country.lower() in set([val.lower() for val in self.neq_condition]):
                    country = random.choice(list(data.countries.keys()))
                return country
            else:
                city = random.choice(list(data.cities.keys()))
                while city.lower() in [val.lower() for val in self.neq_condition]:
                    city = random.choice(list(data.stadt.keys()))
                return city
        elif self.var_type == "TIMESPAN":
            # TODO implement
            return self._draw_number()
        elif self.var_type == "TIMEPOINT":
            # TODO implement
            raise NotImplementedError
        elif self.var_type == "BOOLEAN":
            if isinstance(self.eq_condition, bool):
                return self.eq_condition
            if len(self.neq_condition) > 0:
                return not list(self.neq_condition)[0]
            return random.choice([True, False])
                

@dataclass
class UserResponse:
    relevant: bool
    answer_key: int

@dataclass
class UserInput:
    relevant: bool
    var_name: str
    var_value: any


class ImpossibleGoalError(Exception):
    pass


@dataclass
class SearchPath:
    path: List[DialogNode]
    visited_ids: Set[int]
    variables: Dict[str, VariableValue]
    chosen_response_pks: Dict[str, str]




class UserGoal:
    def __init__(self, data: GraphDataset, start_node: DialogNode, goal_node: DialogNode, initial_user_utterance: str, 
                 answer_parser: AnswerTemplateParser, system_parser: SystemTemplateParser,
                 value_backend: RealValueBackend) -> None:
        self.goal_node_key = goal_node.key

        paths = self.expand_path(start_node ,SearchPath(path=[], visited_ids=set([]), variables={}, chosen_response_pks={}), answerParser=answer_parser)
        if len(paths) == 0:
            raise ImpossibleGoalError
        self.path = random.choice(paths) # choose random path
        self.variables = self._fill_variables(self.path.variables, data) # choose variable values
        self.answer_pks = self.path.chosen_response_pks
        self.visited_ids = self.path.visited_ids

        # substitute values for delexicalised faq questions (not in bst)
        substitution_vars = {}
        required_vars = system_parser.find_variables(initial_user_utterance)
        for var in required_vars:
            if not var in self.variables:
                # draw random value
                value = None
                if var == "COUNTRY":
                    value = data.countries[random.choice(data.country_keys)]
                elif var == "CITY":
                    value = data.cities[random.choice(data.city_keys)]
                substitution_vars[var] = value
            else:
                substitution_vars[var] = self.variables[var]
        self.constraints = self.path.variables
        self.delexicalised_initial_user_utterance = initial_user_utterance
        self.initial_user_utterance = system_parser.parse_template(initial_user_utterance, value_backend, substitution_vars)

    # depth first serach
    def expand_path(self, current_node: DialogNode, currentPath: SearchPath, answerParser: AnswerTemplateParser) -> List[SearchPath]:
        if current_node.key in currentPath.visited_ids:
            # break cycles
            return [] 

        # visit node
        visited_ids = currentPath.visited_ids.union([current_node.key])
        path = currentPath.path + [current_node]

        # check for goal
        if current_node.key == self.goal_node_key:
            # successful path -> return
            return [SearchPath(path, visited_ids, currentPath.variables, currentPath.chosen_response_pks)]

        # visit children
        if current_node.node_type == NodeType.VARIABLE:
            # get variable name and type
            assert len(current_node.answers) == 1, "Should have exactly 1 answer"
            var_answer = current_node.answer_by_index(0)
            var = answerParser.find_variable(var_answer.text)
            new_variables = deepcopy(currentPath.variables)
            if not var.name in currentPath.variables:
                new_variables[var.name] = VariableValue(var.name, var.type)
            if var_answer.connected_node:
                # expand path by following only possible child for userInputNode
                return self.expand_path(var_answer.connected_node,
                                        SearchPath(path=path,
                                                    visited_ids=visited_ids,
                                                    variables=new_variables,
                                                    chosen_response_pks=currentPath.chosen_response_pks),
                                        answerParser=answerParser)
        elif current_node.node_type in [NodeType.INFO, NodeType.START]:
            if current_node.connected_node:
                return self.expand_path(current_node.connected_node,
                                        SearchPath(path=path,
                                                    visited_ids=visited_ids,
                                                    variables=currentPath.variables,
                                                    chosen_response_pks=currentPath.chosen_response_pks),
                                        answerParser=answerParser)
                                                    
        elif current_node.node_type == NodeType.QUESTION:
            paths = []
            for answer in current_node.answers:
                # expand path for each possible answer
                if answer.connected_node:
                    new_chosen_response_pks = deepcopy(currentPath.chosen_response_pks)
                    new_chosen_response_pks[current_node.key] = answer.key
                    paths += self.expand_path(answer.connected_node,
                                                SearchPath(path=path,
                                                            visited_ids=visited_ids,
                                                            variables=currentPath.variables,
                                                            chosen_response_pks=new_chosen_response_pks),
                                                answerParser=answerParser)
            return paths
        elif current_node.node_type == NodeType.LOGIC:
            # generating fitting values is hard:
            # there might be two logicNodes on one path conditioned on the same variable, but the conditions can be different and have to satisfy both nodes' statements at the same time.
            # -> SAT-problem
            # Workaround: don't try to infer a value for these nodes, just save their connections
            # This solution only works as long as there are no possible contradictory branches with the logic nodes on a path for the same variable
            
            # variables should already be known since we require a user input node before one can use a variable in a logic node
            # get variable name of condition: LHS is of form {{VAR_NAME
            paths = []
            var_name = current_node.text.replace("{{", "").strip()
            # collect all conditions first to handle default case (if exists)
            condition_branches = []
            for answer in current_node.answers:
                # get condition and comparison value for current branch
                # RHS is of form: operator value}}
                cond_op, cond_val = answer.text.replace("}}", "").split()
                if not isinstance(cond_val, str):
                    # spaces were inside value, join back together
                    cond_val = " ".join(cond_val)
                cond_val = cond_val.strip('"') # remove quotation marks from string values
                condition_branches.append((cond_op, cond_val, answer))

            # expand paths following different conditions
            for cond_op, cond_val, answer in condition_branches:
                new_variables = deepcopy(currentPath.variables)
                if cond_val == "DEFAULT":
                    compatible = new_variables[var_name].add_default_condition([(other_op, other_val) for other_op, other_val, _ in condition_branches if other_val != "DEFAULT"])
                else:
                    compatible = new_variables[var_name].add_condition(cond_op, cond_val)
                
                # check if variable conditions are compatible
                # IF NOT -> abandon branch 
                if compatible and answer.connected_node:
                    paths += self.expand_path(answer.connected_node,
                                                SearchPath(path=path,
                                                            visited_ids=visited_ids,
                                                            variables=new_variables,
                                                            chosen_response_pks=currentPath.chosen_response_pks),
                                                answerParser=answerParser)
            return paths
        return []
     
    def __len__(self):
        return len(self.path.path)

    def has_reached_goal_node(self, candidate: DialogNode) -> bool:
        """ Returns True, if the candidate node is equal to the goal node, else False """
        return candidate.key == self.goal_node_key

    def _fill_variables(self, variables: Dict[str, VariableValue], data: GraphDataset) -> Dict[str, Any]:
        """ Realizes a dict of VariableValue entries into a dict of randomly drawn values, according to the restrictions """
        return {var_name: variables[var_name].draw_value(data) for var_name in variables}

    def get_user_response(self, current_node: DialogNode) -> Union[UserResponse, None]:
        assert current_node.node_type == NodeType.QUESTION

        # return answer leading to correct branch
        if current_node.key in self.answer_pks:
            # answer is relevant to user goal -> return it
            return UserResponse(relevant=True, answer_key=self.answer_pks[current_node.key])
        else:
            # answer is not relevant to user goal -> pick one at random -> diminishes reward
            # if current_node.answers.count() > 0:
            if len(current_node.answers) > 0:
                # answer_key = random.choice(current_node.answers.values_list("key", flat=True))
                answer_key = current_node.random_answer().key
                return UserResponse(relevant=False, answer_key=answer_key)
        return None

    def get_user_input(self, current_node: DialogNode, bst: Dict[str, any], data: GraphDataset, answerParser: AnswerTemplateParser) -> UserInput:
        assert current_node.node_type == NodeType.VARIABLE
        # assert current_node.answers.count() == 1
        assert len(current_node.answers) == 1
        # TODO add generated / paraphrased utterances?

        # get variable value for node
        # var_answer: DialogAnswer = current_node.answers.first() # should have exactly 1 answer
        var_answer = current_node.answer_by_index(0) # should have exactly 1 answer
        var = answerParser.find_variable(var_answer.text)
        if var.name in self.variables:
            # variable is relevant to user goal -> return it
            return UserInput(relevant=True, var_name=var.name, var_value=self.variables[var.name])
        else:
            # variable is not relevant to user goal -> make one up and return it
            if var.name in bst:
                return UserInput(relevant=False, var_name=var.name, var_value=bst[var.name])
            return UserInput(relevant=False, var_name=var.name, var_value=VariableValue(var_name=var.name, var_type=var.type).draw_value(data))
   


class UserGoalGenerator:
    def __init__(self, graph: GraphDataset, 
            answer_parser: AnswerTemplateParser, system_parser: SystemTemplateParser,
            value_backend: RealValueBackend) -> None:

        self.graph = graph
        self.value_backend = value_backend

        self._goal_nodes_by_distance() # sets self._guided_goal_candidates, self._free_goal_candidates

        self.answer_parser = answer_parser
        self.system_parser = system_parser

    def _is_first_node(self, node: DialogNode) -> bool:
        # first node should not be a goal candidate, since system starts there
        return node.key == self.graph.start_node.connected_node.key

    def _is_free_goal_candidate(self, node: DialogNode) -> bool:
        return len(node.questions) > 0 and not self._is_first_node(node)

    def _is_guided_goal_candidate(self, node: DialogNode) -> bool:
        return node.node_type in [NodeType.INFO, NodeType.QUESTION, NodeType.VARIABLE] and not self._is_first_node(node)

    def _goal_nodes_by_distance(self):
        # 1. create a list of all nodes that are goal node candidates for
        # a) guided mode
        # b) free mode
        self._guided_goal_candidates: Dict[int, List[DialogNode]] = defaultdict(list)
        self._free_goal_candidates: Dict[int, List[DialogNode]] = defaultdict(list)

        # 2. try to find the shortest path to each goal node candidate (-> trivial, if we search goal node candidates by breadth-first search)
        # 3. create a dict that contains a list of nodes with distance(start, goal) <= dict key
        # (meaning, key + 1 is a superset of key)
        level = 0
        current_level_nodes = [self.graph.start_node.connected_node]
        visited_node_ids = set() # cycle breaking
        while len(current_level_nodes) > 0:
            # include all nodes from previous level, s.t. current level is superset of previous level
            self._guided_goal_candidates[level].extend(self._guided_goal_candidates[level-1])
            self._free_goal_candidates[level].extend(self._free_goal_candidates[level-1])

            # traverse current node level, append all children to next level nodes
            next_level_nodes = []
            for current_node in current_level_nodes:
                if current_node.key in visited_node_ids:
                    continue
                visited_node_ids.add(current_node.key)

                # check if current node is a candidate for a free or guided goal
                if self._is_free_goal_candidate(current_node):
                    # node is candidate for free mode
                    self._free_goal_candidates[level].append(current_node)
                if self._is_guided_goal_candidate(current_node):
                    # node is candidate for guided mode (neither start nor logic node)
                    self._guided_goal_candidates[level].append(current_node)
                
                # add children of node to next level nodes
                if current_node.connected_node:
                    next_level_nodes.append(current_node.connected_node)
                    assert len(current_node.answers) == 0
                elif len(current_node.answers) > 0:
                    next_level_nodes += [answer.connected_node for answer in current_node.answers]
            # continue with next level breadth search
            current_level_nodes = next_level_nodes
            level += 1

    def _get_distance_free(self, max_distance: int):
        assert self.graph.get_max_tree_depth() >= max(self._guided_goal_candidates.keys())
        assert self.graph.get_max_tree_depth() >= max(self._free_goal_candidates.keys())
        # choose max. distance
        return min(max_distance, max(self._free_goal_candidates.keys())) if max_distance > 0 else max(self._free_goal_candidates.keys())

    def _get_distance_guided(self, max_distance: int):
        assert self.graph.get_max_tree_depth() >= max(self._guided_goal_candidates.keys())
        assert self.graph.get_max_tree_depth() >= max(self._free_goal_candidates.keys())
        # choose max. distance
        return min(max_distance, max(self._guided_goal_candidates.keys())) if max_distance > 0 else max(self._guided_goal_candidates.keys())

    def draw_goal_guided(self, max_distance: int) -> UserGoal:
        """
        Draw a guided goal with maximum specified distance from the start node.
        If max_distance = 0, all distances will be used.
        """
        # sample node from range [1, max_distance]
        candidate = random.choice(self._guided_goal_candidates[self._get_distance_guided(max_distance)])
        # construct user goal
        goal = UserGoal(data=self.graph, start_node=self.graph.start_node.connected_node, goal_node=candidate,
                        initial_user_utterance="", 
                        answer_parser=self.answer_parser, system_parser=self.system_parser, value_backend=self.value_backend)
        # get initial user utterance from first transition
        initial_answer: Answer = self.graph.answers_by_key[goal.get_user_response(self.graph.start_node.connected_node).answer_key]
        goal.delexicalised_initial_user_utterance = rand_remove_questionmark(random.choice(self.graph.answer_synonyms[initial_answer.text.lower()]))
        goal.initial_user_utterance = deepcopy(goal.delexicalised_initial_user_utterance)
        return goal

    def draw_goal_free(self, max_distance: int) -> UserGoal:
        """
        Draw a free goal with maximum specified distance from the start node.
        If max_distance = 0, all distances will be used.
        """
        # sample node from range [1, max_distance]
        assert len(self._free_goal_candidates[self._get_distance_free(max_distance)]) > 0, f"no questions associated with nodes for distance {len(self._free_goal_candidates[self._get_distance_free(max_distance)])}"
        candidate = random.choice(self._free_goal_candidates[self._get_distance_free(max_distance)])
        # construct user goal
        question = candidate.random_question()
        return UserGoal(data=self.graph, start_node=self.graph.start_node.connected_node, goal_node=candidate,
                        initial_user_utterance=question.text,
                        answer_parser=self.answer_parser, system_parser=self.system_parser, value_backend=self.value_backend)


@dataclass
class DummyGoal:
    goal_node_key: str
    initial_user_utterance: str
    delexicalised_initial_user_utterance: str
    constraints: Dict[str, any]
    answer_pks: Dict[int, int]
    visited_ids: Set[int]

    def has_reached_goal_node(self, node: DialogNode) -> bool:
        return node.key == self.goal_node_key
    
    def get_user_response(self, current_node: DialogNode) -> Union[UserResponse, None]:
        assert current_node.node_type == NodeType.QUESTION

        # return answer leading to correct branch
        return UserResponse(relevant=True, answer_key=self.answer_pks[current_node.key])

    def get_user_input(self, current_node: DialogNode, bst: Dict[str, any], data: GraphDataset, answerParser: AnswerTemplateParser) -> UserInput:
        assert current_node.node_type == NodeType.VARIABLE
        # assert current_node.answers.count() == 1
        assert len(current_node.answers) == 1
        # TODO add generated / paraphrased utterances?

        # get variable value for node
        # var_answer: DialogAnswer = current_node.answers.first() # should have exactly 1 answer
        var_answer = current_node.answer_by_index(0) # should have exactly 1 answer
        var = answerParser.find_variable(var_answer.text)
        # if var.name in self.variables:
            # variable is relevant to user goal -> return it
        return UserInput(relevant=True, var_name=var.name, var_value=self.constraints[var.name])
        # else:
        #     # variable is not relevant to user goal -> make one up and return it
        #     if var.name in bst:
        #         return UserInput(relevant=False, var_name=var.name, var_value=bst[var.name])
        #     return UserInput(relevant=False, var_name=var.name, var_value=VariableValue(var_name=var.name, var_type=var.type).draw_value())
   
