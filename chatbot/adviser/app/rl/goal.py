from dataclasses import dataclass
from pathlib import Path
import random
import string
from typing import Any, Dict, List, Set, Tuple, Union
from chatbot.adviser.app.answerTemplateParser import AnswerTemplateParser
from chatbot.adviser.app.logicParser import LogicTemplateParser
from chatbot.adviser.app.parserValueProvider import RealValueBackend
from chatbot.adviser.app.rl.dialogtree import DialogTree
from chatbot.adviser.app.systemTemplateParser import SystemTemplateParser
from chatbot.adviser.app.rl.dataset import DialogAnswer, DialogNode, FAQQuestion
from copy import deepcopy
import json
import chatbot.adviser.app.rl.dataset as Data

resource_dir = Path(".", 'chatbot', 'static', 'chatbot', 'faq_cache')
nlu_resource_dir = Path(".", 'chatbot', 'static', 'chatbot', 'nlu_resources')

# TODO load location values once and then cache instead of reloading every time a new goal is drawn
class LocationValues:
    def __init__(self) -> None:
        # class for recognizing country and city names as well as time expressions in text
        with open(nlu_resource_dir / 'country_synonyms.json', 'r') as f:
            country_synonyms = json.load(f)
            self.country_keys = [country.lower() for country in country_synonyms.keys()]
            self.countries = {country.lower(): country for country in country_synonyms.keys()}
            self.countries.update({country_syn.lower(): country for country, country_syns in country_synonyms.items()
                                    for country_syn in country_syns})

        with open(nlu_resource_dir / 'city_synonyms.json', 'r') as f:
            city_synonyms = json.load(f)
            self.city_keys = [city.lower() for city in city_synonyms.keys()]
            self.cities = {city.lower(): city for city in city_synonyms.keys() if city != '$REST'}
            self.cities.update({city_syn.lower(): city for city, city_syns in city_synonyms.items()
                                for city_syn in city_syns})

locations = LocationValues()

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
        
    def draw_value(self):
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
            if "land" in self.var_name.lower():
                land = random.choice(list(locations.countries.keys()))
                while land.lower() in set([val.lower() for val in self.neq_condition]):
                    land = random.choice(list(locations.countries.keys()))
                return land
            else:
                stadt = random.choice(list(locations.cities.keys()))
                while stadt.lower() in [val.lower() for val in self.neq_condition]:
                    stadt = random.choice(list(locations.stadt.keys()))
                return stadt
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
    def __init__(self, start_node: DialogNode, goal_node: DialogNode, faq_key: int, initial_user_utterance: str, 
                 answer_parser: AnswerTemplateParser, logic_parser: LogicTemplateParser, system_parser: SystemTemplateParser,
                 value_backend: RealValueBackend) -> None:
        self.goal_node = goal_node
        self.faq_key = faq_key
        self.answerParser = answer_parser
        self.logicParser = logic_parser

        paths = self.expand_path(start_node ,SearchPath(path=[], visited_ids=set([]), variables={}, chosen_response_pks={}))
        if len(paths) == 0:
            raise ImpossibleGoalError
        self.path = random.choice(paths) # choose random path
        self.variables = self._fill_variables(self.path.variables) # choose variable values
        self.answer_pks = self.path.chosen_response_pks

        # substitute values for delexicalised faq questions (not in bst)
        substitution_vars = {}
        required_vars = system_parser.find_variables(initial_user_utterance)
        for var in required_vars:
            if not var in self.variables:
                # draw random value
                value = None
                if var == "LAND":
                    value = locations.countries[random.choice(locations.country_keys)]
                elif var == "STADT":
                    value = locations.cities[random.choice(locations.city_keys)]
                substitution_vars[var] = value
            else:
                substitution_vars[var] = self.variables[var]
        self.initial_user_utterance = system_parser.parse_template(initial_user_utterance, value_backend, substitution_vars)

    # depth first serach
    def expand_path(self, current_node: DialogNode, currentPath: SearchPath) -> List[SearchPath]:
        if current_node.key in currentPath.visited_ids:
            # break cycles
            return [] 

        # visit node
        visited_ids = currentPath.visited_ids.union([current_node.key])
        path = currentPath.path + [current_node]

        # check for goal
        if current_node.key == self.goal_node.key:
            # successful path -> return
            return [SearchPath(path, visited_ids, currentPath.variables, currentPath.chosen_response_pks)]

        # visit children
        if current_node.node_type == "userInputNode":
            # get variable name and type
            assert len(current_node.answers) == 1, "Should have exactly 1 answer"
            var_answer: DialogAnswer = current_node.answers[0]
            var = self.answerParser.find_variable(var_answer.content.text)
            new_variables = deepcopy(currentPath.variables)
            if not var.name in currentPath.variables:
                new_variables[var.name] = VariableValue(var.name, var.type)
            if var_answer.connected_node_key:
                # expand path by following only possible child for userInputNode
                return self.expand_path(Data.objects[self.goal_node.version].node_by_key(var_answer.connected_node_key),
                                        SearchPath(path=path,
                                                    visited_ids=visited_ids,
                                                    variables=new_variables,
                                                    chosen_response_pks=currentPath.chosen_response_pks))
        elif current_node.node_type in ["infoNode", "startNode"]:
            if current_node.connected_node_key:
                return self.expand_path(Data.objects[self.goal_node.version].node_by_key(current_node.connected_node_key),
                                        SearchPath(path=path,
                                                    visited_ids=visited_ids,
                                                    variables=currentPath.variables,
                                                    chosen_response_pks=currentPath.chosen_response_pks))
        elif current_node.node_type in ["userResponseNode"]:
            paths = []
            for answer in current_node.answers:
                # expand path for each possible answer
                if answer.connected_node_key:
                    new_chosen_response_pks = deepcopy(currentPath.chosen_response_pks)
                    new_chosen_response_pks[current_node.key] = answer.key
                    paths += self.expand_path(Data.objects[self.goal_node.version].node_by_key(answer.connected_node_key),
                                                SearchPath(path=path,
                                                            visited_ids=visited_ids,
                                                            variables=currentPath.variables,
                                                            chosen_response_pks=new_chosen_response_pks))
            return paths
        elif current_node.node_type == "logicNode":
            # generating fitting values is hard:
            # there might be two logicNodes on one path conditioned on the same variable, but the conditions can be different and have to satisfy both nodes' statements at the same time.
            # -> SAT-problem
            # Workaround: don't try to infer a value for these nodes, just save their connections
            # This solution only works as long as there are no possible contradictory branches with the logic nodes on a path for the same variable
            
            # variables should already be known since we require a user input node before one can use a variable in a logic node
            # get variable name of condition: LHS is of form {{VAR_NAME
            paths = []
            var_name = current_node.content.text.replace("{{", "").strip()
            # collect all conditions first to handle default case (if exists)
            condition_branches = []
            for answer in current_node.answers:
                # get condition and comparison value for current branch
                # RHS is of form: operator value}}
                cond_op, cond_val = answer.content.text.replace("}}", "").split()
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
                if compatible and answer.connected_node_key:
                    paths += self.expand_path(Data.objects[self.goal_node.version].node_by_key(answer.connected_node_key),
                                                SearchPath(path=path,
                                                            visited_ids=visited_ids,
                                                            variables=new_variables,
                                                            chosen_response_pks=currentPath.chosen_response_pks))
            return paths
     
    def __len__(self):
        return len(self.path.path)

    def has_reached_goal_node(self, candidate: DialogNode) -> bool:
        """ Returns True, if the candidate node is equal to the goal node, else False """
        return candidate.key == self.goal_node.key

    def _fill_variables(self, variables: Dict[str, VariableValue]) -> Dict[str, Any]:
        """ Realizes a dict of VariableValue entries into a dict of randomly drawn values, according to the restrictions """
        return {var_name: variables[var_name].draw_value() for var_name in variables}

    def get_user_response(self, current_node: DialogNode) -> Tuple[UserResponse, None]:
        assert current_node.node_type == "userResponseNode"

        # return answer leading to correct branch
        if current_node.key in self.answer_pks:
            # answer is relevant to user goal -> return it
            return UserResponse(relevant=True, answer_key=self.answer_pks[current_node.key])
        else:
            # answer is not relevant to user goal -> pick one at random -> diminishes reward
            # if current_node.answers.count() > 0:
            if current_node.answer_count() > 0:
                # answer_key = random.choice(current_node.answers.values_list("key", flat=True))
                answer_key = current_node.random_answer().key
                return UserResponse(relevant=False, answer_key=answer_key)
        return None

    def get_user_input(self, current_node: DialogNode, bst: Dict[str, any]) -> UserInput:
        assert current_node.node_type == 'userInputNode'
        # assert current_node.answers.count() == 1
        assert current_node.answer_count() == 1
        # TODO add generated / paraphrased utterances?

        # get variable value for node
        # var_answer: DialogAnswer = current_node.answers.first() # should have exactly 1 answer
        var_answer: DialogAnswer = current_node.answers[0] # should have exactly 1 answer
        var = self.answerParser.find_variable(var_answer.content.text)
        if var.name in self.variables:
            # variable is relevant to user goal -> return it
            return UserInput(relevant=True, var_name=var.name, var_value=self.variables[var.name])
        else:
            # variable is not relevant to user goal -> make one up and return it
            if var.name in bst:
                return UserInput(relevant=False, var_name=var.name, var_value=bst[var.name])
            return UserInput(relevant=False, var_name=var.name, var_value=VariableValue(var_name=var.name, var_type=var.type).draw_value())
   


class UserGoalGenerator:
    def __init__(self, dialog_tree: DialogTree, value_backend: RealValueBackend, paraphrase_fraction: float = 0.0, generate_fraction: bool = 0.0) -> None:
        assert 0 <= paraphrase_fraction <= 1
        assert 0 <= generate_fraction <= 1
        assert 0 <= paraphrase_fraction + generate_fraction <= 1

        self.paraphrase_fraction = paraphrase_fraction
        self.generate_fraction = generate_fraction
        self.original_fraction = 1.0 - paraphrase_fraction - generate_fraction
        self.value_backend = value_backend
        self.version = dialog_tree.version

        # self.faq_keys = list(FAQQuestion.objects.filter(version=dialog_tree.version).values_list("key", flat=True))
        self.faq_keys = Data.objects[dialog_tree.version].faq_keys()
        
        self.start_node = Data.objects[self.version].node_by_key(dialog_tree.get_start_node().connected_node_key) # get first "real" dialog node (start node doesn't contain any information)
        self.answer_parser = AnswerTemplateParser()
        self.logic_parser = LogicTemplateParser()
        self.system_parser = SystemTemplateParser()

        # setup or load cache
        # TODO delete cache on save after serializer
        # TODO add code from google colab 
        if not (resource_dir / "translations.txt").exists():
            # TODO translate node texts ( and answer texts ?)
            pass
        if not (resource_dir / "generated.txt").exists():
            # TODO generate new questions
            pass
        if not (resource_dir / "paraphrased.txt").exists():
            # TODO paraphrase questions
            pass

    def _generate_questions(self):
        pass

    def _paraphrase_questions(self):
        pass

    def draw_goal(self) -> UserGoal:
        # TODO add paraphrasing / generation

        # for now, just draw random node with faq text as goal node
        # faq_candidate: FAQQuestion = FAQQuestion.objects.get(version=self.version, key=random.choice(self.faq_keys))
        faq_candidate: FAQQuestion = Data.objects[self.version].random_faq() 
        goal_node: DialogNode = Data.objects[self.version].node_by_key(faq_candidate.dialog_node_key)
        initial_user_utterance = faq_candidate.text

        # create a trajectory and variable values for reaching goal node
        return UserGoal(self.start_node, goal_node, faq_candidate.key, initial_user_utterance, self.answer_parser, self.logic_parser, self.system_parser, self.value_backend)


@dataclass
class DummyGoal:
    goal_idx: int
    goal_node: DialogNode
    faq_key: int
    initial_user_utterance: str
    variables: Dict[str, any]
    answer_pks: Dict[int, int]
    answer_parser: AnswerTemplateParser

    def has_reached_goal_node(self, node: DialogNode) -> bool:
        return node.key == self.goal_node.key
    
    def get_user_response(self, current_node: DialogNode) -> Union[UserResponse, None]:
        assert current_node.node_type == "userResponseNode"

        # return answer leading to correct branch
        return UserResponse(relevant=True, answer_key=self.answer_pks[current_node.key].answer_key)

    def get_user_input(self, current_node: DialogNode, bst: Dict[str, any]) -> UserInput:
        assert current_node.node_type == 'userInputNode'
        # assert current_node.answers.count() == 1
        assert current_node.answer_count() == 1
        # TODO add generated / paraphrased utterances?

        # get variable value for node
        # var_answer: DialogAnswer = current_node.answers.first() # should have exactly 1 answer
        var_answer: DialogAnswer = current_node.answers[0] # should have exactly 1 answer
        var = self.answer_parser.find_variable(var_answer.content.text)
        # if var.name in self.variables:
            # variable is relevant to user goal -> return it
        return UserInput(relevant=True, var_name=var.name, var_value=self.variables[var.name])
        # else:
        #     # variable is not relevant to user goal -> make one up and return it
        #     if var.name in bst:
        #         return UserInput(relevant=False, var_name=var.name, var_value=bst[var.name])
        #     return UserInput(relevant=False, var_name=var.name, var_value=VariableValue(var_name=var.name, var_type=var.type).draw_value())
   
