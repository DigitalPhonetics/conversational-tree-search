
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
import json
import os
import random
from typing import Dict, List, Set, Tuple, Optional

import pandas as pd
import torch 


class NodeType(Enum):
    INFO = "infoNode"
    VARIABLE = "userInputNode"
    VARIABLE_UPDATE = 'variableUpdateNode'
    QUESTION = "userResponseNode"
    LOGIC = "logicNode"
    START = "startNode"

    def __lt__(self, other):
        return self.value < other.value


class DataAugmentationLevel(Enum):
    NONE = 'none'
    MIXED = 'mixed'
    ARTIFICIAL_ONLY = 'artificial'
    

@dataclass
class DatasetConfig:
    _target_: str
    graph_path: str
    answer_path: str
    use_answer_synonyms: bool
    augmentation: DataAugmentationLevel
    augmentation_path: Optional[str] = None
    question_limit: Optional[int] = 0
    answer_limit: Optional[int] = 0

@dataclass
class Answer:
    key: str
    index: int
    connected_node: "DialogNode"
    text: str
    parent: "DialogNode"

@dataclass
class Question:
    key: str
    text: str
    parent: "DialogNode"
    
@dataclass
class DialogNode:
    key: str
    text: str
    node_type: NodeType
    answers: List[Answer]
    questions: List[Question]
    connected_node: "DialogNode"

    def random_question(self) -> Question:
        return random.choice(self.questions)

    def random_answer(self) -> Answer:
        return random.choice(self.answers)

    def answer_by_index(self, index: int) -> Answer:
        return next(ans for ans in self.answers if ans.index == index)

    def answer_by_key(self, key: str) -> Answer:
        return next(ans for ans in self.answers if ans.key == key)

    def answer_by_connected_node(self, connected_node: "DialogNode") -> Answer:
        return next(ans for ans in self.answers if ans.connected_node.key == connected_node.key)

    def __str__(self) -> str:
        return f"""DialogNode.{self.node_type.name}(key: {self.key}, answers: {len(self.answers)}, questions: {len(self.questions)})
        - connected_node: {self.connected_node.key if self.connected_node else None}
        - text: {self.text[:100]}
        """

    def __repr__(self) -> str:
        return f"""DialogNode.{self.node_type.name}(key: {self.key}, answers: {len(self.answers)}, questions: {len(self.questions)})
        - connected_node: {self.connected_node.key if self.connected_node else None}
        - text: {self.text[:100]}
        """
  

@dataclass
class Tagegeld:
    country: str
    city: str
    daily_allowance: float

# TODO load a1 countries
# TODO load uebernachtungsgeld 

class GraphDataset:
    def __init__(self, graph_path: str, answer_path: str, use_answer_synonyms: bool, augmentation: DataAugmentationLevel, augmentation_path: str = None, resource_dir: str = "resources/",
                 question_limit: int = 0, answer_limit: int = 0) -> None:
        assert isinstance(augmentation, DataAugmentationLevel), f"found {augmentation}"
        self.resource_dir = resource_dir
        self.graph = self._load_graph(resource_dir, graph_path, augmentation, augmentation_path, question_limit)
        self.answer_synonyms = self._load_answer_synonyms(os.path.join(resource_dir, answer_path), use_answer_synonyms, augmentation, answer_limit)

        self.num_guided_goal_nodes = sum([1 for node in self.node_list if (node.node_type in [NodeType.INFO, NodeType.QUESTION, NodeType.VARIABLE] and len(node.answers) > 0) or (node.node_type == NodeType.INFO)])
        self.num_free_goal_nodes = sum([1 for node in self.node_list if len(node.questions) > 0])
        self.num_answer_synonyms = sum([len(self.answer_synonyms[answer]) for answer in self.answer_synonyms])
        self._max_tree_depth = None
        self._max_node_degree = None

        self.action_masks = self._calculate_action_masks()

        print("===== Dataset Statistics =====")
        print("- files: ", graph_path, answer_path)
        print("- synonyms:", use_answer_synonyms)
        print("- depth:", self.get_max_tree_depth(), " - degree:", self.get_max_node_degree())
        print("- answers:", sum([len(self.answer_synonyms[key]) for key in self.answer_synonyms]))
        print("- questions:", len(self.question_list))
        print("- loaded original data:", self._should_load_original_data(augmentation))
        print("- loaded generated data:", self._should_load_generated_data(augmentation))
        print("- question limit:", question_limit, " - maximum loaded: ", max([len(node.questions) for node in self.node_list]))
        print("- answer limit:", answer_limit, " - maximum loaded: ", max([len(self.answer_synonyms[answer]) for answer in self.answer_synonyms]))

    def _should_load_original_data(self, augmentation: DataAugmentationLevel) -> bool:
        return augmentation in [DataAugmentationLevel.NONE, DataAugmentationLevel.MIXED]
    
    def _should_load_generated_data(self, augmentation: DataAugmentationLevel) -> bool:
        return augmentation in [DataAugmentationLevel.ARTIFICIAL_ONLY, DataAugmentationLevel.MIXED]

    def _load_graph(self, resource_dir: str, graph_path: str, augmentation: DataAugmentationLevel, augmentation_path: str, question_limit: int):
        # load graph
        with open(os.path.join(resource_dir, graph_path), "r") as f:
            data = json.load(f)

            self.nodes_by_key: Dict[str, DialogNode] = {}
            self.nodes_by_type: Dict[NodeType, List[DialogNode]] = {}
            self.node_list: List[DialogNode] = []
            self.answers_by_key: Dict[str, Answer] = {}
            self.questions_by_key: Dict[str, Question] = {}
            self.question_list: List[Question] = []
            self.start_node: DialogNode = None

            for dialognode_json in data['nodes']:
                # parse node info (have to create all nodes before we can create the answers because they can be linked to otherwise not yet existing nodes)
                node = DialogNode(key=int(dialognode_json['id']),
                                text=dialognode_json['data']['raw_text'],
                                node_type=NodeType(dialognode_json['type']),
                                answers=[],
                                questions=[],
                                connected_node=None)
                assert not node.key in self.nodes_by_key, f"Node {node.key} already in dataset"
                self.nodes_by_key[node.key] = node
                if not node.node_type in self.nodes_by_type:
                    self.nodes_by_type[node.node_type] = []
                self.nodes_by_type[node.node_type].append(node)
                self.node_list.append(node)
                if node.node_type == NodeType.START:
                    self.start_node = node

                for index, answer_json in enumerate(dialognode_json['data']['answers']):
                    # parse answer info and add to created nodes
                    answer = Answer(
                        key=int(answer_json['id']), 
                        text=answer_json['raw_text'],
                        index=index,
                        connected_node=None,
                        parent=node) # store answers in correct order
                    node.answers.append(answer)
                    self.answers_by_key[answer.key] = answer
                # sort answers
                node.answers.sort(key=lambda ans: ans.index)
                
                if self._should_load_original_data(augmentation):
                    for faq_json in dialognode_json['data']['questions']:
                        question = Question(key=int(faq_json['id']),
                                        text=faq_json['text'],
                                        parent=node)
                        assert not question.key in self.questions_by_key, f"Question {question.key} already in dataset"
                        if question_limit == 0 or (question_limit > 0 and len(node.questions) < question_limit):
                            self.questions_by_key[question.key] = question
                            node.questions.append(question)
                            self.question_list.append(question)
            
            # data augmentation
            if self._should_load_generated_data(augmentation):
                generated_path = os.path.join(resource_dir, augmentation_path)
                with open(generated_path, "r") as f:
                    print("- Loading questions from ", generated_path)
                    augmented_questions = json.load(f)
                    for augmented_question_key in augmented_questions:
                        parent = self.nodes_by_key[int(augmented_questions[augmented_question_key]['dialog_node_key'])]
                        question = Question(key=int(augmented_question_key),
                                            text=augmented_questions[augmented_question_key]['text'],
                                            parent=parent)
                        assert not question.key in self.questions_by_key, f"Question {question.key} already in dataset" 
                        if question_limit == 0 or (question_limit > 0 and len(parent.questions) < question_limit):
                            self.questions_by_key[question.key] = question
                            parent.questions.append(question)
                            self.question_list.append(question)
            
            # parse connections
            for connection in data['connections']:
                fromDialogNode = self.nodes_by_key[int(connection['source'])]
                if fromDialogNode.node_type in [NodeType.START, NodeType.INFO, NodeType.VARIABLE_UPDATE]:
                    fromDialogNode.connected_node = self.nodes_by_key[int(connection['target'])]
                else:
                    fromDialogAnswer = self.answers_by_key[int(connection['sourceHandle'])]
                    fromDialogAnswer.connected_node = self.nodes_by_key[int(connection['target'])]

    def _load_answer_synonyms(self, answer_path: str, use_answer_synonyms: bool, augmentation: DataAugmentationLevel, answer_limit: int):
        # load synonyms
        with open(answer_path, "r") as f:
            answers = json.load(f)
            answer_data = {answer.lower(): answers[answer] for answer in answers}
            if not use_answer_synonyms or augmentation == DataAugmentationLevel.ARTIFICIAL_ONLY:
                if not use_answer_synonyms:
                    print("- not using synonyms")
                else:
                    print("- only artificial answers")
                # key is also the only possible value
                answer_data = {answer.lower(): [answer] for answer in answer_data}
        if use_answer_synonyms and self._should_load_generated_data(augmentation):
            augmentation_path = f"{os.path.dirname(answer_path)}/generated/train_answers.json"
            with open(augmentation_path, "r") as f:
                print(f"Loading augmentation answers from {augmentation_path}")
                generated_answers = json.load(f)
                for key in generated_answers:
                    for syn in generated_answers[key]:
                        if answer_limit == 0 or (answer_limit > 0 and len(answer_data[key]) < answer_limit):
                            answer_data[key].append(syn)
        return answer_data

    def _calculate_action_masks(self) -> Dict[int, torch.IntTensor]:
        """ Pre-calculate action mask per node s.t. it is usable without database lookups """
        masks = {}
        for node in self.node_list:
            mask = torch.zeros(self.get_max_node_degree() + 1, dtype=torch.int8) # add ASK action
            if node.node_type == NodeType.QUESTION:
                skip_answer_count = len(node.answers)
                if skip_answer_count > 0:
                    mask[:skip_answer_count+1] = 1 # no invalid SKIP actions (0: ASK, 1,...N: SKIP)
                elif skip_answer_count == 0:
                    mask[0] = 1 # ASK, no skips because we have no answers
            elif node.node_type in [NodeType.INFO, NodeType.VARIABLE]:
                mask[:2] = 1 # no invalid SKIP actions (ASK and only 1 default SKIP possible)
            else:
                # don't care about startNode and logicNode, they are automatically processed without agent
                continue
            masks[node.key] = mask
        return masks

    def _get_max_tree_depth(self) -> int:
        """ Return maximum tree depth (max. number of steps to leave node) in whole graph """
        level = 0
        current_level_nodes = [self.start_node.connected_node]
        visited_node_ids = set() # cycle breaking
        while len(current_level_nodes) > 0:
            # traverse current node level, append all children to next level nodes
            next_level_nodes = []
            for current_node in current_level_nodes:
                if current_node.key in visited_node_ids:
                    continue
                visited_node_ids.add(current_node.key)

                # add children of node to next level nodes
                if current_node.connected_node:
                    next_level_nodes.append(current_node.connected_node)
                    assert len(current_node.answers) == 0
                elif len(current_node.answers) > 0:
                    next_level_nodes += [answer.connected_node for answer in current_node.answers]
            # continue with next level breadth search
            current_level_nodes = next_level_nodes
            level += 1
        return level

    def get_max_tree_depth(self) -> int:
        """ Return maximum tree depth (max. number of steps to leave node) in whole graph (cached) """
        if not self._max_tree_depth:
            # calculate, then cache value
            self._max_tree_depth = self._get_max_tree_depth()
        return self._max_tree_depth

    def _get_max_node_degree(self) -> int:
        """ Return highest node degree in whole graph """
        max_degree = 0
        # for node in DialogNode.objects.filter(version=self.version):
        for node in self.node_list:
            # answer_count = node.answers.count()
            answer_count = len(node.answers)
            if answer_count > max_degree:
                max_degree = answer_count
        return max_degree

    def get_max_node_degree(self) -> int:
        """ Return highest node degree in whole graph (cached) """
        if not self._max_node_degree:
            # calculate, then cache value
            self._max_node_degree = self._get_max_node_degree()
        return self._max_node_degree

    def count_question_nodes(self) -> int:
        return len(self.nodes_by_type[NodeType.QUESTION])

    def random_question(self) -> Question:
        return random.choice(self.question_list)


class ReimburseGraphDataset(GraphDataset):
    def __init__(self, graph_path: str, answer_path: str, use_answer_synonyms: bool, augmentation: DataAugmentationLevel, augmentation_path: str = None, resource_dir: str = "resources/", question_limit: int = 0, answer_limit: int = 0) -> None:
        assert isinstance(augmentation, DataAugmentationLevel), f"found {augmentation}"
        super().__init__(graph_path=graph_path, answer_path=answer_path, use_answer_synonyms=use_answer_synonyms, augmentation=augmentation, augmentation_path=augmentation_path, resource_dir=resource_dir, question_limit=question_limit, answer_limit=answer_limit)
        self.a1_countries = self._load_a1_countries(resource_dir)
        self.hotel_costs, self.country_list, self.city_list = self._load_hotel_costs(resource_dir)
        self._load_country_synonyms(resource_dir)
        self._load_city_synonyms(resource_dir)


    def _load_a1_countries(self, resource_dir: str):
        with open(os.path.join(resource_dir, "en/reimburse/a1_countries.json"), "r") as f:
            a1_countries = json.load(f)
        return a1_countries

    def _load_hotel_costs(self, resource_dir: str) -> Tuple[Dict[str, Dict[str, float]], Set[str], Set[str]]:
        """
        Returns:
            hotel_costs: country -> city -> value
            country_list: Set[str]            
            city_list: Set[str]
        """
        # load max. hotel costs
        hotel_costs = defaultdict(lambda: dict())
        country_list = set()
        city_list = set()

        content = pd.read_excel(os.path.join(resource_dir, "en/reimburse/TAGEGELD_AUSLAND.xlsx"))
        for idx, row in content.iterrows():
            country = row['Land']
            city = row['Stadt']
            country_list.add(country)
            city_list.add(city)
            daily_allowance = row['Tagegeld LRKG']
            hotel_costs[country][city] = Tagegeld(country=country, city=city, daily_allowance=daily_allowance)
        return hotel_costs, country_list, city_list
    
    def _load_country_synonyms(self, resource_dir: str):
        with open(os.path.join(resource_dir, 'en/reimburse/country_synonyms.json'), 'r') as f:
            country_synonyms = json.load(f)
            self.country_keys = [country.lower() for country in country_synonyms.keys()]
            self.countries = {country.lower(): country for country in country_synonyms.keys()}
            self.countries.update({country_syn.lower(): country for country, country_syns in country_synonyms.items()
                                    for country_syn in country_syns})
    
    def _load_city_synonyms(self, resource_dir: str):
        with open(os.path.join(resource_dir, 'en/reimburse/city_synonyms.json'), 'r') as f:
            city_synonyms = json.load(f)
            self.city_keys = [city.lower() for city in city_synonyms.keys()]
            self.cities = {city.lower(): city for city in city_synonyms.keys() if city != '$REST'}
            self.cities.update({city_syn.lower(): city for city, city_syns in city_synonyms.items()
                                for city_syn in city_syns})



class OnboardingGraphDataset(GraphDataset):
    def __init__(self, graph_path: str, answer_path: str, use_answer_synonyms: bool, augmentation: DataAugmentationLevel, augmentation_path: str = None, resource_dir: str = "resources/", question_limit: int = 0, answer_limit: int = 0) -> None:
        assert isinstance(augmentation, DataAugmentationLevel), f"found {augmentation}"
        super().__init__(graph_path=graph_path, answer_path=answer_path, use_answer_synonyms=use_answer_synonyms, augmentation=augmentation, augmentation_path=augmentation_path, resource_dir=resource_dir, question_limit=question_limit, answer_limit=answer_limit)

    def _load_answer_synonyms(self, answer_path: str, use_answer_synonyms: bool, augmentation: DataAugmentationLevel, answer_limit: int):
        # we don't have synonyms here - extract answers from graph file instead (answer_path == train graph!)
        answer_data = {answer.text.lower(): [answer.text] for answer in self.answers_by_key.values()} # key is also the only possible value
        if not use_answer_synonyms or augmentation == DataAugmentationLevel.ARTIFICIAL_ONLY:
            if not use_answer_synonyms:
                print("- not using synonyms")
            else:
                print("- only artificial answers")
        if use_answer_synonyms and self._should_load_generated_data(augmentation):
            augmentation_path = f"{os.path.dirname(answer_path)}/generated/train_answers.json"
            with open(augmentation_path, "r") as f:
                print(f"Loading augmentation answers from {augmentation_path}")
                generated_answers = json.load(f)
                for key in generated_answers:
                    for syn in generated_answers[key]:
                        if answer_limit == 0 or (answer_limit > 0 and len(answer_data[key]) < answer_limit):
                            answer_data[key].append(syn)
        return answer_data

    def _load_a1_countries(self, resource_dir: str):
        with open(os.path.join(resource_dir, "en/a1_countries.json"), "r") as f:
            a1_countries = json.load(f)
        return a1_countries

    def _load_hotel_costs(self, resource_dir: str) -> Tuple[Dict[str, Dict[str, float]], Set[str], Set[str]]:
        """
        Returns:
            hotel_costs: country -> city -> value
            country_list: Set[str]            
            city_list: Set[str]
        """
        # load max. hotel costs
        hotel_costs = defaultdict(lambda: dict())
        country_list = set()
        city_list = set()

        content = pd.read_excel(os.path.join(resource_dir, "en/TAGEGELD_AUSLAND.xlsx"))
        for idx, row in content.iterrows():
            country = row['Land']
            city = row['Stadt']
            country_list.add(country)
            city_list.add(city)
            daily_allowance = row['Tagegeld LRKG']
            hotel_costs[country][city] = Tagegeld(country=country, city=city, daily_allowance=daily_allowance)
        return hotel_costs, country_list, city_list
    
    def _load_country_synonyms(self, resource_dir: str):
        with open(os.path.join(resource_dir, 'en/country_synonyms.json'), 'r') as f:
            country_synonyms = json.load(f)
            self.country_keys = [country.lower() for country in country_synonyms.keys()]
            self.countries = {country.lower(): country for country in country_synonyms.keys()}
            self.countries.update({country_syn.lower(): country for country, country_syns in country_synonyms.items()
                                    for country_syn in country_syns})
    
    def _load_city_synonyms(self, resource_dir: str):
        with open(os.path.join(resource_dir, 'en/city_synonyms.json'), 'r') as f:
            city_synonyms = json.load(f)
            self.city_keys = [city.lower() for city in city_synonyms.keys()]
            self.cities = {city.lower(): city for city in city_synonyms.keys() if city != '$REST'}
            self.cities.update({city_syn.lower(): city for city, city_syns in city_synonyms.items()
                                for city_syn in city_syns})
