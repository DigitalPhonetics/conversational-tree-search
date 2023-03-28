
from collections import defaultdict
from dataclasses import dataclass
import json
import random
from typing import List, Set

import pandas as pd

@dataclass
class DatasetConfig:
    _target_: str
    graph_path: str
    answer_path: str
    use_answer_synonyms: bool

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
    node_type: str
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

@dataclass
class Tagegeld:
    land: str
    stadt: str
    tagegeldsatz: float

# TODO load a1 countries
# TODO load uebernachtungsgeld 

class GraphDataset:
    def __init__(self, graph_path: str, answer_path: str, use_answer_synonyms: bool) -> None:
        print("Dataset: ", graph_path, answer_path, use_answer_synonyms)

        self.graph = self._load_graph()
        self.answer_synonyms = self._load_answer_synonyms(answer_path, use_answer_synonyms)
        self.a1_countries = self._load_a1_countries()
        self.hotel_costs = self._load_hotel_costs()

        self._max_tree_depth = None
        self._max_node_degree = None
      
    def _load_graph(self, graph_path: str):
        # load graph
        with open(graph_path, "r") as f:
            data = json.load(f)

            self.nodes_by_key = {}
            self.nodes_by_type = {}
            self.node_list = []
            self.answers_by_key = {}
            self.questions_by_key = {}
            self.question_list = []
            self.start_node = None

            for dialognode_json in data['nodes']:
                # parse node info (have to create all nodes before we can create the answers because they can be linked to otherwise not yet existing nodes)
                node = DialogNode(key=int(dialognode_json['id']),
                                text=dialognode_json['data']['raw_text'],
                                node_type=dialognode_json['type'],
                                answers=[],
                                questions=[],
                                connected_node=None)
                assert not node.key in self.nodes_by_key, f"Node {node.key} already in dataset"
                self.nodes_by_key[node.key] = node
                if not node.node_type in self.nodes_by_type:
                    self.nodes_by_type[node.node_type] = []
                self.nodes_by_type[node.node_type].append(node)
                self.node_list.append(node)
                if node.node_type == 'startNode':
                    self.start_node = node

                for index, answer_json in enumerate(dialognode_json['data']['answers']):
                    # parse answer info and add to created nodes
                    answer = Answer(
                        key=int(answer_json['id']), 
                        text=answer_json['raw_text'],
                        answer_index=index,
                        connected_node=None,
                        parent=node) # store answers in correct order
                    node.answers.append(answer)
                    self.answers_by_key[answer.key] = answer
                # sort answers
                node.answers.sort(key=lambda ans: ans.answer_index)
                
                for faq_json in dialognode_json['data']['questions']:
                    question = Question(key=int(faq_json['id']),
                                    text=faq_json['text'],
                                    parent=node)
                    assert not question.key in self.questions_by_key, f"Question {question.key} already in dataset"
                    self.questions_by_key[question.key] = question
                    node.questions.append(question)
                    self.question_list.append(question)
            
            # parse connections
            for connection in data['connections']:
                fromDialogNode = self.nodes_by_key[int(connection['source'])]
                if fromDialogNode.node_type == 'startNode' or fromDialogNode.node_type == 'infoNode':
                    fromDialogNode.connected_node = self.nodes_by_key[int(connection['target'])]
                else:
                    fromDialogAnswer = self.answers_by_key[int(connection['sourceHandle'])]
                    fromDialogAnswer.connected_node = self.nodes_by_key[int(connection['target'])]

    def _load_answer_synonyms(self, answer_path: str, use_answer_synonyms: bool):
        # load synonyms
        with open(answer_path, "r") as f:
            answer_data = json.load(f)
            if not use_answer_synonyms:
                print("- not using synonyms")
                # key is also the only possible value
                answer_data = {answer: [answer] for answer in answer_data}
        return answer_data

    def _load_a1_countries(self):
        with open("resources/a1_countries.json", "r") as f:
            a1_countries = json.load(f)
        return a1_countries

    def _load_hotel_costs(self):
        # load max. hotel costs
        hotel_costs = defaultdict(lambda: dict())
        content = pd.read_excel("resources/TAGEGELD_AUSLAND.xlsx")
        for idx, row in content.iterrows():
            land = row['Land']
            stadt = row['Stadt']
            tagegeld = row['Tagegeld LRKG']
            hotel_costs[land][stadt] = Tagegeld(land=land, stadt=stadt, tagegeldsatz=tagegeld)
        return hotel_costs
    
    def _get_max_tree_depth(self, current_node: DialogNode, current_max_depth: int, visited: Set[int]) -> int:
        """ Return maximum tree depth (max. number of steps to leave node) in whole graph """

        if current_node.key in visited:
            return current_max_depth
        visited.add(current_node.key)

        if current_node.node_type == 'startNode':
            # begin recursion at start node
            current_node = current_node.connected_node

        # if current_node.answers.count() > 0:
        if len(current_node.answers) > 0:
            # normal node
            # continue recursion by visiting children
            max_child_depth = max([self._get_max_tree_depth(answer.connected_node, current_max_depth + 1, visited) for answer in current_node.answers if answer.connected_node])
            return max_child_depth
        elif current_node.connected_node_key:
            # node without answers, e.g. info node
            # continue recursion by visiting children
            return self._get_max_tree_depth(current_node.connected_node, current_max_depth + 1, visited)
        else:
            # reached leaf node
            return current_max_depth

    def get_max_tree_depth(self) -> int:
        """ Return maximum tree depth (max. number of steps to leave node) in whole graph (cached) """
        if not self._max_tree_depth:
            # calculate, then cache value
            self._max_tree_depth = self._get_max_tree_depth(current_node=self.start_node, current_max_depth=0, visited=set([]))
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