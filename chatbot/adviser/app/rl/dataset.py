from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from functools import reduce
import json
import random
from typing import Dict, List

@dataclass
class Content:
    text: str

@dataclass
class DialogAnswer:
    key: str
    answer_index: int
    connected_node_key: str
    content: Content
    version: int

@dataclass 
class FAQQuestion:
    key: str
    text: str
    dialog_node_key: str
    version: int

@dataclass
class DialogNode:
    key: str
    content: Content
    node_type: str
    answers: List[DialogAnswer]
    faq_questions: List[FAQQuestion]
    connected_node_key: str
    version: int

    def random_faq(self) -> FAQQuestion:
        return random.choice(self.faq_questions)

    def faq_questions_count(self) -> int:
        return len(self.faq_questions)

    def answer_count(self) -> int:
        return len(self.answers)

    def random_answer(self) -> DialogAnswer:
        return random.choice(self.answers)

    def answer_by_index(self, index) -> DialogAnswer:
        for answer in self.answers:
            if answer.answer_index == index:
                return answer
        raise ValueError

    def answer_by_key(self, key: str) -> DialogAnswer:
        for answer in self.answers:
            if answer.key == key:
                return answer
        raise ValueError

    def answer_by_goalnode_key(self, key) -> DialogAnswer:
        for answer in self.answers:
            if answer.connected_node_key and answer.connected_node_key == key:
                return answer
        raise ValueError

@dataclass
class Tagegeld:
    land: str
    stadt: str
    tagegeldsatz: float

def _preprocess_table(filename: str) -> Dict[str, Dict[str, Tagegeld]]: # land, stadt
    import pandas as pd
    new_rows = defaultdict(lambda: dict())
    content = pd.read_excel(filename)
    for idx, row in content.iterrows():
        # TODO can we make this parser more generic? contains instead of equality, ignore case, ...
        land = row['Land']
        stadt = row['Stadt']
        tagegeld = row['Tagegeld LRKG']
        new_rows[land][stadt] = Tagegeld(land=land, stadt=stadt, tagegeldsatz=tagegeld)
    return new_rows


def html_to_raw_text(html: str):
		""" Convert string with html tags to unformatted raw text (removing all tags) """
		return html.replace('&nbsp;', ' ') \
				.replace("&Auml;", 'Ä') \
				.replace('&auml;', 'ä') \
				.replace('&Ouml;', 'Ö') \
				.replace('&ouml;', 'ö') \
				.replace('&Uuml;', 'Ü') \
				.replace('&uuml;', 'ü') \
				.replace('&szlig;', 'ß') \
				.replace('&euro;', '€') \
				.replace('&bdquo;', '"') \
				.replace('&ldquo;', '"') \
				.replace('&sbquo;', "'") \
				.replace('&lsquo;', "'") \
				.replace('\n', "")


class NodeType(Enum):
    INFO = "infoNode"
    VARIABLE = "userInputNode"
    QUESTION = "userResponseNode"
    LOGIC = "logicNode"


class Dataset:
    def __init__(self, nodes_by_key, nodes_by_type, answers_by_key, faq_by_key, faq_list, start_node, node_list, tagegeld_by_land) -> None:
        self._node_by_key = nodes_by_key
        self._node_by_type = nodes_by_type
        self._answer_by_key = answers_by_key
        self._faq_by_key = faq_by_key
        self._faq_list = faq_list
        self._start_node = start_node
        self._node_list = node_list
        self._tagegeld_by_land = tagegeld_by_land
        self._country_list = set(self._tagegeld_by_land.keys())
        self._city_list = reduce(lambda l1, l2: set(l1).union(l2), [list(cities.keys()) for cities in tagegeld_by_land.values()])
        self._num_faq_nodes = sum([1 for node in node_list if len(node.faq_questions) > 0])
        self._num_guided_goal_nodes = sum([1 for node in node_list if (node.node_type in [NodeType.QUESTION.value, NodeType.VARIABLE.value] and node.answer_count() > 0) or (node.node_type == NodeType.INFO.value)])

    def count_faqs(self) -> int:
        return len(self._faq_by_key)

    def count_nodes(self) -> int:
        return len(self._node_by_key)

    def count_faq_nodes(self) -> int:
        return self._num_faq_nodes

    def count_guided_goal_node_candidates(self) -> int:
        return self._num_guided_goal_nodes

    def count_countries(self) -> int:
        return len(self._country_list)

    def count_cities(self) -> int:
        return len(self._city_list)

    def faq_list(self) -> List[FAQQuestion]:
        return self._faq_list

    def random_faq(self) -> FAQQuestion:
        return random.choice(self._faq_list)

    def faq_by_key(self, key: str) -> FAQQuestion:
        assert key in self._faq_by_key, f"Expected {key} to be in FAQ list"
        return self._faq_by_key[key]

    def faq_keys(self) -> List[str]:
        return list(self._faq_by_key.keys())

    def node_by_key(self, key: str) -> DialogNode:
        assert key in self._node_by_key, f"Expected {key} to be in Node list"
        return self._node_by_key[key]

    def nodes_by_type(self, node_type: str) -> List[DialogNode]:
        return self._node_by_type[node_type]

    def node_types(self) -> List[str]:
        return list(self._node_by_type.keys())

    def nodes(self) -> List[DialogNode]:
        return self._node_list
    
    def start_node(self) -> DialogNode:
        return self._start_node

    def answer_by_key(self, key: str) -> DialogAnswer:
        assert key in self._answer_by_key, f"Expected {key} to be in Answer List"
        return self._answer_by_key[key]

    def tagegeld(self, land: str, stadt: str) -> float:
        return self._tagegeld_by_land[land][stadt].tagegeldsatz

    @classmethod
    def fromJSON(cls, json_str: str, version: int):
        with open(json_str) as f:
            data = json.load(f)

        nodes_by_key = {}
        nodes_by_type = {}
        node_list = []
        answers_by_key = {}
        questions_by_key = {}
        question_list = []
        start_node = None
        
        for dialognode_json in data['nodes']:
            # parse node info (have to create all nodes before we can create the answers because they can be linked to otherwise not yet existing nodes)
            content_markup = dialognode_json['data']['raw_text']
            content_text = html_to_raw_text(content_markup)
            
            node = DialogNode(version=version, key=int(dialognode_json['id']), content=Content(content_text), node_type=dialognode_json['type'], answers=[], faq_questions=[], connected_node_key=None)
            assert not node.key in nodes_by_key, f"Node {node.key} already in dataset"
            nodes_by_key[node.key] = node
            if not node.node_type in nodes_by_type:
                nodes_by_type[node.node_type] = []
            nodes_by_type[node.node_type].append(node)
            node_list.append(node)
            if node.node_type == 'startNode':
                start_node = node

            for index, answer_json in enumerate(dialognode_json['data']['answers']):
                # parse answer info and add to created nodes
                answer_markup = answer_json['raw_text']
                answer_text = html_to_raw_text(answer_markup)

                answer = DialogAnswer(version=version, key=int(answer_json['id']), content=Content(answer_text), answer_index=index, connected_node_key=None) # store answers in correct order
                node.answers.append(answer)
                answers_by_key[answer.key] = answer
            
            for faq_json in dialognode_json['data']['questions']:
                question = FAQQuestion(version=version, key=int(faq_json['id']),text=faq_json['text'], dialog_node_key=node.key)
                assert not question.key in questions_by_key, f"Question {question.key} already in dataset"
                questions_by_key[question.key] = question
                node.faq_questions.append(question)
                question_list.append(question)
        
        # parse connections
        for connection in data['connections']:
            fromDialogNode = nodes_by_key[int(connection['source'])]
            if fromDialogNode.node_type == 'startNode' or fromDialogNode.node_type == 'infoNode':
                fromDialogNode.connected_node_key = int(connection['target'])
            else:
                fromDialogAnswer = answers_by_key[int(connection['sourceHandle'])]
                fromDialogAnswer.connected_node_key = int(connection['target'])

        # sort answers
        for key in nodes_by_key:
            node = nodes_by_key[key]
            node.answers.sort(key=lambda ans: ans.answer_index)

        # load tagegeld
        tagegeld = _preprocess_table("TAGEGELD_AUSLAND.xlsx")

        assert start_node
        return cls(nodes_by_key, nodes_by_type, answers_by_key, questions_by_key, question_list, start_node, node_list, tagegeld)

objects: Dict[int, Dataset] = {
    0: None,
    1: None
}