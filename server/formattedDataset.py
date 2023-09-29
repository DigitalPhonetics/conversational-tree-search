import json
import os
from typing import Dict, List
from data.dataset import DataAugmentationLevel, ReimburseGraphDataset, DialogNode, Question, NodeType, Answer

class FormattedReimburseGraphDataset(ReimburseGraphDataset):
    def _load_graph(self, resource_dir: str, graph_path: str, augmentation: DataAugmentationLevel, augmentation_path: str):
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
                    node.markup = dialognode_json['data']['markup']
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
