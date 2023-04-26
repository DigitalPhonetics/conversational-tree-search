from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from encoding.base import Encoding
from data.dataset import DialogNode, GraphDataset


"""
IDEA: encode node position in tree in a structured way instead of a random one-hot-encoding
https://papers.nips.cc/paper/2019/hash/6e0917469214d8fbd8c517dcdc6b8dcf-Abstract.html

Here, we encode non-regular trees:
* start node = 0
* each tree level can have a different node degree
    * but all nodes inside the same level are padded to the same degree
"""
class TreePositionEncoding(Encoding):
    def __init__(self, device: str, data: GraphDataset) -> None:
        super().__init__(device)
        self.data = data

        # build encoding
        self.node_mapping, self.encodings_length = self._process_node_tree()

    @torch.no_grad()
    def encode(self, dialog_node: DialogNode) -> torch.FloatTensor:
        """
        Returns:
            torch.FloatTensor (1 x encoding_dim)
        """
        return torch.tensor([[int(bit) for bit in self.node_mapping[dialog_node.key].zfill(self.encodings_length)]], dtype=torch.float, device=self.device)

    @torch.no_grad()
    def batch_encode(self, dialog_node: List[DialogNode], **kwargs) -> torch.FloatTensor:
        """
        Returns:
            torch.FloatTensor (batch x encoding_dim)
        """
        return torch.tensor([[int(bit) for bit in self.node_mapping[node.key].zfill(self.encodings_length)] for node in dialog_node], dtype=torch.float, device=self.device)

    def get_encoding_dim(self) -> int:
        return self.encodings_length

    def _get_max_node_degree_on_current_level(self, current_level_nodes: List[Tuple[str, DialogNode]]) -> int:
        """ Returns the maximum node degree in the current tree depth """
        max_degree = 0
        for path, node in current_level_nodes:
            node_degree = len(node.answers)
            if node_degree > max_degree:
                max_degree = node_degree
        return max_degree

    def _process_node_tree(self) -> Tuple[Dict[str, str], int]:
        print("Building tree embedding for nodes...")
        current_level_nodes: List[Tuple[str, DialogNode]] = [("", self.data.start_node)] # load start node
        encodings_map = {current_level_nodes[0][1].key: ""}
        visited_node_ids = set() # prevent loops
        
        encoding_length = 0
        while len(current_level_nodes) > 0:
            # process current tree level
            next_level_nodes = []
            max_node_degree = self._get_max_node_degree_on_current_level(current_level_nodes)
            encoding_length += max_node_degree
            for node_path, node in current_level_nodes:
                if node.key in visited_node_ids:
                    continue # break loops
                visited_node_ids.add(node.key)
                for answer in node.answers:
                    if answer.connected_node:
                        # append binary node answer index to path leading to current node, padded to max level degree
                        new_path = node_path + f"{answer.index:b}".zfill(max_node_degree)
                        encodings_map[answer.connected_node.key] = new_path
                        next_level_nodes.append((new_path, answer.connected_node))
                if node.connected_node:
                    new_path = node_path + "1".zfill(max_node_degree) # if no answers but connected node -> answer index = 1
                    encodings_map[node.connected_node.key] = new_path
                    next_level_nodes.append((new_path, node.connected_node))
            current_level_nodes = next_level_nodes

        print("Done")
        return encodings_map, encoding_length




class AnswerPositionEncoding(Encoding):
    """ one hot encoding for answer position inside node """
    def __init__(self, device: str, data: GraphDataset) -> None:
        super().__init__(device=device)
        # get maximum node degree
        self.max_degree = data.get_max_node_degree()
    
    def get_encoding_dim(self) -> int:
        return self.max_degree
    
    @torch.no_grad()
    def encode(self, dialog_node: DialogNode):
        """
        Returns:
            # answers(dialog_node) x max_node_degree
        """
        num_answers = len(dialog_node.answers)
        if num_answers > 0: 
            # node with answers
            return F.one_hot(torch.tensor(list(range(num_answers)), dtype=torch.long, device=self.device), num_classes=self.max_degree)
        elif dialog_node.connected_node: 
            # node without answers, but directly connected neighbour
            return F.one_hot(torch.tensor([0], dtype=torch.long, device=self.device), num_classes=self.max_degree)
        else:
            # node without answers and neighbours
            return torch.zeros((1, self.max_degree), dtype=torch.float, device=self.device)

    @torch.no_grad()
    def batch_encode(self, dialog_node: List[DialogNode], **kwargs) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        """
        Returns:
            # batch x max_answers x max_node_degree, batch (num_answers)
        """
        # return pad_sequence([self._encode(node) for node in dialog_node], batch_first=True), torch.tensor([node.answers.count() for node in dialog_node], device=self.device)
        return pad_sequence([self._encode(node) for node in dialog_node], batch_first=True), torch.tensor([len(node.answers) for node in dialog_node], device=self.device)
       