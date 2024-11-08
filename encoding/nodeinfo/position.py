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

        # build encoding
        self.node_mapping, self.encodings_length = self._process_node_tree(data)

    @torch.no_grad()
    def encode(self, dialog_node: DialogNode) -> torch.FloatTensor:
        """
        Returns:
            torch.FloatTensor (1 x encoding_dim)
        """
        return torch.tensor([self.node_mapping[dialog_node.key]], dtype=torch.float)

    @torch.no_grad()
    def batch_encode(self, dialog_node: List[DialogNode], **kwargs) -> torch.FloatTensor:
        """
        Returns:
            torch.FloatTensor (batch x encoding_dim)
        """
        return torch.tensor([self.node_mapping[node.key] for node in dialog_node], dtype=torch.float)

    def get_encoding_dim(self) -> int:
        return self.encodings_length

    def _get_max_node_degree_on_current_level(self, current_level_nodes: List[Tuple[str, DialogNode]]) -> int:
        """ Returns the maximum node degree in the current tree depth """
        max_degree = 0
        for path, node in current_level_nodes:
            # node degree should still be 1 even if there are no answers, but directly connected nodes
            node_degree = 1 if node.connected_node else len(node.answers) 
            if node_degree > max_degree:
                max_degree = node_degree
        return max_degree

    def _process_node_tree(self, data: GraphDataset) -> Tuple[Dict[str, str], int]:
        print("Building tree embedding for nodes...")
        current_level_nodes: List[Tuple[str, DialogNode]] = [("", data.start_node.connected_node)] # load start node
        encodings_map = {current_level_nodes[0][1].key: ""}
        visited_node_ids = set() # prevent loops
        
        encoding_length = 0
        while len(current_level_nodes) > 0:
            assert encodings_map[data.start_node.connected_node.key] == ''
            # process current tree level
            next_level_nodes = []
            max_node_degree = self._get_max_node_degree_on_current_level(current_level_nodes)
            encoding_length += max_node_degree
            for node_path, node in current_level_nodes:
                if node.key in visited_node_ids:
                    continue # break loops
                visited_node_ids.add(node.key)
                
                if node.connected_node and not node.connected_node.key in visited_node_ids:
                    new_path = "1".rjust(max_node_degree, "0") + node_path # if no answers but connected node -> answer index = 1
                    encodings_map[node.connected_node.key] = new_path
                    next_level_nodes.append((new_path, node.connected_node))
                else:
                    for answer in node.answers:
                        if answer.connected_node and not answer.connected_node.key in visited_node_ids:
                            # append binary node answer index to path leading to current node, padded to max level degree
                            new_path = "".join(str(i) for i in F.one_hot(torch.tensor([answer.index]), num_classes=max_node_degree).squeeze(0).tolist()) + node_path
                            encodings_map[answer.connected_node.key] = new_path
                            next_level_nodes.append((new_path, answer.connected_node))
            current_level_nodes = next_level_nodes

        print("Done")
        # pad all encodings to final length
        return {key: [int(char) for char in encodings_map[key].rjust(encoding_length, '0')] for key in encodings_map}, encoding_length




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
            return F.one_hot(torch.tensor(list(range(num_answers)), dtype=torch.long), num_classes=self.max_degree)
        elif dialog_node.connected_node: 
            # node without answers, but directly connected neighbour
            return F.one_hot(torch.tensor([0], dtype=torch.long), num_classes=self.max_degree)
        else:
            # node without answers and neighbours
            return torch.zeros((1, self.max_degree), dtype=torch.float)

    # TODO 
    @torch.no_grad()
    def batch_encode(self, dialog_node: List[DialogNode], max_actions: int, **kwargs) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        """
        Returns:
            batch x max_actions x encoding
        """
        encoding = torch.zeros(len(dialog_node), max_actions, self.max_degree)
        for node_idx, node in enumerate(dialog_node):
            if len(node.answers) > 0:
                encoding[node_idx, :len(node.answers), :] = F.one_hot(torch.tensor(list(range(len(node.answers)))), num_classes=self.max_degree)
            elif node.connected_node:
                # node without answers, but directly connected neighbour
                encoding[node_idx, 0, 0] = 1.0
        return encoding
   