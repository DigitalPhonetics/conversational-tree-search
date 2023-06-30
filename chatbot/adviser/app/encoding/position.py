from typing import Dict, List, Tuple
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from chatbot.adviser.app.encoding.encoding import Encoding
from chatbot.adviser.app.rl.dialogtree import DialogTree
from chatbot.adviser.app.rl.dataset import DialogNode
import chatbot.adviser.app.rl.dataset as Data

"""
IDEA: encode node position in tree in a structured way instead of a random one-hot-encoding
https://papers.nips.cc/paper/2019/hash/6e0917469214d8fbd8c517dcdc6b8dcf-Abstract.html

Here, we encode non-regular trees:
* start node = 0
* each tree level can have a different node degree
    * but all nodes inside the same level are padded to the same degree
"""
class TreePositionEncoding(Encoding):
    def __init__(self, device: str) -> None:
        super().__init__(device)

        # build encoding
        self.node_mapping, self.encodings_length = self._process_node_tree()
        # build encoding vectors

    @torch.no_grad()
    def encode(self, dialog_node: DialogNode, **kwargs) -> torch.FloatTensor:
        """
        Returns:
            torch.FloatTensor (1 x encoding_dim)
        """
        return torch.tensor([self.node_mapping[dialog_node.key]], dtype=torch.float, device=self.device)

    @torch.no_grad()
    def batch_encode(self, dialog_node: List[DialogNode], **kwargs) -> torch.FloatTensor:
        """
        Returns:
            torch.FloatTensor (batch x encoding_dim)
        """
        return torch.tensor([self.node_mapping[node.key] for node in dialog_node], dtype=torch.float, device=self.device)


    def get_encoding_dim(self) -> int:
        return self.encodings_length
    
    def _get_start_node(self) -> DialogNode:
        """ Returns the start node in a safe way """
        startNode = Data.objects[0].node_by_key(Data.objects[0].start_node().connected_node_key)
        assert startNode.node_type != "startNode"
        return startNode

    def _get_max_node_degree_on_current_level(self, current_level_nodes: List[DialogNode]) -> int:
        """ Returns the maximum node degree in the current tree depth """
        max_degree = 0
        for path, node in current_level_nodes:
            # node_degree = node.answers.count()
            node_degree = node.answer_count()
            if node_degree > max_degree:
                max_degree = node_degree
        return max_degree

    def _process_node_tree(self) -> Tuple[Dict[str, str], int]:
        print("Building tree embedding for nodes...")
        current_level_nodes: List[Tuple[str, DialogNode]] = [("", self._get_start_node())] # load start node
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
                
                if node.connected_node_key and not node.connected_node_key in visited_node_ids:
                    new_path = "1".rjust(max_node_degree, "0") + node_path # if no answers but connected node -> answer index = 1
                    encodings_map[node.connected_node_key] = new_path
                    next_level_nodes.append((new_path, Data.objects[0].node_by_key(node.connected_node)))
                else:
                    for answer in node.answers:
                        if answer.connected_node_key and not answer.connected_node_key in visited_node_ids:
                            # append binary node answer index to path leading to current node, padded to max level degree
                            new_path = "".join(str(i) for i in F.one_hot(torch.tensor([answer.answer_index]), num_classes=max_node_degree).squeeze(0).tolist()) + node_path
                            encodings_map[answer.connected_node_key] = new_path
                            next_level_nodes.append((new_path, Data.objects[0].node_by_key(answer.connected_node_key)))
            current_level_nodes = next_level_nodes

        print("Done")
        # pad all encodings to final length
        return {key: [int(char) for char in encodings_map[key].rjust(encoding_length, '0')] for key in encodings_map}, encoding_length




class NodeTypeEncoding(Encoding):
    """ one hot encoding for node type """
    def __init__(self, device: str) -> None:
        # get all possible node types
        super().__init__(device)
        # self.node_types = [result['node_type'] for result in DialogNode.objects.values('node_type').distinct()]
        self.node_types = Data.objects[0].node_types()
        self.encoding_size = len(self.node_types)
        # create one-hot encoding for them
        self.encoding = { node_type: str(i) for i, node_type in enumerate(self.node_types) }

    def get_encoding_dim(self) -> int:
        return self.encoding_size

    @torch.no_grad()
    def encode(self, dialog_node: DialogNode, **kwargs) -> torch.FloatTensor:
        """
        Returns:
            torch.FloatTensor (1 x encoding_dim)
        """
        return torch.tensor([list(int(bit) for bit in self.encoding[dialog_node.node_type].zfill(self.encoding_size))], dtype=torch.float, device=self.device)

    @torch.no_grad()
    def batch_encode(self, dialog_node: DialogNode, **kwargs) -> torch.FloatTensor:
        """
        Returns:
            torch.FloatTensor (1 x encoding_dim)
        """
        return torch.tensor([[list(int(bit) for bit in self.encoding[node.node_type].zfill(self.encoding_size))] for node in dialog_node], dtype=torch.float, device=self.device)



class AnswerPositionEncoding(Encoding):
    """ one hot encoding for answer position inside node """
    def __init__(self, device: str, dialog_tree: DialogTree) -> None:
        # get maximum node degree
        self.max_degree = dialog_tree.get_max_node_degree()
        self.device = device
    
    def get_encoding_dim(self) -> int:
        return self.max_degree
    
    @torch.no_grad()
    def encode(self, dialog_node: DialogNode, **kwargs):
        """
        Returns:
            # 1 x answers(dialog_node) x max_node_degree
        """
        # num_answers = dialog_node.answers.count()
        num_answers = dialog_node.answer_count()
        if num_answers > 0:
            return F.one_hot(torch.tensor([list(range(num_answers))], dtype=torch.long, device=self.device), num_classes=self.max_degree)
        # elif dialog_node.connected_node:
        elif dialog_node.connected_node_key:
            return F.one_hot(torch.tensor([[0]], dtype=torch.long, device=self.device), num_classes=self.max_degree)
        else:
            return torch.zeros((1, 1, self.max_degree), dtype=torch.float, device=self.device)

    @torch.no_grad()
    def batch_encode(self, dialog_node: List[DialogNode], **kwargs) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        """
        Returns:
            # batch x max_answers x max_node_degree, batch (num_answers)
        """
        # return pad_sequence([self._encode(node) for node in dialog_node], batch_first=True), torch.tensor([node.answers.count() for node in dialog_node], device=self.device)
        return pad_sequence([self._encode(node) for node in dialog_node], batch_first=True), torch.tensor([node.answer_count() for node in dialog_node], device=self.device)
        