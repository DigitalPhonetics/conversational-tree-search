
from typing import List
import torch
import torch.nn.functional as F

from data.dataset import DialogNode, GraphDataset, NodeType
from encoding.base import Encoding


class NodeTypeEncoding(Encoding):
    """ one hot encoding for node type """
    def __init__(self, device: str, data: GraphDataset) -> None:
        super().__init__(device)
        # get all possible node types
        self.node_types = sorted(list(data.nodes_by_type.keys()))  # sort alphabetically for unique order
        # create one-hot encoding for them
        self.encoding_size = len(self.node_types)
        self.encoding = { node_type: F.one_hot(torch.tensor([i]), num_classes=self.encoding_size) for i, node_type in enumerate(self.node_types) if node_type not in [NodeType.VARIABLE_UPDATE] }

    def get_encoding_dim(self) -> int:
        return self.encoding_size

    @torch.no_grad()
    def encode(self, dialog_node: DialogNode) -> torch.FloatTensor:
        """
        Returns:
            torch.FloatTensor (1 x encoding_dim)
        """
        return self.encoding[dialog_node.node_type].clone().float()

    @torch.no_grad()
    def batch_encode(self, dialog_node: List[DialogNode]) -> torch.FloatTensor:
        """
        Returns:
            torch.FloatTensor (batch x encoding_dim)
        """
        return torch.cat([self.encoding[node.node_type] for node in dialog_node]).float()