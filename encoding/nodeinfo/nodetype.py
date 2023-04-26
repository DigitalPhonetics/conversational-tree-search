
import torch

from data.dataset import DialogNode, GraphDataset
from encoding.base import Encoding


class NodeTypeEncoding(Encoding):
    """ one hot encoding for node type """
    def __init__(self, device: str, data: GraphDataset) -> None:
        super().__init__(device)
        # get all possible node types
        self.node_types = sorted(list(data.nodes_by_type.keys()))  # sort alphabetically for unique order
        # create one-hot encoding for them
        self.encoding_size = len(self.node_types)
        self.encoding = { node_type: str(i) for i, node_type in enumerate(self.node_types) }

    def get_encoding_dim(self) -> int:
        return self.encoding_size

    @torch.no_grad()
    def encode(self, dialog_node: DialogNode) -> torch.FloatTensor:
        """
        Returns:
            torch.FloatTensor (1 x encoding_dim)
        """
        return torch.tensor([list(int(bit) for bit in self.encoding[dialog_node.node_type].zfill(self.encoding_size))], dtype=torch.float, device=self.device)

    @torch.no_grad()
    def batch_encode(self, dialog_node: DialogNode) -> torch.FloatTensor:
        """
        Returns:
            torch.FloatTensor (1 x encoding_dim)
        """
        return torch.tensor([[list(int(bit) for bit in self.encoding[node.node_type].zfill(self.encoding_size))] for node in dialog_node], dtype=torch.float, device=self.device)

