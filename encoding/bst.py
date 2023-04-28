from typing import Any, Dict, List
import torch
from chatbot.adviser.app.answerTemplateParser import AnswerTemplateParser
from data.dataset import GraphDataset, NodeType

from encoding.base import Encoding


class BSTEncoding(Encoding):
    """
    Binary indicator vector for BST (variable value is in history or not) - but doesn't contain the actual values
    """
    def __init__(self, device: str, data: GraphDataset) -> None:
        super().__init__(device)
        self.variables = self._extract_variables(data)
        
    def _extract_variables(self, data: GraphDataset) -> List[str]:
        # extract all available variables by looking at the variable nodes
        answerParser = AnswerTemplateParser()
        variables = set()
        for node in data.nodes_by_type[NodeType.VARIABLE]:
            answer = node.answer_by_index(0)
            expected_var = answerParser.find_variable(answer.text)
            variables.add(expected_var.name)
        return sorted(list(variables)) # sort alphabetically for unique order

    def get_encoding_dim(self) -> int:
        return len(self.variables)

    @torch.no_grad()
    def encode(self, bst: Dict[str, Any]) -> torch.FloatTensor:
        return torch.tensor([1.0 if var_name in bst else 0.0 for var_name in self.variables]).unsqueeze(dim=0)

    @torch.no_grad()
    def batch_encode(self, bst: List[Dict[str, Any]]) -> torch.FloatTensor:
        """
        Returns:
            batch x bst_entries (one-hot encoded)
        """
        return torch.tensor([[1.0 if var_name in batch_item else 0.0 for var_name in self.variables] for batch_item in bst]).unsqueeze(dim=0)
