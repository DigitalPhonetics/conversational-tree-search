from typing import Any, Dict, List
import torch
from chatbot.adviser.app.answerTemplateParser import AnswerTemplateParser

from chatbot.adviser.app.encoding.encoding import Encoding
from data.dataset import GraphDataset, NodeType

class BSTEncoding(Encoding):
    def __init__(self, device: str, data: GraphDataset) -> None:
        super().__init__(device)
        self.variables = self._extract_variables(data)
        
    def _extract_variables(self, data: GraphDataset) -> List[str]:
        # extract all available variables by looking at the variable nodes
        answerParser = AnswerTemplateParser()
        variables = set()
        for node in data.nodes_by_type[NodeType.VARIABLE]:
            # answer = node.answers.all()[0]
            answer = node.answers[0]
            expected_var = answerParser.find_variable(answer.text)
            variables.add(expected_var.name)
        return sorted(list(variables)) # sort alphabetically for unique order

    def get_encoding_dim(self) -> int:
        return len(self.variables)

    @torch.no_grad()
    def encode(self, bst: Dict[str, Any], **kwargs) -> torch.FloatTensor:
        return torch.tensor([1.0 if var_name in bst else 0.0 for var_name in self.variables], device=self.device).unsqueeze(dim=0)

    @torch.no_grad()
    def batch_encode(self, bst: List[Dict[str, Any]], **kwargs) -> torch.FloatTensor:
        """
        Returns:
            batch x bst_entries (one-hot encoded)
        """
        return torch.tensor([[1.0 if var_name in batch_item else 0.0 for var_name in self.variables] for batch_item in bst], device=self.device).unsqueeze(dim=0)
