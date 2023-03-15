
from typing import Set
from chatbot.adviser.app.rl.dataset import DialogNode
import chatbot.adviser.app.rl.dataset as Data

class DialogTree:
    def __init__(self, version: int = 0) -> None:
        self._max_tree_depth = None
        self._max_node_degree = None
        self.version = version

    def get_start_node(self) -> DialogNode:
        """ Get the "real" start node in a safe way:
            * Locates the ONLY possible node of type 'startNode'
        """
        # startNode = DialogNode.objects.get(node_type="startNode", version=self.version)
        startNode = Data.objects[self.version].start_node()
        return startNode

    def _get_max_tree_depth(self, current_node: DialogNode, current_max_depth: int, visited: Set[int]) -> int:
        """ Return maximum tree depth (max. number of steps to leave node) in whole graph """

        if current_node.key in visited:
            return current_max_depth
        visited.add(current_node.key)

        if current_node.node_type == 'startNode':
            # begin recursion at start node
            current_node = Data.objects[self.version].node_by_key(current_node.connected_node_key)

        # if current_node.answers.count() > 0:
        if current_node.answer_count() > 0:
            # normal node
            # continue recursion by visiting children
            # max_child_depth = max([self._get_max_tree_depth(answer.connected_node, current_max_depth + 1, visited) for answer in current_node.answers.all() if answer.connected_node])
            max_child_depth = max([self._get_max_tree_depth(Data.objects[self.version].node_by_key(answer.connected_node_key), current_max_depth + 1, visited) for answer in current_node.answers if answer.connected_node_key])
            return max_child_depth
        elif current_node.connected_node_key:
            # node without answers, e.g. info node
            # continue recursion by visiting children
            return self._get_max_tree_depth(Data.objects[self.version].node_by_key(current_node.connected_node_key), current_max_depth + 1, visited)
        else:
            # reached leaf node
            return current_max_depth


    def get_max_tree_depth(self) -> int:
        """ Return maximum tree depth (max. number of steps to leave node) in whole graph (cached) """
        if not self._max_tree_depth:
            # calculate, then cache value
            self._max_tree_depth = self._get_max_tree_depth(current_node=self.get_start_node(), current_max_depth=0, visited=set([]))
        return self._max_tree_depth

    def _get_max_node_degree(self) -> int:
        """ Return highest node degree in whole graph """
        max_degree = 0
        # for node in DialogNode.objects.filter(version=self.version):
        for node in Data.objects[self.version].nodes():
            # answer_count = node.answers.count()
            answer_count = node.answer_count()
            if answer_count > max_degree:
                max_degree = answer_count
        return max_degree

    def get_max_node_degree(self) -> int:
        """ Return highest node degree in whole graph (cached) """
        if not self._max_node_degree:
            # calculate, then cache value
            self._max_node_degree = self._get_max_node_degree()
        return self._max_node_degree
