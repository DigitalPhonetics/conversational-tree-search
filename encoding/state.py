from dataclasses import dataclass
import itertools
from typing import Any, Dict, List, Tuple

import torch
import torch.nn.functional as F

from utils.utils import EnvInfo, State
from config import ActionConfig, StateConfig, ActionType
from data.cache import Cache
from data.dataset import DialogNode, GraphDataset, NodeType



def chain_dialog_history(sys_utterances: List[str], usr_utterances: List[str], sys_token: str = "", usr_token: str = "", sep_token: str = "") -> List[Tuple[str,str,str,str,str]]:
    """
    Interleave system and user utterances to a combined history.

    Args:
        sys_utterances: List of all system utterances (one turn per list entry)
        usr_utterances: List of all user utterances (one turn per list entry)
        sys_token: Token appended to each system utterance, e.g. "[SYS]" -> [SYS] sys turn 1 [USR] usr turn 1 [SYS] sys turn 2 ...
        usr_token: Token appended to each user utterance, e.g. "[SYS]" -> [SYS] sys turn 1 [USR] usr turn 1 [SYS] sys turn 2 ...
        sep_token Seperator token added between each system and user utterance, e.g. "[SEP]" -> sys 1 [SEP] usr turn 1 [SEP] sys turn 2 ...

    Returns: 
        List[Tuple(sys_token, sys_turn, sep_token, usr_token, usr_turn)]
    """
    turns = len(sys_utterances)
    assert len(usr_utterances) == turns
    return list(itertools.chain(zip([sys_token] * turns, [utterance for utterance in sys_utterances], [sep_token] * turns, [usr_token] * turns, [utterance for utterance in usr_utterances])))



@dataclass
class StateDims:
    state_vector: int
    action_vector: int
    state_action_subvector: int # the size of the action subspace in the state space (if actions in state space, else 0)
    num_actions: int # real number of actions, independent of space sizes
        

class StateEncoding:
    """
    Transform observations into tensors (using the encoder cache).
    Also, calculates sizes of state- and action space.
    """
    def __init__(self, cache: Cache, state_config: StateConfig, action_config: ActionConfig, data: GraphDataset) -> None:
        self.cache = cache
        self.data = data
        self.state_config = state_config
        self.action_config = action_config
        self.space_dims = self._get_space_dims(state_config=state_config, action_config=action_config, data=data)

        print("Space dimensions:", str(self.space_dims))

    def _get_space_dims(self, state_config: StateConfig, action_config: ActionConfig, data: GraphDataset) -> StateDims:
        # 1 output for all actions, if actions in state space - otherwise, calculate max. node degree + 1 Action for asking
        action_dim = 1 
        if not action_config.in_state_space:
            action_dim += data.get_max_node_degree() # 1 ASK action + node_degree skip actions

        state_dim = self.cache.action_type_encoding.get_encoding_dim() if action_config.in_state_space else 0  # 2 inputs if actions in state space for encoding the action type: ASK or SKIP, otherwise 0
        state_action_subvector = state_dim
        if state_config.last_system_action:
            # add one-hot encoding of action type
            state_dim += self.cache.action_type_encoding.get_encoding_dim()
        if state_config.beliefstate:
            state_dim += self.cache.bst_encoding.get_encoding_dim()
        if state_config.node_position:
            state_dim += self.cache.node_position_encoding.get_encoding_dim()
        if state_config.node_type:
            state_dim += self.cache.node_type_encoding.get_encoding_dim()
        if state_config.action_position:
            assert action_config.in_state_space, "Can only encode action features if actions are in state space"
            state_dim += self.cache.action_position_encoding.get_encoding_dim()
            state_action_subvector += self.cache.action_position_encoding.get_encoding_dim()
        if state_config.node_text.active:
            state_dim += self.cache.text_embeddings[state_config.node_text._target_].get_encoding_dim()
        if state_config.initial_user_utterance.active:
            state_dim += self.cache.text_embeddings[state_config.initial_user_utterance._target_].get_encoding_dim()
        if state_config.current_user_utterance.active:
            state_dim += self.cache.text_embeddings[state_config.current_user_utterance._target_].get_encoding_dim()
        if state_config.dialog_history.active:
            state_dim += self.cache.text_embeddings[state_config.dialog_history._target_].get_encoding_dim()
        if state_config.action_text.active:
            assert action_config.in_state_space, "Can only encode action features if actions are in state space"
            state_dim += self.cache.text_embeddings[state_config.action_text._target_].get_encoding_dim()
            state_action_subvector += self.cache.text_embeddings[state_config.action_text._target_].get_encoding_dim()

        num_actions = data.get_max_node_degree() + 1 # real number of actions, independent of space sizes (+ 1 for ASK action, rest is skip actions)
        return StateDims(state_vector=state_dim, action_vector=action_dim, state_action_subvector=state_action_subvector, num_actions=num_actions)


    def encode(self, observation: Dict[EnvInfo, Any], sys_token: str, usr_token: str, sep_token: str):
        """
        Calls encoders from cache to transform observation/info dicts into a state vector.

        Returns:
            state_vector (FloatTensor): state_dim, if action_config.in_state_space = False,
                                        num_actions(state) x state_dim, else (NOTE num_actions varies with state!)
        """

        node: DialogNode = self.data.nodes_by_key[observation[EnvInfo.DIALOG_NODE_KEY]]

        # encode state
        state_encoding = []
        if self.state_config.beliefstate:
            state_encoding.append(self.cache.bst_encoding.encode(bst=observation[EnvInfo.BELIEFSTATE]))
        if self.state_config.last_system_action:
            state_encoding.append(self.cache.action_type_encoding.encode(action=observation[EnvInfo.LAST_SYSTEM_ACT]))
        if self.state_config.node_position:
            state_encoding.append(self.cache.node_position_encoding.encode(dialog_node=node))
        if self.state_config.node_type:
            state_encoding.append(self.cache.node_type_encoding.encode(dialog_node=node))
        if self.state_config.node_text and self.state_config.node_text.active:
            state_encoding.append(self.cache.encode_text(state_input_key=State.NODE_TEXT, text=node.text))
        if self.state_config.initial_user_utterance and self.state_config.initial_user_utterance.active:
            state_encoding.append(self.cache.encode_text(state_input_key=State.INITIAL_USER_UTTERANCE, text=observation[EnvInfo.INITIAL_USER_UTTERANCE]))
        if self.state_config.dialog_history and self.state_config.dialog_history.active:
            dialog_history = chain_dialog_history(sys_utterances=observation[EnvInfo.SYSTEM_UTTERANCE_HISTORY], usr_utterances=observation[EnvInfo.USER_UTTERANCE_HISTORY],
                                                            sys_token=sys_token, usr_token=usr_token, sep_token=sep_token)
            dialog_history = ["".join(turn) for turn in dialog_history]
            dialog_history = "".join(dialog_history)
            state_encoding.append(self.cache.encode_text(state_input_key=State.DIALOG_HISTORY, text=dialog_history))
        if self.state_config.current_user_utterance and self.state_config.current_user_utterance.active:
            state_encoding.append(self.cache.encode_text(State.CURRENT_USER_UTTERANCE, text=observation[EnvInfo.CURRENT_USER_UTTERANCE]))

        # print([s.size() for s in state_encoding])
        state_encoding = torch.cat(state_encoding, dim=-1) # 1 x state_dim - state_action_subvector

        assert state_encoding.size(0) == 1, f"observation encoding for single observation should not produce batch, but found dimensions {state_encoding.size()}"
        assert state_encoding.size(1) == self.space_dims.state_vector - self.space_dims.state_action_subvector, f"Expected observation vector without action encodings to be of size {self.space_dims.state_vector - self.space_dims.state_action_subvector}, but found dimensions {state_encoding.size()}"

        # encode actions
        if self.action_config.in_state_space:
            # actions are part of input space: embed them, and concatenate each of them with the state encoding vector
            # each action encoding starts with the action type encoding (ASK or SKIP as a one-hot encoding: [1,0] - ask, or [0,1] - skip)
            num_answers = len(node.answers) # get number of skip actions
            action_encoding = []
            pad_rows = self.space_dims.num_actions - num_answers - 1  # subtract 1 for ASK action
            
            # encode ask action (available for all nodes) except logic / start nodes
            # ASK action should just tell action type, and not have any action position or action text (pad with 0 after action type)
            assert node.node_type not in [NodeType.START]
            action_encoding.append(F.one_hot(torch.tensor([ActionType.ASK.value], dtype=torch.long), num_classes=self.space_dims.state_action_subvector)) # 1 x state_action_subvector

            if num_answers == 0 and node.connected_node:
                # no answers, but a connected node: add zero-padded SKIP action
                action_encoding.append(F.one_hot(torch.tensor([ActionType.SKIP.value], dtype=torch.long), num_classes=self.space_dims.state_action_subvector)) # 1 x state_action_subvector
                pad_rows -= 1 # subtract 1 for default SKIP action
            elif num_answers > 0 and (self.state_config.action_position or self.state_config.action_text.active):
                answer_info_encoding = [
                    F.one_hot(torch.tensor([ActionType.SKIP.value] * num_answers, dtype=torch.long), num_classes=2) # always add action type (ASK or SKIP)
                ]
                if self.state_config.action_position:
                    answer_info_encoding.append(self.cache.action_position_encoding.encode(dialog_node=node)) # num_answers x max_node_degree
                if self.state_config.action_text.active:
                    answer_info_encoding.append(self.cache.encode_answer_text(node=node)) # num_answers x embedding_size
                # concatenate answer info
                answer_info_encoding = torch.cat(answer_info_encoding, dim=-1) # num_answers x state_action_subvector (<= 2 + max_node_degree + embedding_size)
                assert answer_info_encoding.size(0) == num_answers, f"expected {num_answers} answers as first dimension, got {answer_info_encoding.size()}"
                assert answer_info_encoding.size(1) == self.space_dims.state_action_subvector, f"expected {self.space_dims.state_action_subvector} in last dimension, got {answer_info_encoding.size()}"
                action_encoding.append(answer_info_encoding)
            action_encoding = torch.cat(action_encoding, dim=0) # num_answers x state_action_subvector
            # zero-pad rows (bottom) to max. action number
            action_encoding = F.pad(action_encoding, (0,0,0, pad_rows), 'constant', 0.0)
            assert action_encoding.size(0) == self.space_dims.num_actions
            # concatenate actions with state (duplicate state encoding for each action)
            state_encoding = state_encoding.repeat(self.space_dims.num_actions, 1) # num_actions x state_dim - state_action_subvector
            state_encoding = torch.cat((state_encoding, action_encoding), -1) # num_actions x state_dim
            if pad_rows > 0:
                state_encoding[-pad_rows:, :] = 0.0 # mask repeated state in padded rows
        return state_encoding.squeeze()

    def batch_encode(self, observation: List[Dict[EnvInfo, Any]], sys_token: str, usr_token: str, sep_token: str):
        nodes: List[DialogNode] = [self.data.nodes_by_key[obs[EnvInfo.DIALOG_NODE_KEY]] for obs in observation]

        state_encoding = []
        if self.state_config.beliefstate:
            state_encoding.append(self.cache.bst_encoding.batch_encode(bst=[obs[EnvInfo.BELIEFSTATE] for obs in observation]))
        if self.state_config.last_system_action:
            state_encoding.append(self.cache.action_type_encoding.batch_encode(action=[obs[EnvInfo.LAST_SYSTEM_ACT] for obs in observation]))
        if self.state_config.node_position:
            state_encoding.append(self.cache.node_position_encoding.batch_encode(dialog_node=nodes))
        if self.state_config.node_type:
            state_encoding.append(self.cache.node_type_encoding.batch_encode(dialog_node=nodes))
        if self.state_config.node_text and self.state_config.node_text.active:
            state_encoding.append(self.cache.batch_encode_text(state_input_key=State.NODE_TEXT, text=[node.text for node in nodes]))
        if self.state_config.initial_user_utterance and self.state_config.initial_user_utterance.active:
            state_encoding.append(self.cache.batch_encode_text(state_input_key=State.INITIAL_USER_UTTERANCE, text=[obs[EnvInfo.INITIAL_USER_UTTERANCE] for obs in observation]))
        if self.state_config.dialog_history and self.state_config.dialog_history.active:
            dialog_history = [chain_dialog_history(sys_utterances=obs[EnvInfo.SYSTEM_UTTERANCE_HISTORY], usr_utterances=obs[EnvInfo.USER_UTTERANCE_HISTORY],
                                                            sys_token=sys_token, usr_token=usr_token, sep_token=sep_token) for obs in observation]
            dialog_history = [["".join(turn) for turn in batch_item] for batch_item in dialog_history]
            dialog_history = ["".join(batch_item) for batch_item in dialog_history]
            state_encoding.append(self.cache.batch_encode_text(state_input_key=State.DIALOG_HISTORY, text=dialog_history))
        if self.state_config.current_user_utterance and self.state_config.current_user_utterance.active:
            state_encoding.append(self.cache.batch_encode_text(State.CURRENT_USER_UTTERANCE, text=[obs[EnvInfo.CURRENT_USER_UTTERANCE] for obs in observation]))

        state_encoding = torch.cat(state_encoding, dim=-1) # batch x (state_dim - state_action_subvector)

        assert state_encoding.size(0) == len(nodes), f"observation encoding for batch observation should match batch size {len(nodes)}, but found dimensions {state_encoding.size()}"
        assert state_encoding.size(1) == self.space_dims.state_vector - self.space_dims.state_action_subvector, f"Expected observation vector without action encodings to be of size {self.space_dims.state_vector - self.space_dims.state_action_subvector}, but found dimensions {state_encoding.size()}"

        # encode actions
        if self.action_config.in_state_space:
            # actions are part of input space: embed them, and concatenate each of them with the state encoding vector
            # each action encoding starts with the action type encoding (ASK or SKIP as a one-hot encoding: [1,0] - ask, or [0,1] - skip)
            action_encoding = torch.zeros(len(nodes), self.space_dims.num_actions, self.space_dims.state_action_subvector)
            subvec_index = 0
            
            # encode ask action (available for all nodes) except logic / start nodes
            # ASK action should just tell action type, and not have any action position or action text (pad with 0 after action type)
            # assert node.node_type not in [NodeType.START]
            action_encoding[:,0,:] =  F.one_hot(torch.tensor([ActionType.ASK.value] * len(nodes), dtype=torch.long), num_classes=self.space_dims.state_action_subvector) # batch x state_action_subvector
            subvec_index += 2 # action type encoding
            if self.state_config.action_text.active:
                action_text_embedding_name = self.cache.state_embedding_cfg[State.ACTION_TEXT]._target_
                action_text_embedding = self.cache.text_embeddings[action_text_embedding_name]
                subvec_index += action_text_embedding.get_encoding_dim()
                action_encoding[:, 1:, :subvec_index] = self.cache.batch_encode_answer_text(node=nodes, action_space_dim=self.space_dims.num_actions-1) # batch x max_answers x embedding; start from 1 and actions-1 because of ASK action
            if self.state_config.action_position:
                action_encoding[:, 1:, subvec_index:subvec_index+self.cache.action_position_encoding.get_encoding_dim()] = self.cache.action_position_encoding.batch_encode(dialog_node=nodes, max_actions=self.space_dims.num_actions-1) # start from 1 and actions-1 because of ASK action
            
            # concatenate actions with state (duplicate state encoding for each action)
            state_encoding = state_encoding.unsqueeze(1).repeat(1, self.space_dims.num_actions, 1) # batch x num_actions x (state_dim - state_action_subvector)
            state_encoding = torch.cat((state_encoding, action_encoding), -1) # batch x num_actions x state_dim

            # mask repeated state in zero-rows
            for node_idx, node in enumerate(nodes):
                has_connected_node = not isinstance(node.connected_node, type(None))
                node_action_count = max(len(node.answers) + 1, 1 + int(has_connected_node)) # add ASK action, second argument: no answers: node gets a default SKIP action, if not end of tree
                if node_action_count < self.space_dims.num_actions:
                    # we have padding to do!
                    state_encoding[node_idx, node_action_count:, :] = 0.0
            return state_encoding
        return state_encoding
