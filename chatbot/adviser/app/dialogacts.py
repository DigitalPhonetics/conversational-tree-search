from chatbot.adviser.utils.transmittable import Transmittable
from enum import Enum

class SysActionType(Enum):
    ASK = 'ask',
    SKIP = 'skip',
    STOP = 'stop'

class SysAct(Transmittable):
    def __init__(self, act_type: SysActionType, dialog_node_id: int = None, dialog_answer_id: int = None) -> None:
        self.act_type = act_type
        self.dialog_node_id = dialog_node_id
        self.dialog_answer_id = dialog_answer_id

    def serialize(self):
        return {
                'type': self.act_type.name,
                'dialog_node_id': self.dialog_node_id,
                'dialog_answer_id': self.dialog_answer_id
            }

    @staticmethod
    def deserialize(obj: dict):
        return SysAct(
            act_type=SysActionType[obj['type']],
            dialog_node_id=obj['dialog_node_id'],
            dialog_answer_id=obj['dialog_answer_id']
        )