from chatbot.adviser.utils.transmittable import Transmittable
from enum import Enum


class UserActionType(Enum):
	Hello = 'hello'
	Help = 'help'
	Escalate = 'escalate'
	Bad = 'bad'
	NormalUtterance = 'utterance'
	UnrecognizedValue = 'norecogval'
	TooManyValues = 'toomanyval'

class UserAct(Transmittable):
	"""
	The class for a user action as used in the dialog.

	Args:
		text (str): A textual representation of the user action.
		act_type (UserActionType): The type of the user action.
		slot (str): The slot to which the user action refers - might be ``None`` depending on the
			user action. Default: ``None``.
		value (str): The value to which the user action refers - might be ``None`` depending on the
			user action. Default: ``None``.
		score (float): A value from 0. (not important) to 1. (important) indicating how important
			the information is for the belief state. Default: ``1.0``.

	"""

	def __init__(self, text: str = "", act_type: UserActionType = None, slot: str = None,
				 value: str = None, score: float = 1.0):
		self.text = text
		self.type = act_type
		self.slot = slot
		self.value = value
		self.score = score

	def __repr__(self):
		return "UserAct(\"{}\", {}, {}, {}, {})".format(
			self.text, self.type, self.slot, self.value, self.score)

	def __eq__(self, other):  # to check for equality for tests
		return (self.type == other.type and
				self.slot == other.slot and
				self.value == other.value and
				self.score == other.score)

	def __hash__(self):
		return hash(self.type) * hash(self.slot) * hash(self.value) * hash(self.score)

	def serialize(self):
		return {
			'text': self.text,
			'type': self.type.name,
			'slot': self.slot,
			'value': self.value,
			'score': self.score
		}

	@staticmethod
	def deserialize(obj: dict):
		return UserAct(text=obj['text'], act_type=UserActionType[obj['type']],
						slot=obj['slot'], value=obj['value'], score=obj['score'])   
