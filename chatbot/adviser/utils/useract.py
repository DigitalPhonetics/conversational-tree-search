###############################################################################
#
# Copyright 2019, University of Stuttgart: Institute for Natural Language Processing (IMS)
#
# This file is part of Adviser.
# Adviser is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3.
#
# Adviser is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Adviser.  If not, see <https://www.gnu.org/licenses/>.
#
###############################################################################

"""This module provides the necessary classes for a user action."""

from enum import Enum
from chatbot.adviser.utils.transmittable import Transmittable

class UserActionType(Enum):
    """The type for a user action as used in :class:`UserAct`."""

    Inform = 'inform'
    NegativeInform = 'negativeinform'
    Request = 'request'
    Hello = 'hello'
    Bye = 'bye'
    Thanks = 'thanks'
    Affirm = 'affirm'
    Deny = 'deny'
    RequestAlternatives = 'reqalts'
    Ack = 'ack'
    Bad = 'bad'
    Confirm = 'confirm'
    SelectDomain = 'selectdomain'


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