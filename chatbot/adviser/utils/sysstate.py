from chatbot.adviser.utils.transmittable import Transmittable
from chatbot.adviser.utils.sysact import SysAct

class SysState(Transmittable):

    def __init__(self, last_act: SysAct = None, last_offer: str = None, last_request_slot: str = None, lastInformedPrimKeyVal: str = None):
        self.last_act = last_act
        self.last_offer = last_offer
        self.last_request_slot = last_request_slot
        self.lastInformedPrimKeyVal = lastInformedPrimKeyVal

    def serialize(self):
        return {
            'last_act': self.last_act.serialize() if self.last_act else self.last_act,
            'last_offer': self.last_offer,
            'last_request_slot': self.last_request_slot,
            'lastInformedPrimKeyVal': self.lastInformedPrimKeyVal
        }

    @staticmethod
    def deserialize(obj: dict):
        last_act = SysAct.deserialize(obj['last_act']) if obj['last_act'] else obj['last_act']
        return SysState(last_act=last_act, last_offer=obj['last_offer'], last_request_slot=obj['last_request_slot'], lastInformedPrimKeyVal=obj['lastInformedPrimKeyVal'])