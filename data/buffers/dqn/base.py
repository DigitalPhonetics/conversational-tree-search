from config import INSTANCES, InstanceType


class BufferBase:
    def __init__(self) -> None:
        global INSTANCES
        INSTANCES[InstanceType.BUFFER] = self