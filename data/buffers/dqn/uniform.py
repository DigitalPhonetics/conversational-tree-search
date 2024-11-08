from data.buffers.dqn.base import BufferBase


class UniformReplayBuffer(BufferBase):
    def __init__(self) -> None:
        super().__init__()