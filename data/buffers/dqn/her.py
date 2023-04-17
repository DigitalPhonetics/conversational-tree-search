from data.buffers.dqn.base import BufferBase


class HindsightExperienceReplayBuffer(BufferBase):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        print("HER INIT", kwargs)