from data.buffers.dqn.base import BufferBase


def beta_schedule(start_b: float, duration: int, t: int):
        slope = (1.0 - start_b) / duration
        return min(slope * t + start_b, 1.0)

# TODO call beta_schedule in store() method

class PrioritizedReplayBuffer(BufferBase):
    def __init__(self,  **kwargs) -> None:
        super().__init__()
        print("INIT PRIO", kwargs)
        pass