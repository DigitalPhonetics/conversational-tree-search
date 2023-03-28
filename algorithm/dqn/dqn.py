
from config import INSTANCES, InstanceType
from algorithm.algorithm import Algorithm


class DQNAlgorithm(Algorithm):
    def __init__(self, buffer, targets,
        timesteps_per_reset: int,
        reset_exploration_times: int,
        max_grad_norm: float,
        batch_size: int,
        gamma: float,
        exploration_fraction: float,
        eps_start: float,
        eps_end: float,
        train_frequency: int,
        warmup_turns: int,
        target_network_update_frequency: int,
        q_value_clipping: float
        ) -> None:
        global INSTANCES
        print("DQN Trainer")

        INSTANCES[InstanceType.ALGORITHM] = self

    def run_single_timestep(engine, timestep):
        pass

