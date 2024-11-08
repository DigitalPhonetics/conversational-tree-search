from enum import Enum


class GoalDistanceMode(Enum):
    # TODO do something with average return? 
    FULL_DISTANCE = "full_distance"
    INCREMENT_EVERY_N_EPISODES = "increment_every_n_episodes"