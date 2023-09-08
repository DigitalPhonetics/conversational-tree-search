from copy import deepcopy
from enum import Enum
import hashlib
import json
import os
from pathlib import Path
import random
from statistics import mean
from typing import Dict, List
import torch




class State(Enum):
    # embeddable state
    LAST_SYSACT = "last_system_action"
    BELIEFSTATE = 'beliefstate'
    NODE_POSITION = 'node_positions' # position in tree 
    NODE_TYPE = 'node_type'
    NODE_TEXT = 'node_text'
    INITIAL_USER_UTTERANCE = "initial_user_utterance"
    DIALOG_HISTORY = 'dialog_history'

    # embeddable action state
    ACTION_TEXT = "action_text"
    ACTION_POSITION = 'action_position' # position in tree

    # always embedded
    CURRENT_USER_UTTERANCE = "current_user_utterance"


class EnvInfo(Enum):
    # STATE INFO
    DIALOG_NODE_KEY = 'dialog_node'
    # PREV_NODE_KEY = 'prev_node'
    BELIEFSTATE = 'beliefstate'
    
    # STEP INFO
    EPISODE_REWARD = "episode_reward"
    EPISODE_LENGTH = "current_step"
    PERCIEVED_LENGTH = "percieved_length"
    EPISODE = "current_episode"
    LAST_SYSTEM_ACT = "last_system_act"
    LAST_VALID_SKIP_TRANSITION_IDX = "last_valid_skip_transition_idx"
    
    # GOAL INFO
    REACHED_GOAL_ONCE = "reached_goal_once"
    ASKED_GOAL = "asked_goal"
    GOAL = "goal"
    
    # ENV INFO
    ENV_MODE = "env_mode"
    IS_FAQ = "is_faq"
    
    # TEXTS
    INITIAL_USER_UTTERANCE = "initial_user_utterance"
    CURRENT_USER_UTTERANCE = "current_user_utterance"
    USER_UTTERANCE_HISTORY = 'user_utterance_history'
    SYSTEM_UTTERANCE_HISTORY = "system_utterance_history"


class AutoSkipMode(Enum):
    NONE = "none",
    ORACLE = "oracle"
    SIMILARITY = "similarity"


class AverageMetric:
    def __init__(self, name: str, running_avg: int = 0) -> None:
        self.values = []
        self.running_avg = running_avg
        self.name = name
    
    def log(self, value: float):
        if len(self.values) + 1 > self.running_avg:
            self.values = self.values[1:]
        self.values.append(value)

    def eval(self):
        """
        If running_avg > 0, returns the running average of the last N elements,
        otherwise returns the full mean
        """
        if not self.values:
            return 0.0 
        return mean(self.values)

    def reset(self):
        self.values = []

    
def _get_file_hash(filename: str) -> str:
    BLOCK_SIZE = 65536 # The size of each read from the file
    file_hash = hashlib.sha256() # Create the hash object, can use something other than `.sha256()` if you wish
    with open(filename, 'rb') as f: # Open the file to read it's bytes
        fb = f.read(BLOCK_SIZE) # Read from the file. Take in the amount declared above
        while len(fb) > 0: # While there is still data being read from the file
            file_hash.update(fb) # Update the hash
            fb = f.read(BLOCK_SIZE) # Read the next block from the file

    return file_hash.hexdigest() # Get the hexadecimal digest of the hash


def safe_division(num: float, denum: float) -> float:
    if denum == 0:
        return None
    return num / denum


def _munchausen_stable_logsoftmax(q: torch.FloatTensor, tau: float) -> torch.FloatTensor:
    v = q.max(-1, keepdim=True)[0]
    tau_lse = v + tau * torch.log(
        torch.sum(
            torch.exp((q - v)/tau), dim=-1, keepdim=True
        )
    )
    return q - tau_lse # batch x 1

def _munchausen_stable_softmax(q: torch.FloatTensor, tau: float) -> torch.FloatTensor:
    return torch.softmax((q-q.max(-1, keepdim=True)[0])/tau, -1) # batch



class ExperimentLogging(Enum):
    NONE = "none"
    OFFLINE = 'offline'
    ONLINE = 'online'


def rand_remove_questionmark(text: str):
    # replace question marks in FAQ questions with 50% chance
    if random.random() < 0.5:
        return text.replace("?", "")
    return deepcopy(text)


class EnvironmentMode(Enum):
    TRAIN = 0,
    EVAL = 1
    TEST = 2
