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

from chatbot.adviser.app.encoding.text import FinetunedGBertEmbeddings, GBertEmbeddings, SentenceEmbeddings

resource_dir = Path(".", 'resources', 'en')

class StateEntry(Enum):
    DIALOG_NODE = 'dialog_node'
    DIALOG_NODE_KEY = "dialog_node_key"
    ORIGINAL_USER_UTTERANCE = "original_user_utterance"
    CURRENT_USER_UTTERANCE = "current_user_utterance"
    SYSTEM_UTTERANCE_HISTORY = "system_utterance_history"
    USER_UTTERANCE_HISTORY = 'user_utterance_history'
    BST = 'bst'
    LAST_SYSACT = "last_sysact"
    DIALOG_HISTORY = 'dialog_history'
    NOISE = "noise"

class EnvInfo(Enum):
    NODE_KEY = "node_key"
    PREV_NODE_KEY = 'prev_node_key'
    EPISODE_REWARD = "episode_reward"
    EPISODE_LENGTH = "current_step"
    EPISODE = "current_episode"
    REACHED_GOAL_ONCE = "reached_goal_once"
    ASKED_GOAL = "asked_goal"
    IS_FAQ = "is_faq_mode"


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


def _del_checkpoint(filename: str):
    os.remove(filename)


def _save_checkpoint(global_step: int, episode_counter: int, train_counter: int, run_name: str, model_state_dict: dict, optimizer_state_dict: dict, epsilon: float,
                     torch_rng, numpy_rng, rand_rng):
    
    torch.save({
        "model": model_state_dict,
        # "config": self.args,
        "global_step": global_step,
        "optimizer": optimizer_state_dict,
        "epsilon": epsilon,
        "episode_counter": episode_counter,
        "train_counter": train_counter,
        "torch_rng": torch_rng,
        "numpy_rng": numpy_rng,
        "rand_rng": rand_rng
    }, f"/mount/arbeitsdaten/asr-2/vaethdk/tmp_debugging_weights/{run_name}/ckpt_{global_step}.pt")



EMBEDDINGS = {
    'gbert-large': {
        'class': GBertEmbeddings,
        'args': {
            'pretrained_name': 'deepset/gbert-large',
            'embedding_dim': 1024,
        }
    },
    'finetuned-gbert-large': {
        'class': FinetunedGBertEmbeddings,
        'args': {
            'pretrained_name': 'gbert-finetuned',
            'embedding_dim': 1024,
        }
    },
    'cross-en-de-roberta-sentence-transformer': {
        'class': SentenceEmbeddings,
        'args': {
            'pretrained_name': 'T-Systems-onsite/cross-en-de-roberta-sentence-transformer',
            'embedding_dim': 768,
        }
    },
    'mpnet-base': {
        'class': SentenceEmbeddings,
        'args': {
            'pretrained_name': 'sentence-transformers/all-mpnet-base-v2',
            'embedding_dim': 768,
        }
    },
    'distiluse-base-multilingual-cased-v2': {
        'class': SentenceEmbeddings,
        'args': {
            'pretrained_name': 'distiluse-base-multilingual-cased-v2',
            'embedding_dim': 512,
        }
    },
}


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


def _load_answer_synonyms(mode: EnvironmentMode, use_synonyms: bool, use_joint_dataset: bool = False) -> Dict[str, List[str]]:
    if use_joint_dataset:
        path = "resources/en/traintest_answers.json"
    else:
        if mode in [EnvironmentMode.TRAIN, EnvironmentMode.EVAL]:
            path = "resources/en/train_answers.json"
        else:
            path = "resources/en/test_answers.json"
    answers = None
    with open(path, "r") as f:
        answers = json.load(f)
    if not use_synonyms:
        # choose key to have same data for train and test set
        answers = {answer.lower(): [answer] for answer in answers}
    else:
        answers = {answer.lower(): answers[answer] for answer in answers}
    return answers

def _load_a1_laenderliste():
    a1_laenderliste = None
    with open(resource_dir / "a1_countries.json", "r") as f:
        a1_laenderliste = json.load(f)
    return a1_laenderliste


class NodeType(Enum):
    INFO = "infoNode"
    VARIABLE = "userInputNode"
    QUESTION = "userResponseNode"
    LOGIC = "logicNode"
