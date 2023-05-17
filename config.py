
from dataclasses import dataclass
from enum import Enum, IntEnum
from typing import Any, Optional, Type 
from hydra.core.config_store import ConfigStore
import torch
import torch.nn as nn
from chatbot.adviser.app.rl.utils import AutoSkipMode

from data.dataset import DatasetConfig
from encoding.text.base import TextEmbeddingConfig
from utils.envutils import GoalDistanceMode


INSTANCES = {}


class WandbLogLevel(Enum):
    NONE = 'none',
    OFFLINE = 'offline'
    ONLINE = 'online'

class DialogLogLevel(Enum):
    NONE = 'none'
    FULL = 'full'

class ActionType(IntEnum):
    ASK = 0
    SKIP = 1


class InstanceType(Enum):
    ALGORITHM = 'algorithm'
    CACHE = 'cache'
    CONFIG = 'config'
    BUFFER = 'buffer'

@dataclass
class CacheConfig:
    _target_: str
    active: bool
    host: str = "localhost"
    port: int = 64123


@dataclass
class TrainingStageConfig:
    dataset: DatasetConfig
    every_steps: int
    noise: float
    steps: int

@dataclass
class EvalStageConfig:
    dataset: DatasetConfig
    every_steps: int
    noise: float
    dialogs: int

@dataclass
class OptimizerConfig:
    lr: float
    class_path: str

@dataclass
class ActionConfig:
    in_state_space: bool
    action_masking: bool

@dataclass
class StateConfig:
    last_system_action: bool
    beliefstate: bool
    node_position: bool
    node_type: bool
    action_position: bool
    node_text: Optional[TextEmbeddingConfig] = None
    dialog_history: Optional[TextEmbeddingConfig] = None
    action_text: Optional[TextEmbeddingConfig] = None
    current_user_utterance: Optional[TextEmbeddingConfig] = None
    initial_user_utterance: Optional[TextEmbeddingConfig] = None

    
@dataclass
class EnvironmentConfig:
    guided_free_ratio: float
    auto_skip: AutoSkipMode
    normalize_rewards: bool
    max_steps: int
    user_patience: int
    stop_when_reaching_goal: bool 
    num_train_envs: int 
    num_val_envs: int
    num_test_envs: int
    goal_distance_mode: GoalDistanceMode
    goal_distance_increment: int
    sys_token: Optional[str] = ""
    usr_token: Optional[str] = ""
    sep_token: Optional[str] = ""


@dataclass
class PolicyConfig:
    _target_: str
    activation_fn: str
    net_arch: Any

@dataclass
class LoggingConfig:
    dialog_log: DialogLogLevel
    wandb_log: WandbLogLevel
    log_interval: int

@dataclass
class Experiment:
    _target_: str
    device: str
    cache: CacheConfig
    seed: int
    cudnn_deterministic: bool
    optimizer: OptimizerConfig
    policy: PolicyConfig
    algorithm: Any
    logging: LoggingConfig
    environment: EnvironmentConfig
    actions: ActionConfig
    state: StateConfig
    training: Optional[TrainingStageConfig] = None
    validation: Optional[EvalStageConfig] = None
    testing: Optional[EvalStageConfig] = None

@dataclass
class ConfigEntrypoint:
    experiment: Experiment

def register_configs() -> None:
    cs = ConfigStore.instance()
    cs.store(name="config_schema", node=ConfigEntrypoint)

