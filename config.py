
from dataclasses import dataclass
from enum import Enum, IntEnum
from typing import Any, Optional, Type 
from hydra.core.config_store import ConfigStore
import torch
import torch.nn as nn

from data.dataset import DatasetConfig
from encoding.text.base import TextEmbeddingConfig
from rl.utils import AutoSkipMode, GoalDistanceMode


class InstanceArgs(Enum):
    MAX_DISTANCE = 'max_distance'

INSTANCES = {
    InstanceArgs.MAX_DISTANCE: 0
}


class WandbLogLevel(Enum):
    NONE = 'NONE',
    OFFLINE = 'OFFLINE'
    ONLINE = 'ONLIONE'

class DialogLogLevel(Enum):
    NONE = 'NONE'
    FULL = 'FILL'

class ActionType(IntEnum):
    ASK = 0
    SKIP = 1


class InstanceType(Enum):
    ALGORITHM = 'algorithm'
    CONFIG = 'config'
    BUFFER = 'buffer'
    STATE_ENCODING = 'state_encoding'

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
    stop_on_invalid_skip: bool
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
    keep_checkpoints: int

@dataclass
class Experiment:
    _target_: str
    device: str
    seed: int
    torch_compile: bool
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

