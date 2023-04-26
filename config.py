
from dataclasses import dataclass
from enum import Enum, IntEnum
from typing import Any, Optional 
from hydra.core.config_store import ConfigStore
from chatbot.adviser.app.rl.utils import AutoSkipMode

from data.dataset import DatasetConfig
from encoding.text.base import TextEmbeddingConfig


INSTANCES = {}




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
    stop_action: bool

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
    sys_token: Optional[str] = ""
    usr_token: Optional[str] = ""
    sep_token: Optional[str] = ""


@dataclass
class Experiment:
    _target_: str
    device: str
    cache: CacheConfig
    seed: int
    cudnn_deterministic: bool
    optimizer: OptimizerConfig
    model: Any
    algorithm: Any
    logging: Any
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

