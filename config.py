
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional, Type, Union
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

from data.dataset import DatasetConfig
from embeddings.text.base import TextEmbeddingConfig
from environment.base import EnvironmentConfig



INSTANCES = {}

class InstanceType(Enum):
    ALGORITHM = 'algorithm'
    CACHE = 'cache'
    CONFIG = 'config'

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
    original_user_utterance: Optional[TextEmbeddingConfig] = None
    dialog_history: Optional[Any] = None
    action_text: Optional[TextEmbeddingConfig] = None


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

