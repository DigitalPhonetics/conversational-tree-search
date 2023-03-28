from dataclasses import dataclass, field
from typing import List, Type

import hydra
from hydra.core.config_store import ConfigStore
from hydra.utils import instantiate
from omegaconf import OmegaConf

from config import INSTANCES, InstanceType, register_configs

cs = ConfigStore.instance()
register_configs()

@hydra.main(version_base=None, config_path="conf", config_name="default")
def load_cfg(cfg):
    global INSTANCES
    INSTANCES[InstanceType.CONFIG] = cfg
    print(OmegaConf.to_yaml(cfg))
    trainer = instantiate(cfg.experiment)
    print(INSTANCES)


if __name__ == "__main__":
    load_cfg()