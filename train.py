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

    train_data = None
    val_data = None
    test_data = None

    train_env = None
    val_env = None
    test_env = None

    if "training" in cfg.experiment: 
        train_data = instantiate(cfg.experiment.training.dataset, _target_='data.dataset.GraphDataset')
        train_env = instantiate(cfg.experiment.environment, _target_="environment.base.CTSEnvironment", mode='train', dataset=train_data)
    if "validation" in cfg.experiment:
        val_data = instantiate(cfg.experiment.validation.dataset, _target_='data.dataset.GraphDataset')
        val_env = instantiate(cfg.experiment.environment, _target_="environment.base.CTSEnvironment", mode='val', dataset=val_data)
    if "testing" in cfg.experiment:
        test_data = instantiate(cfg.experiment.testing.dataset, _target_='data.dataset.GraphDataset')
        test_env = instantiate(cfg.experiment.environment, _target_="environment.base.CTSEnvironment", mode='test', dataset=test_data)
    
    
    # trainer = instantiate(cfg.experiment)



if __name__ == "__main__":
    load_cfg()