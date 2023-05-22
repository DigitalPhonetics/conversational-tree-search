from typing import Tuple
import hydra
from hydra.core.config_store import ConfigStore
from hydra.utils import instantiate
from omegaconf import OmegaConf
from algorithm.dqn.buffer import CustomReplayBuffer
from algorithm.dqn.dqn import CustomDQN

from config import INSTANCES, ActionConfig, InstanceType, StateConfig, WandbLogLevel, register_configs, EnvironmentConfig, DatasetConfig
from data.cache import Cache
from data.dataset import GraphDataset
from encoding.state import StateEncoding
from stable_baselines3.common.vec_env import VecMonitor
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env

import wandb
from wandb.integration.sb3 import WandbCallback

from environment.vec.vecenv import CustomVecEnv
from environment.cts import CTSEnvironment

import os

from training.stats import CustomEvalCallback
os.environ['TOKENIZERS_PARALLELISM'] = "True"

cs = ConfigStore.instance()
register_configs()


CACHE_HOST = "localhost"
CACHE_PORT = 64123


def to_class(path:str):
    from pydoc import locate
    class_instance = locate(path)
    return class_instance

def setup_cache_and_encoding(device: str, data: GraphDataset, state_config: StateConfig, action_config: ActionConfig,) -> Tuple[Cache, StateEncoding]:
    cache = Cache(device=device, data=data, host=CACHE_HOST, port=CACHE_PORT, state_config=state_config)
    encoding = StateEncoding(cache=cache, state_config=state_config, action_config=action_config, data=data)
    return cache, encoding

def setup_data_and_vecenv(device: str, dataset_cfg: DatasetConfig, environment_cfg: EnvironmentConfig, 
                         mode: str, n_envs: int, log_dir: str,
                         cache: Cache, encoding: StateEncoding,
                         state_config: StateConfig, action_config: ActionConfig) -> Tuple[GraphDataset, Cache, StateEncoding, CustomVecEnv]:
    data = instantiate(dataset_cfg, _target_='data.dataset.GraphDataset')
    if isinstance(cache, type(None)):
        cache, encoding = setup_cache_and_encoding(device=device, data=data, state_config=state_config, action_config=action_config)
    kwargs = {
        # "env_id": env_id,
        "mode": mode,
        "dataset": data,
        **environment_cfg
    }
    vec_env = make_vec_env(env_id=CTSEnvironment, 
                                n_envs=n_envs, env_kwargs=kwargs, 
                                vec_env_cls=CustomVecEnv, vec_env_kwargs={
                                    "state_encoding": encoding,
                                    "sys_token": environment_cfg.sys_token,
                                    "usr_token": environment_cfg.usr_token,
                                    "sep_token": environment_cfg.sep_token
                                })
    vec_env = VecMonitor(vec_env, filename=log_dir)
    return data, cache, encoding, vec_env


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

    cache = None
    state_encoding = None
    
    callbacks = []
    if cfg.experiment.logging.wandb_log != WandbLogLevel.NONE:
        if cfg.experiment.logging.wandb_log == WandbLogLevel.OFFLINE:
            os.environ['WANDB_MODE'] = 'offline'
        run = wandb.init(
            project="cts_en_stablebaselines",
            config=cfg,
            sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
            # monitor_gym=True,  # auto-upload the videos of agents playing the game
            # save_code=True,  # optional
        )
        run_id = run.id
        callbacks.append(WandbCallback(
            model_save_path=f"/mount/arbeitsdaten/asr-2/vaethdk/tmp_debugging_weights/{run_id}",
            verbose=2)
        )
    else:
        import random
        run_id = f"run_{random.randint(0, 999999)}"
    
    if "training" in cfg.experiment and not isinstance(cfg.experiment.training, type(None)): 
        train_data, cache, state_encoding, train_env = setup_data_and_vecenv(device=cfg.experiment.device, dataset_cfg=cfg.experiment.training.dataset, environment_cfg=cfg.experiment.environment,
                                                                        mode="train", n_envs=cfg.experiment.environment.num_train_envs, log_dir=None,
                                                                        cache=cache, encoding=state_encoding,
                                                                        state_config=cfg.experiment.state, action_config=cfg.experiment.actions)

        train_env = VecMonitor(train_env)
    if "validation" in cfg.experiment and not isinstance(cfg.experiment.validation, type(None)): 
        val_data, cache, state_encoding, val_env = setup_data_and_vecenv(device=cfg.experiment.device, dataset_cfg=cfg.experiment.validation.dataset, environment_cfg=cfg.experiment.environment,
                                                                        mode="val", n_envs=cfg.experiment.environment.num_val_envs, log_dir=f"/mount/arbeitsdaten/asr-2/vaethdk/tmp_debugging_weights/{run_id}/best_eval/monitor_logs",
                                                                        cache=cache, encoding=state_encoding,
                                                                        state_config=cfg.experiment.state, action_config=cfg.experiment.actions)
        callbacks.append(CustomEvalCallback(eval_env=val_env, mode='eval',
                             best_model_save_path=f"/mount/arbeitsdaten/asr-2/vaethdk/tmp_debugging_weights/{run_id}/best_eval/weights",
                             log_path=f"/mount/arbeitsdaten/asr-2/vaethdk/tmp_debugging_weights/{run_id}/best_eval/logs",
                             eval_freq=max(cfg.experiment.validation.every_steps // cfg.experiment.environment.num_val_envs, 1),
                             deterministic=True, 
                             render=False,
                             n_eval_episodes=cfg.experiment.validation.dialogs))
    if "testing" in cfg.experiment and not isinstance(cfg.experiment.testing, type(None)):
        test_data, cache, state_encoding, test_env = setup_data_and_vecenv(device=cfg.experiment.device, dataset_cfg=cfg.experiment.testing.dataset, environment_cfg=cfg.experiment.environment,
                                                                        mode="test", n_envs=cfg.experiment.environment.num_test_envs, log_dir=f"/mount/arbeitsdaten/asr-2/vaethdk/tmp_debugging_weights/{run_id}/best_test/monitor_logs",
                                                                        cache=cache, encoding=state_encoding,
                                                                        state_config=cfg.experiment.state, action_config=cfg.experiment.actions)
        callbacks.append(CustomEvalCallback(eval_env=test_env, mode='test',
                        best_model_save_path=f"/mount/arbeitsdaten/asr-2/vaethdk/tmp_debugging_weights/{run_id}/best_test/weights",
                        log_path=f"/mount/arbeitsdaten/asr-2/vaethdk/tmp_debugging_weights/{run_id}/best_test/logs",
                        eval_freq=max(cfg.experiment.testing.every_steps // cfg.experiment.environment.num_test_envs, 1),
                        deterministic=True,
                        render=False,
                        n_eval_episodes=cfg.experiment.testing.dialogs))
    
    # trainer = instantiate(cfg.experiment)

    from stable_baselines3 import HerReplayBuffer
    # check_env(train_env)

    # TODO missing parameters in our config:
    # - tau
    # - gradient_steps
    net_arch = OmegaConf.to_container(cfg.experiment.policy.net_arch)
    net_arch['state_dims'] = state_encoding.space_dims # patch arguments
    policy_kwargs = {
        "activation_fn": to_class(cfg.experiment.policy.activation_fn),   
        "net_arch": net_arch
    }
    model = CustomDQN(policy=to_class(cfg.experiment.policy._target_), policy_kwargs=policy_kwargs,
                env=train_env, 
                batch_size=cfg.experiment.algorithm.dqn.batch_size,
                verbose=1, device=cfg.experiment.device,  
                learning_rate=cfg.experiment.optimizer.lr, 
                exploration_initial_eps=cfg.experiment.algorithm.dqn.eps_start, exploration_final_eps=cfg.experiment.algorithm.dqn.eps_end, exploration_fraction=cfg.experiment.algorithm.dqn.exploration_fraction,
                buffer_size=cfg.experiment.algorithm.dqn.buffer.backend.buffer_size, 
                learning_starts=cfg.experiment.algorithm.dqn.warmup_turns,
                gamma=cfg.experiment.algorithm.dqn.gamma,
                train_freq=cfg.experiment.algorithm.dqn.train_frequency,
                target_update_interval=cfg.experiment.algorithm.dqn.target_network_update_frequency * cfg.experiment.environment.num_train_envs,
                max_grad_norm=cfg.experiment.algorithm.dqn.max_grad_norm,
                tensorboard_log=f"runs/{run_id}",
                replay_buffer_class=CustomReplayBuffer,
                optimize_memory_usage=False,
            ) # TODO configure replay buffer class!
    
    model.learn(total_timesteps=cfg.experiment.algorithm.dqn.timesteps_per_reset, log_interval=cfg.experiment.logging.log_interval, progress_bar=False,
                    callback=CallbackList(callbacks)
        )
    run.finish()



if __name__ == "__main__":
    load_cfg()