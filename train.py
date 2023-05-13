from typing import Tuple
import gymnasium as gym
import hydra
from hydra.core.config_store import ConfigStore
from hydra.utils import instantiate
from omegaconf import OmegaConf
from algorithm.dqn.dqn import CustomDQNPolicy

from config import INSTANCES, ActionConfig, InstanceType, StateConfig, register_configs
from data.cache import Cache
from data.dataset import GraphDataset
from encoding.state import StateEncoding
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback


import wandb
from wandb.integration.sb3 import WandbCallback

from environment.vec.vecenv import CustomVecEnv
from environment.vec.cts import CTSEnvironment


cs = ConfigStore.instance()
register_configs()


CACHE_HOST = "localhost"
CACHE_PORT = 64123


def setup_cache_and_encoding(device: str, data: GraphDataset, state_config: StateConfig, action_config: ActionConfig) -> Tuple[Cache, StateEncoding]:
    cache = Cache(device=device, data=data, host=CACHE_HOST, port=CACHE_PORT, state_config=state_config)
    encoding = StateEncoding(cache=cache, state_config=state_config, action_config=action_config, data=data)
    return cache, encoding


def make_env(env_id: int, cache: Cache, mode: str, dataset: GraphDataset, state_encoding, environment_config):
    return CTSEnvironment(env_id=env_id, cache=cache, mode=mode, dataset=dataset, state_encoding=state_encoding, **environment_config)


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

    run = wandb.init(
        project="cts_en_stablebaselines",
        config=cfg,
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        # monitor_gym=True,  # auto-upload the videos of agents playing the game
        # save_code=True,  # optional
    )
    # TODO make wandb callback configurable and optional for testing 
    callbacks = [
        WandbCallback(
            model_save_path=f"/mount/arbeitsdaten/asr-2/vaethdk/tmp_debugging_weights/{run.id}",
            verbose=2,
        )
    ]

    env_id = 0
    if "training" in cfg.experiment: 
        train_data = instantiate(cfg.experiment.training.dataset, _target_='data.dataset.GraphDataset')
        if not cache:
            cache, state_encoding = setup_cache_and_encoding(device=cfg.experiment.device, data=train_data, state_config=cfg.experiment.state, action_config=cfg.experiment.actions)
        # train_env = CTSEnvironment(env_id=env_id, cache=cache, mode='train', dataset=train_data, state_encoding=state_encoding, **cfg.experiment.environment)
        # subprocess vec env test
        # from stable_baselines3.common.vec_env import SubprocVecEnv
        from stable_baselines3.common.env_util import make_vec_env
        gym.envs.register(
            id='simulator-v1',
            entry_point='environment.cts:CTSEnvironment',
            max_episode_steps=1000,
        )
        kwargs = {
            "env_id": env_id,
            # "cache": cache,
            "mode":'train',
            "dataset": train_data,
            # "state_encoding": state_encoding, 
            **cfg.experiment.environment
        }
        # NOTE: this uses a dummy vec env only, otherwise we get a threading error!
        train_env = make_vec_env(env_id=CTSEnvironment, 
                                 n_envs=cfg.experiment.environment.num_train_envs, env_kwargs=kwargs, 
                                 vec_env_cls=CustomVecEnv, vec_env_kwargs={
                                     "state_encoding": state_encoding,
                                     "sys_token": cfg.experiment.environment.sys_token,
                                     "usr_token": cfg.experiment.environment.usr_token,
                                     "sep_token": cfg.experiment.environment.sep_token
                                 })
        # train_env = Monitor(train_env)
        print("TRAIN ENV")
    # if "validation" in cfg.experiment:
    #     val_data = instantiate(cfg.experiment.validation.dataset, _target_='data.dataset.GraphDataset')
    #     if not cache:
    #         cache, state_encoding = setup_cache_and_encoding(device=cfg.experiment.device, data=val_data, state_config=cfg.experiment.state, action_config=cfg.experiment.actions)
    #     val_env = CTSEnvironment(env_id=200, cache=cache, mode='val', dataset=val_data, state_encoding=state_encoding, **cfg.experiment.environment)
    #     val_env = Monitor(val_env)
    #     callbacks.append(EvalCallback(val_env, best_model_save_path="/mount/arbeitsdaten/asr-2/vaethdk/tmp_debugging_weights/{run.id}/best_eval/weights",
    #                          log_path="/mount/arbeitsdaten/asr-2/vaethdk/tmp_debugging_weights/{run.id}/best_eval/logs", eval_freq=cfg.experiment.validation.every_steps,
    #                          deterministic=True, render=False))
    #     print("VAL ENV")
    # if "testing" in cfg.experiment:
    #     test_data = instantiate(cfg.experiment.testing.dataset, _target_='data.dataset.GraphDataset')
    #     if not cache:
    #         cache, state_encoding = setup_cache_and_encoding(device=cfg.experiment.device, data=test_data, state_config=cfg.experiment.state, action_config=cfg.experiment.actions)
    #     test_env = instantiate(cfg.experiment.environment, _target_="environment.cts.CTSEnvironment", env_id=env_id, cache=cache, mode='test', dataset=test_data, state_encoding=state_encoding)
    #     callbacks.append(EvalCallback(test_env, best_model_save_path="/mount/arbeitsdaten/asr-2/vaethdk/tmp_debugging_weights/{run.id}/best_test/weights",
    #                     log_path="/mount/arbeitsdaten/asr-2/vaethdk/tmp_debugging_weights/{run.id}/best_test/logs", eval_freq=cfg.experiment.testing.every_steps,
    #                     deterministic=True, render=False))
    #     print("TEST ENV")
    
    # trainer = instantiate(cfg.experiment)


    print("STATE SIZE", train_env.observation_space)
    print("ACTION SIZE", train_env.action_space)
    s1 = train_env.reset()
    print(s1.shape)

    # dqn = CustomQNetwork(observation_space=train_env.observation_space, action_space=train_env.action_space,
                            # hidden_layer_sizes=[1024,1024])

    from stable_baselines3 import DQN
    from stable_baselines3.common.env_checker import check_env
    # check_env(train_env)

    policy_kwargs = {
        "hidden_layer_sizes": [1024, 1024],
        "normalization_layers": False
    }
   
    # TODO missing parameters in our config:
    # - tau
    # - gradient_steos
    model = DQN(CustomDQNPolicy, train_env, verbose=1, device=cfg.experiment.device, policy_kwargs=policy_kwargs, 
                learning_rate=cfg.experiment.optimizer.lr, 
                exploration_initial_eps=cfg.experiment.algorithm.dqn.eps_start, exploration_final_eps=cfg.experiment.algorithm.dqn.eps_end, exploration_fraction=cfg.experiment.algorithm.dqn.exploration_fraction,
                buffer_size=cfg.experiment.algorithm.dqn.buffer.backend.buffer_size, 
                learning_starts=cfg.experiment.algorithm.dqn.warmup_turns,
                gamma=cfg.experiment.algorithm.dqn.gamma,
                train_freq=cfg.experiment.algorithm.dqn.train_frequency,
                target_update_interval=cfg.experiment.algorithm.dqn.target_network_update_frequency,
                max_grad_norm=cfg.experiment.algorithm.dqn.max_grad_norm,
                tensorboard_log=f"runs/{run.id}")
    

    model.learn(total_timesteps=cfg.experiment.algorithm.dqn.timesteps_per_reset, log_interval=10000, progress_bar=False,
                    callback=CallbackList(callbacks)
        )
    run.finish()


    # TEST CODE
    # print("RESET")
    # print("STATE S1", s1)
    # s2, reward, done, info = train_env.step(0) # ASK
    # print("========= STEP ASK ==========")
    # print(s2.size())
    # print("STATE S2", s2)
    # print("REWARD", reward)
    # print("DONE", done)
    # print("INFO", info)
    # print("ANSWERS", info[EnvInfo.DIALOG_NODE].answers)
    # print("========= STEP 2 = SKIP -1 ==========")
    # s3, reward, done, info = train_env.step(2) # SKIP
    # print("STATE S3", s3)
    # print(s3.size())
    # print("REWARD", reward)
    # print("DONE", done)
    # print("INFO", info)
    


if __name__ == "__main__":
    load_cfg()