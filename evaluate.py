import json
import re
import shutil
from typing import Tuple
import hydra
from hydra.core.config_store import ConfigStore
from hydra.utils import instantiate
from omegaconf import OmegaConf
from algorithm.dqn.buffer import CustomReplayBuffer
# from algorithm.dqn.buffer import CustomReplayBuffer
from algorithm.dqn.dqn import CustomDQN
from algorithm.dqn.her import HindsightExperienceReplayWrapper
# from environment.old.cts import OldCTSEnv, OldCustomVecEnv
# from environment.old.her import OldHindsightExperienceReplayWrapper
from utils.utils import AutoSkipMode, to_class

from config import INSTANCES, ActionConfig, InstanceType, StateConfig, WandbLogLevel, register_configs, EnvironmentConfig, DatasetConfig
from data.cache import Cache
from data.dataset import GraphDataset
from encoding.state import StateEncoding
from stable_baselines3.common.vec_env import VecMonitor
from stable_baselines3.common.callbacks import CallbackList #, CheckpointCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import configure_logger
import torch as th
import logging

import wandb
# from wandb.integration.sb3 import WandbCallback

from environment.vec.vecenv import CustomVecEnv
from environment.cts import CTSEnvironment

import os

run_id = "run_1694965093"
cfg_name = "reimburse_eval_debug.yaml"
DEVICE = "cuda:0"

print("EVALUATING for run with id", run_id)

th.set_num_threads(8) # default on server: 32
th.set_num_interop_threads(8) # default on server: 32

from training.stats import CustomEvalCallback
os.environ['TOKENIZERS_PARALLELISM'] = "True"

OmegaConf.register_resolver("run_dir", lambda : f"./.evaluation")

cs = ConfigStore.instance()
register_configs()


def get_latest_checkpoint_name(ckpt_path: str) -> str:
    highest_number = -1
    for file in os.listdir(ckpt_path):
        if ".pt" in file or ".zip" in file:
            number = int(file.strip("ckpt_").strip(".pt").strip(".zip"))
            if number > highest_number:
                highest_number = number
    if os.path.isfile(f"{ckpt_path}/ckpt_{highest_number}.pt"):
        return f"ckpt_{highest_number}.pt"
    elif os.path.isfile(f"{ckpt_path}/ckpt_{highest_number}.zip"):
        return f"ckpt_{highest_number}.zip"
    else:
        assert False, f"File not found in {ckpt_path} with ckpt num {highest_number}"

path = f"/mount/arbeitsdaten/asr-2/vaethdk/cts_newcodebase_weights/{run_id}/best_eval/weights"
ckpt_dir = os.path.join(path, 'tmp')
ckpt_name = get_latest_checkpoint_name(path)
print("Latest checkpoint:", ckpt_name)
zip_path = f"{path}/{ckpt_name}"

print("Extracting files from ", zip_path, "...")
if not os.path.exists(ckpt_dir):
    import subprocess
    print(f"CMD: unzip -j {zip_path} -d {ckpt_dir}" )
    subprocess.call(['unzip', '-j', zip_path, '-d', ckpt_dir])


with open(f"{ckpt_dir}/data", "r") as f:
    ckpt_data = json.load(f)


print("Beginning normal initalization!")

def setup_cache_and_encoding(device: str, data: GraphDataset, state_config: StateConfig, action_config: ActionConfig, torch_compile: bool) -> Tuple[Cache, StateEncoding]:
    cache = Cache(device=device, data=data, state_config=state_config, torch_compile=torch_compile)
    encoding = StateEncoding(cache=cache, state_config=state_config, action_config=action_config, data=data)
    return cache, encoding

def setup_data_and_vecenv(device: str, dataset_cfg: DatasetConfig, environment_cfg: EnvironmentConfig, 
                         mode: str, n_envs: int, log_dir: str,
                         cache: Cache, encoding: StateEncoding,
                         state_config: StateConfig, action_config: ActionConfig,
                         torch_compile: bool,
                         save_terminal_obs: bool,
                         noise: float) -> Tuple[GraphDataset, Cache, StateEncoding, CustomVecEnv]:
    data = instantiate(dataset_cfg)
    if isinstance(cache, type(None)):
        cache, encoding = setup_cache_and_encoding(device=device, data=data, state_config=state_config, action_config=action_config, torch_compile=torch_compile)
    kwargs = {
        # "env_id": env_id,
        "mode": mode,
        "dataset": data,
        "state_encoding": encoding,
        "noise": noise,
        **environment_cfg
    }
    vec_env = make_vec_env(env_id=CTSEnvironment, 
                                n_envs=n_envs, env_kwargs=kwargs, 
                                vec_env_cls=CustomVecEnv, vec_env_kwargs={
                                    "state_encoding": encoding,
                                    "sys_token": environment_cfg.sys_token,
                                    "usr_token": environment_cfg.usr_token,
                                    "sep_token": environment_cfg.sep_token,
                                    "save_terminal_obs": save_terminal_obs
                                })
    vec_env = VecMonitor(vec_env, filename=log_dir)
    return data, cache, encoding, vec_env


@hydra.main(version_base=None, config_path="conf", config_name=cfg_name)
def load_cfg(cfg):
    global INSTANCES
    print(f"SYS TOKEN: {cfg.experiment.environment.sys_token}, USR TOKEN: {cfg.experiment.environment.usr_token}, SEP TOKEN: {cfg.experiment.environment.sep_token}")
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
    
    eval_callback = None
    test_callback = None

    eval_logger = None
    test_logger = None
   
    
    # if "validation" in cfg.experiment and not isinstance(cfg.experiment.validation, type(None)):  
    #   if not os.path.exists("./.evaluation/eval"):
    #     os.makedirs('./.evaluation/eval')
    #     val_data, cache, state_encoding, val_env = setup_data_and_vecenv(device=cfg.experiment.device, dataset_cfg=cfg.experiment.validation.dataset, environment_cfg=cfg.experiment.environment,
    #                                                                     mode="eval", n_envs=cfg.experiment.environment.num_val_envs, log_dir=f"./.evaluation/eval",
    #                                                                     cache=cache, encoding=state_encoding,
    #                                                                     state_config=cfg.experiment.state, action_config=cfg.experiment.actions,
    #                                                                     torch_compile=cfg.experiment.torch_compile,
    #                                                                     save_terminal_obs=cfg.experiment.algorithm.dqn.save_terminal_obs,
    #                                                                     noise=cfg.experiment.validation.noise)
    #     eval_callback = CustomEvalCallback(eval_env=val_env, mode='eval',
    #                          best_model_save_path=f"./.evaluation",
    #                          log_path=f"./.evaluation/eval",
    #                          eval_freq=1,
    #                          deterministic=True, 
    #                          render=False,
    #                          n_eval_episodes=cfg.experiment.validation.dialogs,
    #                          keep_checkpoints=0)
    if "testing" in cfg.experiment and not isinstance(cfg.experiment.testing, type(None)):
        if not os.path.exists("./.evaluation/test"):
            os.makedirs('./.evaluation/test')
        test_logger = logging.getLogger("chat")
        test_logger.record = lambda key, value, **kwargs : test_logger.info(f"{key}: {value}")
        test_logger.dump = lambda *args, **kwargs: print(args)
        test_logger.setLevel(logging.INFO)
        chat_log_file_handler = logging.FileHandler("./.evaluation/test/stats.txt")
        chat_log_file_handler.setLevel(logging.INFO)
        test_logger.addHandler(chat_log_file_handler)
        test_data, cache, state_encoding, test_env = setup_data_and_vecenv(device=cfg.experiment.device, dataset_cfg=cfg.experiment.testing.dataset, environment_cfg=cfg.experiment.environment,
                                                                        mode="test", n_envs=cfg.experiment.environment.num_test_envs, log_dir=f"./.evaluation/test",
                                                                        cache=cache, encoding=state_encoding,
                                                                        state_config=cfg.experiment.state, action_config=cfg.experiment.actions,
                                                                        torch_compile=cfg.experiment.torch_compile,
                                                                        save_terminal_obs=cfg.experiment.algorithm.dqn.save_terminal_obs,
                                                                        noise=cfg.experiment.testing.noise)
        test_callback = CustomEvalCallback(eval_env=test_env, mode='test',
                        best_model_save_path=f"./.evaluation/test",
                        log_path=f"./.evaluation/test",
                        eval_freq=1,
                        deterministic=True,
                        render=False,
                        n_eval_episodes=cfg.experiment.testing.dialogs,
                        keep_checkpoints=0)
        test_callback.logger = test_logger
    
    # trainer = instantiate(cfg.experiment)
    INSTANCES[InstanceType.STATE_ENCODING] = state_encoding

    net_arch = OmegaConf.to_container(cfg.experiment.policy.net_arch)
    net_arch['state_dims'] = state_encoding.space_dims # patch arguments

    optim = OmegaConf.to_container(cfg.experiment.optimizer)
    lr = optim.pop("lr")
    
    optim_class = to_class(optim.pop('class_path'))
    print("Optim ARGS:", optim_class, optim)
    policy_kwargs = {
        "activation_fn": to_class(cfg.experiment.policy.activation_fn),   
        "net_arch": net_arch,
        "torch_compile": cfg.experiment.torch_compile,
        "optimizer_class": optim_class,
        "optimizer_kwargs": optim
    }
    replay_buffer_kwargs = {
        "num_train_envs": cfg.experiment.environment.num_train_envs,
        "batch_size": cfg.experiment.algorithm.dqn.batch_size,
        "dataset": train_data,
        "append_ask_action": False,
        # "state_encoding": state_encoding,
        "auto_skip": AutoSkipMode.NONE,
        "normalize_rewards": True,
        "stop_when_reaching_goal": cfg.experiment.environment.stop_when_reaching_goal,
        "stop_on_invalid_skip": cfg.experiment.environment.stop_on_invalid_skip,
        "max_steps": cfg.experiment.environment.max_steps,
        "user_patience": cfg.experiment.environment.user_patience,
        "sys_token": cfg.experiment.environment.sys_token,
        "usr_token": cfg.experiment.environment.usr_token,
        "sep_token": cfg.experiment.environment.sep_token,
        "alpha": cfg.experiment.algorithm.dqn.buffer.backend.alpha,
        "beta": cfg.experiment.algorithm.dqn.buffer.backend.beta,
        "use_lap": cfg.experiment.algorithm.dqn.buffer.backend.use_lap,
        "noise": 0.0
    }
    replay_buffer_class = CustomReplayBuffer
    dqn_target_cls =  to_class(cfg.experiment.algorithm.dqn.targets._target_)
    dqn_target_args = {'gamma': cfg.experiment.algorithm.dqn.gamma}
    dqn_target_args.update(cfg.experiment.algorithm.dqn.targets) 
    model = CustomDQN(
                policy=to_class(cfg.experiment.policy._target_), policy_kwargs=policy_kwargs,
                target=dqn_target_cls(**dqn_target_args),
                seed=cfg.experiment.seed,
                env=test_env if test_callback else val_env, 
                batch_size=cfg.experiment.algorithm.dqn.batch_size,
                verbose=1, device=cfg.experiment.device,  
                learning_rate=lr, 
                exploration_initial_eps=cfg.experiment.algorithm.dqn.eps_start, exploration_final_eps=cfg.experiment.algorithm.dqn.eps_end, exploration_fraction=cfg.experiment.algorithm.dqn.exploration_fraction,
                buffer_size=512, 
                learning_starts=cfg.experiment.algorithm.dqn.warmup_turns,
                gamma=cfg.experiment.algorithm.dqn.gamma,
                train_freq=1, # how many rollouts to perform before training once (one rollout = num_train_envs steps)
                gradient_steps=max(cfg.experiment.environment.num_train_envs // cfg.experiment.training.every_steps, 1),
                target_update_interval=cfg.experiment.algorithm.dqn.target_network_update_frequency * cfg.experiment.environment.num_train_envs,
                max_grad_norm=cfg.experiment.algorithm.dqn.max_grad_norm,
                tensorboard_log=f"runs/{run_id}",
                replay_buffer_class=replay_buffer_class,
                optimize_memory_usage=False,
                replay_buffer_kwargs=replay_buffer_kwargs,
                action_masking=cfg.experiment.actions.action_masking,
                actions_in_state_space=cfg.experiment.actions.in_state_space
            ) # TODO configure replay buffer class!
    print("INIT COMPLETE!")
    # load weights
    print("Loading weights...")
    model.policy.load_state_dict(th.load(f"{ckpt_dir}/policy.pth", map_location=lambda storage, loc: storage.cuda(0)))
    model.set_random_seed(12345)

    assert isinstance(cfg.experiment.logging.wandb_log, WandbLogLevel)
    # if cfg.experiment.logging.wandb_log != WandbLogLevel.NONE:
    #     if cfg.experiment.logging.wandb_log == WandbLogLevel.OFFLINE:
    #         os.environ['WANDB_MODE'] = 'offline'
    #     run = wandb.init(
    #         project="cts_en_stablebaselines",
    #         config=OmegaConf.to_container(cfg, resolve=True),
    #         sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
    #         # monitor_gym=True,  # auto-upload the videos of agents playing the game
    #         save_code=False,  # optional
    #         dir=f"./.evaluation",
    #     )


    # evaluate 
    model._logger = configure_logger(verbose=1, tensorboard_log="./.evaluation", tb_log_name="DQN", reset_num_timesteps=True)
    model.policy.set_training_mode(False)
    model.policy.eval()

    if eval_callback:
        print("EVALUATING....")
        eval_callback.model = model
        eval_callback.manual_call()
    if test_callback:
        print("TESTING...")
        test_callback.model = model
        test_callback.manual_call()
    
    # run.finish()



if __name__ == "__main__":
    load_cfg()