
from typing import Tuple, Union, Optional, Dict, Any

import io
import os
import pathlib
import shutil

import gymnasium as gym
from omegaconf import DictConfig, OmegaConf
import torch
from algorithm.dqn.buffer import CustomReplayBuffer
from algorithm.dqn.her import HindsightExperienceReplayWrapper
from config import ActiveLearningConfig, DialogLogLevel, WandbLogLevel
from data.cache import Cache
from data.dataset import GraphDataset
from encoding.state import StateEncoding
from hydra import compose, initialize
from gymnasium import Env

from utils.utils import AutoSkipMode, get_git_branch, get_git_commit_hash, save_git_patch, to_class

from datetime import datetime
from stat import S_IFREG
from stream_zip import ZIP_64, stream_zip

import stable_baselines3 as sb3
from stable_baselines3.common.save_util import data_to_json
from stable_baselines3.common.utils import get_system_info

import torch as th

def _local_files(names):
    now  = datetime.now()
    def contents(name):
        with open(name, 'rb') as f:
            while chunk := f.read(65536):
                yield chunk

    return (
        (name, now, S_IFREG | 0o600, ZIP_64, contents(name))
        for name in names
    )
    
def save_config(save_path: Union[str, pathlib.Path, io.BufferedIOBase],
                omegaconf: DictConfig):
    # should do this immediately after starting run... same for config!! -> create one initial copy, then copy to zip file

    with open(f"{save_path}/config.yaml", "w") as f:
        # save config
        OmegaConf.save(config=omegaconf, f=f)
    
    with open(f"{save_path}/info.txt", "w") as f:
        # save sb3 info and git info
        f.write(get_system_info(print_info=False)[1])
        f.write(f"\nstable_baselines3_version: {sb3.__version__}")
        f.write(f"\nbranch: {get_git_branch()}")
        f.write(f"\ncommit: {get_git_commit_hash()}")
    save_git_patch(patch_file=f"{save_path}/git_patch.patch") 

        

def save_to_zip_file(
    save_path: Union[str, pathlib.Path, io.BufferedIOBase],
    params: Optional[Dict[str, Any]] = None,
    replay_buffer = None,
    verbose: int = 0,
) -> None:
    """
    Save model data to a zip archive.

    :param save_path: Where to store the model.
        if save_path is a str or pathlib.Path ensures that the path actually exists.
    :param data: Class parameters being stored (non-PyTorch variables)
    :param params: Model parameters being stored expected to contain an entry for every
                   state_dict with its name and the state_dict.
    :param pytorch_variables: Other PyTorch variables expected to contain name and value of the variable.
    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    """

    # First, write individual files to disk

    # data/params can be None, so do not
    # try to serialize them blindly
    os.makedirs(save_path, exist_ok=True)

    base_path = os.path.join(save_path, '..', '..', '..')
    filenames = [f"{base_path}/config.yaml", f"{base_path}/info.txt", f"{base_path}/git_patch.patch"] # add git infos and config to checkpoint

    for file_name, dict_ in params.items():
        # model weights
        f_param = f"{save_path}/{file_name}.pth"
        th.save(dict_, f_param)
        filenames.append(f_param)
    
    # Create a zip-archive and write our objects there.
    with open(f"{save_path}.zip", 'wb') as f:
        for chunk in stream_zip(_local_files(filenames)):
            f.write(chunk)
    
    if replay_buffer is not None:
        th.save(replay_buffer.save_params(), f"{base_path}/replay_buffer_tmp.pth")
            
    # Cleanup
    shutil.rmtree(save_path)
    if replay_buffer is not None:
        if os.path.isfile(f"{base_path}/replay_buffer.pth"):
            # delete old buffer 
            os.remove(f"{base_path}/replay_buffer.pth")
        # keep current replay buffer
        shutil.move(src=f"{base_path}/replay_buffer_tmp.pth", dst=f"{base_path}/replay_buffer.pth")
  

def load_model(dqn_cls, ckpt_path: str, cfg_name: str, device: str, data: GraphDataset, cfg_path = "./conf/") -> Tuple[DictConfig, "CustomDQN", StateEncoding]:
    # load config
    with initialize(version_base=None, config_path="../" + cfg_path):
        # parse config
        print("Parsing config...")
        cfg = compose(config_name=cfg_name)
        # print(OmegaConf.to_yaml(cfg))

        # disable logging
        cfg.experiment.logging.dialog_log = DialogLogLevel.NONE
        cfg.experiment.logging.wandb_log = WandbLogLevel.NONE
        cfg.experiment.logging.log_interval = 9999999
        cfg.experiment.logging.keep_checkpoints = 9

        # load encodings
        print("Loading encodings...")
        state_cfg = cfg.experiment.state
        action_cfg = cfg.experiment.actions
        cache = Cache(device=device, data=data, state_config=state_cfg, torch_compile=False)
        encoding = StateEncoding(cache=cache, state_config=state_cfg, action_config=action_cfg, data=data)

        # setup spaces
        action_space = gym.spaces.Discrete(encoding.space_dims.num_actions)
        if encoding.action_config.in_state_space == True:
            # state space: max. node degree (#actions) x state dim
            observation_space = gym.spaces.Box(low=float('-inf'), high=float('inf'), shape=(encoding.space_dims.num_actions, encoding.space_dims.state_vector,)) #, dtype=np.float32)
        else:
            observation_space = gym.spaces.Box(low=float('-inf'), high=float('inf'), shape=(encoding.space_dims.state_vector,)) #, dtype=np.float32)

        class CustomEnv(Env):
            def __init__(self, observation_space, action_space) -> None:
                self.observation_space = observation_space
                self.action_space = action_space
        dummy_env = CustomEnv(observation_space=observation_space, action_space=action_space)

        # setup model
        print("Setting up model...")
        net_arch = OmegaConf.to_container(cfg.experiment.policy.net_arch)
        net_arch['state_dims'] = encoding.space_dims # patch arguments
        optim = OmegaConf.to_container(cfg.experiment.optimizer)
        optim_class = to_class(optim.pop('class_path'))
        lr = optim.pop('lr')
        print("Optim ARGS:", optim_class, lr, optim)


        if "num_prediction_heads" in cfg.experiment.policy.net_arch:
            num_prediction_heads = cfg.experiment.policy.net_arch.num_prediction_heads
        else:
            num_prediction_heads = 1
        if "bernoulli_p_train_heads" in cfg.experiment.policy.net_arch:
            bernoulli_p_train_heads =  cfg.experiment.policy.net_arch.bernoulli_p_train_heads
        else:
            bernoulli_p_train_heads = 1
        if "noise_std_init" in cfg.experiment.policy.net_arch:
            noise_std_init = cfg.experiment.policy.net_arch.noise_std_init
        else:
            noise_std_init = 0

        active_learning_args =  cfg.experiment.algorithm.active_learning if 'active_learning' in cfg.experiment.algorithm else ActiveLearningConfig(active=False)
        
        policy_kwargs = {
            "activation_fn": to_class(cfg.experiment.policy.activation_fn),   
            "net_arch": net_arch,
            "torch_compile": cfg.experiment.torch_compile,
            "optimizer_class": optim_class,
            "optimizer_kwargs": optim
        }

        print("Replay buffer kwargs")
        replay_buffer_kwargs = {
            "num_train_envs": cfg.experiment.environment.num_train_envs,
            "batch_size": cfg.experiment.algorithm.dqn.batch_size,
            "dataset": data,
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
            "noise": cfg.experiment.training.noise,
            "num_prediction_heads": num_prediction_heads,
            "bernoulli_p_train_heads": bernoulli_p_train_heads
        }
        replay_buffer_class = HindsightExperienceReplayWrapper
        dqn_target_cls =  to_class(cfg.experiment.algorithm.dqn.targets._target_)
        dqn_target_args = {'gamma': cfg.experiment.algorithm.dqn.gamma}
        dqn_target_args.update(cfg.experiment.algorithm.dqn.targets) 
        print("Create model instance...")
        model = dqn_cls(policy=to_class(cfg.experiment.policy._target_), policy_kwargs=policy_kwargs,
                    target=dqn_target_cls(**dqn_target_args),
                    seed=cfg.experiment.seed,
                    env=dummy_env, 
                    batch_size=cfg.experiment.algorithm.dqn.batch_size,
                    verbose=1, device=cfg.experiment.device,  
                    learning_rate=lr, 
                    exploration_initial_eps=cfg.experiment.algorithm.dqn.eps_start, exploration_final_eps=cfg.experiment.algorithm.dqn.eps_end, exploration_fraction=cfg.experiment.algorithm.dqn.exploration_fraction,
                    buffer_size=1, # we don't need to store experience, will only increase RAM usage 
                    learning_starts=cfg.experiment.algorithm.dqn.warmup_turns,
                    gamma=cfg.experiment.algorithm.dqn.gamma,
                    train_freq=1, # how many rollouts to perform before training once (one rollout = num_train_envs steps)
                    gradient_steps=max(cfg.experiment.environment.num_train_envs // cfg.experiment.training.every_steps, 1),
                    target_update_interval=cfg.experiment.algorithm.dqn.target_network_update_frequency * cfg.experiment.environment.num_train_envs,
                    max_grad_norm=cfg.experiment.algorithm.dqn.max_grad_norm,
                    tensorboard_log=None,
                    replay_buffer_class=replay_buffer_class,
                    optimize_memory_usage=False,
                    replay_buffer_kwargs=replay_buffer_kwargs,
                    action_masking=cfg.experiment.actions.action_masking,
                    actions_in_state_space=cfg.experiment.actions.in_state_space,
                    num_prediction_heads=num_prediction_heads,
                    noise_std_init=noise_std_init,
                    active_learning_args=active_learning_args
                )
        
        # restore weights
        print("Restoring weights...")
        ckpt_params = torch.load(f"{ckpt_path}/policy.pth", map_location=device)
        model.policy.load_state_dict(ckpt_params)
        model.policy.set_training_mode(False)
        model.policy.eval()
    return cfg, model, encoding
