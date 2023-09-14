import itertools
from statistics import mean
import warnings
from copy import deepcopy
from typing import Any, Callable, DefaultDict, Dict, List, Optional, Sequence, Type, Union

import gymnasium as gym
import numpy as np
import torch as th

from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvIndices, VecEnvObs, VecEnvStepReturn
from stable_baselines3.common.vec_env.patch_gym import _patch_env
from stable_baselines3.common.vec_env.util import obs_space_info

from encoding.state import StateEncoding
from environment.cts import CTSEnvironment


class CustomVecEnv(VecEnv):
    """
    Creates a simple vectorized wrapper for multiple environments, calling each environment in sequence on the current
    Python process. This is useful for computationally simple environment such as ``Cartpole-v1``,
    as the overhead of multiprocess or multithread outweighs the environment computation time.
    This can also be used for RL methods that
    require a vectorized environment, but that you want a single environments to train with.

    :param env_fns: a list of functions
        that return environments to vectorize
    :raises ValueError: If the same environment instance is passed as the output of two or more different env_fn.
    """

    actions: np.ndarray

    def __init__(self, env_fns: List[Callable[[], gym.Env]],
                 state_encoding: StateEncoding,
                 sys_token: str,
                 usr_token: str,
                 sep_token: str,
                 save_terminal_obs: bool
                 ):
        self.envs: List[CTSEnvironment] = [_patch_env(fn()) for fn in env_fns]
        self.sys_token = sys_token
        self.usr_token = usr_token
        self.sep_token = sep_token
        self.save_terminal_obs = save_terminal_obs
        print("SAVE TERMINAL OBS:", self.save_terminal_obs)

        self.state_encoding = state_encoding
        # setup state space info
        self.action_space = gym.spaces.Discrete(state_encoding.space_dims.num_actions)
        if state_encoding.action_config.in_state_space == True:
            # state space: max. node degree (#actions) x state dim
            self.observation_space = gym.spaces.Box(low=float('-inf'), high=float('inf'), shape=(state_encoding.space_dims.num_actions, state_encoding.space_dims.state_vector,)) #, dtype=np.float32)
        else:
            self.observation_space = gym.spaces.Box(low=float('-inf'), high=float('inf'), shape=(state_encoding.space_dims.state_vector,)) #, dtype=np.float32)

        VecEnv.__init__(self, len(env_fns), self.observation_space, self.action_space)
        self.keys, shapes, dtypes = obs_space_info(self.observation_space)

        self.buf_obs: List[Dict[str, Any]] = [{} for _ in range(self.num_envs)]
        self.buf_dones = np.zeros((self.num_envs,), dtype=bool)
        self.buf_rews = np.zeros((self.num_envs,), dtype=np.float32)
        self.buf_infos: List[Dict[str, Any]] = [{} for _ in range(self.num_envs)]
        self.metadata = {}

    def step_async(self, actions: np.ndarray) -> None:
        self.actions = actions

    def step_wait(self) -> VecEnvStepReturn:
        for env_idx in range(self.num_envs):
            obs, self.buf_rews[env_idx], terminated, truncated, self.buf_infos[env_idx] = self.envs[env_idx].step(
                self.actions[env_idx]
            )
            self.buf_infos[env_idx] = obs
            # convert to SB3 VecEnv api
            self.buf_dones[env_idx] = terminated
            # See https://github.com/openai/gym/issues/3102
            # Gym 0.26 introduces a breaking change
            self.buf_infos[env_idx]["TimeLimit.truncated"] = False

            if self.buf_dones[env_idx]:
                # save final observation where user can get it, then reset
                if self.save_terminal_obs:
                    self.buf_infos[env_idx]["terminal_observation"] = deepcopy(obs)
                # obs, self.reset_infos[env_idx] = self.envs[env_idx].reset()
                obs = self.envs[env_idx].reset()
            self._save_obs(env_idx, obs)

        # batch encode!
        return (self._obs_from_buf(), np.copy(self.buf_rews), np.copy(self.buf_dones), deepcopy(self.buf_infos))


    def seed(self, seed: Optional[int] = None) -> Sequence[Union[None, int]]:
        # Avoid circular import
        from stable_baselines3.common.utils import compat_gym_seed

        if seed is None:
            seed = np.random.randint(0, 2**32 - 1)
        seeds = []
        for idx, env in enumerate(self.envs):
            seeds.append(compat_gym_seed(env, seed=seed + idx))  # type: ignore[func-returns-value]
        return seeds


    def reset(self) -> VecEnvObs:
        for env_idx in range(self.num_envs):
            obs = self.envs[env_idx].reset()
            self._save_obs(env_idx, obs)
        return self._obs_from_buf()


    def close(self) -> None:
        for env in self.envs:
            env.close()


    def get_images(self) -> Sequence[Optional[np.ndarray]]:
        if self.render_mode != "rgb_array":
            warnings.warn(
                f"The render mode is {self.render_mode}, but this method assumes it is `rgb_array` to obtain images."
            )
            return [None for _ in self.envs]
        return [env.render() for env in self.envs]  # type: ignore[misc]


    def render(self, mode: Optional[str] = None) -> Optional[np.ndarray]:
        """
        Gym environment rendering. If there are multiple environments then
        they are tiled together in one image via ``BaseVecEnv.render()``.

        :param mode: The rendering type.
        """
        return super().render(mode=mode)


    def _save_obs(self, env_idx: int, obs: VecEnvObs) -> None:
        self.buf_obs[env_idx] = obs

    @th.no_grad()
    def _obs_from_buf(self) -> VecEnvObs:
        batch_encoding = self.state_encoding.batch_encode(self.buf_obs, sys_token=self.sys_token, usr_token=self.usr_token, sep_token=self.sep_token)
        # convert all terminal observations into vectors (found in buf_infos['terminal_observations']) because stable-baselines resets done environments immediately in step()
        terminal_observation_indices = [env_idx for env_idx, info in enumerate(self.buf_infos) if (not isinstance(info, type(None))) and "terminal_observation" in info]
        if len(terminal_observation_indices) > 0:
            batch_encoding_terminal_obs = self.state_encoding.batch_encode([self.buf_infos[env_idx]['terminal_observation'] for env_idx in terminal_observation_indices], sys_token=self.sys_token, usr_token=self.usr_token, sep_token=self.sep_token)
            for list_idx, env_idx in enumerate(terminal_observation_indices):
                self.buf_infos[env_idx]['terminal_observation'] = batch_encoding_terminal_obs[list_idx]
        
        return batch_encoding
    
    @property 
    def current_episode(self) -> int:
        return sum([env.current_episode for env in self.envs])
    
    @property
    def turn_counter(self) -> int:
        return sum([env.turn_counter for env in self.envs])

    def get_attr(self, attr_name: str, indices: VecEnvIndices = None) -> List[Any]:
        """Return attribute from vectorized environment (see base class)."""
        target_envs = self._get_target_envs(indices)
        return [getattr(env_i, attr_name) for env_i in target_envs]


    def set_attr(self, attr_name: str, value: Any, indices: VecEnvIndices = None) -> None:
        """Set attribute inside vectorized environments (see base class)."""
        target_envs = self._get_target_envs(indices)
        for env_i in target_envs:
            setattr(env_i, attr_name, value)


    def env_method(self, method_name: str, *method_args, indices: VecEnvIndices = None, **method_kwargs) -> List[Any]:
        """Call instance methods of vectorized environments."""
        target_envs = self._get_target_envs(indices)
        return [getattr(env_i, method_name)(*method_args, **method_kwargs) for env_i in target_envs]


    def env_is_wrapped(self, wrapper_class: Type[gym.Wrapper], indices: VecEnvIndices = None) -> List[bool]:
        """Check if worker environments are wrapped with a given wrapper"""
        target_envs = self._get_target_envs(indices)
        # Import here to avoid a circular import
        from stable_baselines3.common import env_util

        return [env_util.is_wrapped(env_i, wrapper_class) for env_i in target_envs]


    def _get_target_envs(self, indices: VecEnvIndices) -> List[gym.Env]:
        indices = self._get_indices(indices)
        return [self.envs[i] for i in indices]

    def stats_asked_goals_free(self):
        values = list(itertools.chain(*[env.free_env.asked_goals for env in self.envs if hasattr(env, "free_env")]))
        if len(values) == 0:
            return 0.0
        return mean(values)

    def stats_asked_goals_guided(self):
        values = list(itertools.chain(*[env.guided_env.asked_goals for env in self.envs if hasattr(env, "guided_env")]))
        if len(values) == 0:
            return 0.0
        return mean(values)

    def stats_reached_goals_free(self):
        values = list(itertools.chain(*[env.free_env.reached_goals for env in self.envs if hasattr(env, "free_env")]))
        if len(values) == 0:
            return 0.0
        return mean(values)

    def stats_reached_goals_guided(self):
        values = list(itertools.chain(*[env.guided_env.reached_goals for env in self.envs if hasattr(env, "guided_env")]))
        if len(values) == 0:
            return 0.0
        return mean(values)
    
    def stats_goal_node_coverage_free(self):
        coverage_dicts = [env.free_env.goal_node_coverage for env in self.envs if hasattr(env, "free_env")]
        coverage_dict = self._join_dicts(coverage_dicts)
        return len(coverage_dict) / self.envs[0].data.num_free_goal_nodes

    def stats_goal_node_coverage_guided(self):
        coverage_dicts = [env.guided_env.goal_node_coverage for env in self.envs if hasattr(env, "guided_env")]
        coverage_dict = self._join_dicts(coverage_dicts)
        return len(coverage_dict) / self.envs[0].data.num_guided_goal_nodes
    
    def _join_dicts(self, dicts: List[DefaultDict[str, float]]) -> Dict[str, float]:
        """
        Sums the values from each dict by key
        """
        result = {}
        keys = set(itertools.chain(*[d.keys() for d in dicts]))
        for key in keys:
            result[key] = sum([d[key] for d in dicts])
        return result

    def stats_node_coverage(self):
        coverage_dicts = []
        for env in self.envs:
            if hasattr(env, "free_env"):
                coverage_dicts.append(env.free_env.node_coverage)
            if hasattr(env, "guided_env"): 
                coverage_dicts.append(env.guided_env.node_coverage)
        coverage_dict = self._join_dicts(coverage_dicts)
        return len(coverage_dict) / len(self.envs[0].data.node_list)
    
    def stats_synonym_coverage_questions(self):
        coverage_dict = self._join_dicts([env.free_env.coverage_question_synonyms for env in self.envs if hasattr(env, "free_env")])
        return len(coverage_dict) / len(self.envs[0].data.question_list)

    def stats_synonym_coverage_answers(self):
        coverage_dicts = []
        for env in self.envs:
            if hasattr(env, "guided_env"):
                coverage_dicts.append(env.guided_env.coverage_answer_synonyms)
            if hasattr(env, "free_env"):
                coverage_dicts.append(env.free_env.coverage_answer_synonyms)
        coverage_dict = self._join_dicts(coverage_dicts)
        return len(coverage_dict) / self.envs[0].data.num_answer_synonyms
    
    def stats_actioncount_skips_indvalid(self):
        count = 0
        for env in self.envs:
            if hasattr(env, "free_env"):
                count += env.free_env.actioncount_skip_invalid
            if hasattr(env, "guided_env"): 
                count += env.guided_env.actioncount_skip_invalid
        return count

    def stats_actioncount_ask_variable_irrelevant(self):
        count = 0
        for env in self.envs:
            if hasattr(env, "free_env"):
                count += env.free_env.actioncount_ask_variable_irrelevant
            if hasattr(env, "guided_env"): 
                count += env.guided_env.actioncount_ask_variable_irrelevant
        return count

    def stats_actioncount_ask_question_irrelevant(self):
        count = 0
        for env in self.envs:
            if hasattr(env, "free_env"):
                count += env.free_env.actioncount_ask_question_irrelevant
            if hasattr(env, "guided_env"): 
                count += env.guided_env.actioncount_ask_question_irrelevant
        return count

    def stats_actioncount_missingvariable(self):
        count = 0
        for env in self.envs:
            if hasattr(env, "free_env"):
                count += env.free_env.actioncount_missingvariable
            if hasattr(env, "guided_env"): 
                count += env.guided_env.actioncount_missingvariable
        return count
