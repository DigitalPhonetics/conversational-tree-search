from collections import defaultdict
from statistics import mean
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import gymnasium as gym
import numpy as np
import torch as th

from stable_baselines3.common import type_aliases
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, VecMonitor, is_vecenv_wrapped

from utils.utils import EnvInfo


def custom_evaluate_policy(
    model: "type_aliases.PolicyPredictor",
    env: Union[gym.Env, VecEnv],
    n_eval_episodes: int = 10,
    deterministic: bool = True,
    render: bool = False,
    callback: Optional[Callable[[Dict[str, Any], Dict[str, Any]], None]] = None,
    reward_threshold: Optional[float] = None,
    return_episode_rewards: bool = False,
    warn: bool = True,
) -> Union[Tuple[float, float], Tuple[List[float], List[int]]]:
    """
    Runs policy for ``n_eval_episodes`` episodes and returns average reward.
    If a vector env is passed in, this divides the episodes to evaluate onto the
    different elements of the vector env. This static division of work is done to
    remove bias. See https://github.com/DLR-RM/stable-baselines3/issues/402 for more
    details and discussion.

    .. note::
        If environment has not been wrapped with ``Monitor`` wrapper, reward and
        episode lengths are counted as it appears with ``env.step`` calls. If
        the environment contains wrappers that modify rewards or episode lengths
        (e.g. reward scaling, early episode reset), these will affect the evaluation
        results as well. You can avoid this by wrapping environment with ``Monitor``
        wrapper before anything else.

    :param model: The RL agent you want to evaluate. This can be any object
        that implements a `predict` method, such as an RL algorithm (``BaseAlgorithm``)
        or policy (``BasePolicy``).
    :param env: The gym environment or ``VecEnv`` environment.
    :param n_eval_episodes: Number of episode to evaluate the agent
    :param deterministic: Whether to use deterministic or stochastic actions
    :param render: Whether to render the environment or not
    :param callback: callback function to do additional checks,
        called after each step. Gets locals() and globals() passed as parameters.
    :param reward_threshold: Minimum expected reward per episode,
        this will raise an error if the performance is not met
    :param return_episode_rewards: If True, a list of rewards and episode lengths
        per episode will be returned instead of the mean.
    :param warn: If True (default), warns user about lack of a Monitor wrapper in the
        evaluation environment.
    :return: Mean reward per episode, std of reward per episode.
        Returns ([float], [int]) when ``return_episode_rewards`` is True, first
        list containing per-episode rewards and second containing per-episode lengths
        (in number of steps).
    """
    is_monitor_wrapped = False
    # Avoid circular import
    from stable_baselines3.common.monitor import Monitor

    if not isinstance(env, VecEnv):
        env = DummyVecEnv([lambda: env])  # type: ignore[list-item, return-value]

    is_monitor_wrapped = is_vecenv_wrapped(env, VecMonitor) or env.env_is_wrapped(Monitor)[0]

    if not is_monitor_wrapped and warn:
        warnings.warn(
            "Evaluation environment is not wrapped with a ``Monitor`` wrapper. "
            "This may result in reporting modified episode lengths and rewards, if other wrappers happen to modify these. "
            "Consider wrapping environment first with ``Monitor`` wrapper.",
            UserWarning,
        )

    n_envs = env.num_envs
    
    dialog_log = []
    dialog_log_end_ptrs_free = [0] * n_envs
    dialog_log_end_ptrs_guided = [0] * n_envs

    episode_rewards_free = []
    episode_rewards_guided = []
    episode_lengths_free = []
    episode_lengths_guided = []
    percieved_lengths_free = []
    percieved_lengths_guided = []

    total_dialogs = 0
    free_dialogs = 0
    guided_dialogs = 0

    intent_accuracies = []
    intent_consistencies = []
    intent_episode_log = defaultdict(list)
    intent_tp = 0
    intent_fp = 0
    # intent_tn = 0
    intent_fn = 0

    episode_counts = np.zeros(n_envs, dtype="int")
    # Divides episodes among different sub environments in the vector as evenly as possible
    episode_count_targets = np.array([(n_eval_episodes + i) // n_envs for i in range(n_envs)], dtype="int")

    current_rewards = np.zeros(n_envs)
    current_lengths = np.zeros(n_envs, dtype="int")
    observations = env.reset()
    episode_starts = np.ones((env.num_envs,), dtype=bool)
    while (episode_counts < episode_count_targets).any():
        # NOTE: we abuse 'state' return from model.predict and return intent logits instead!
        actions, intent_classes = model.predict(
            observations,  # type: ignore[arg-type]
            state=None,
            episode_start=episode_starts,
            deterministic=deterministic,
        )     
        new_observations, rewards, dones, infos = env.step(actions)
        current_rewards += rewards
        current_lengths += 1
        for i in range(n_envs):
            if episode_counts[i] < episode_count_targets[i]:
                # unpack values so that the callback can access the local variables
                reward = rewards[i]
                done = dones[i]
                info = infos[i]
                episode_starts[i] = done

                if callback is not None:
                    callback(locals(), globals())

                # record intent accuracy and consistency
                if th.is_tensor(intent_classes):
                    # intent accuracy
                    env.envs[i].active_env.episode_log.append(f"{env.envs[i].active_env.env_id}-{env.envs[i].active_env.current_episode}$ -> PREDICTED INTENT: {'FAQ' if intent_classes[i].item() == 1 else 'GUIDED'}")
                    if info[EnvInfo.IS_FAQ] == True:
                        if intent_classes[i].item() == True:
                            intent_tp += 1
                        else:
                            intent_fn += 1
                    elif info[EnvInfo.IS_FAQ] == False:
                        if intent_classes[i].item() == True:
                            intent_fp += 1
                        # else:
                        #     intent_tn += 1
                    intent_accuracy = float(intent_classes[i].item() == info[EnvInfo.IS_FAQ])
                    intent_accuracies.append(intent_accuracy)
                    # intent consistency
                    intent_episode_log[i].append(intent_accuracy)
                    if done:
                        intent_consistencies.append(mean(intent_episode_log[i]))
                        # reset consistency for i-th env
                        intent_episode_log[i] = []
                if dones[i]:
                    # record env mode: free or guided
                    total_dialogs += 1

                    if info[EnvInfo.IS_FAQ]:
                        free_dialogs += 1
                        episode_rewards_free.append(current_rewards[i])
                        episode_lengths_free.append(current_lengths[i])
                        percieved_lengths_free.append(info[EnvInfo.PERCIEVED_LENGTH])
                        dialog_log_end_ptrs_free[i] = env.envs[i].last_log_end_ptr_free
                    else:
                        guided_dialogs += 1
                        episode_rewards_guided.append(current_rewards[i])
                        episode_lengths_guided.append(current_lengths[i])
                        percieved_lengths_guided.append(info[EnvInfo.PERCIEVED_LENGTH])
                        dialog_log_end_ptrs_guided[i] = env.envs[i].last_log_end_ptr_guided
                    episode_counts[i] += 1
                    current_rewards[i] = 0
                    current_lengths[i] = 0

        observations = new_observations
        if render:
            env.render()
    
    for idx, sub_env in enumerate(env.envs):
        if hasattr(sub_env, "free_env"):
            dialog_log.extend(sub_env.free_env.episode_log[:dialog_log_end_ptrs_free[idx]])
        if hasattr(sub_env, "guided_env"):
            dialog_log.extend(sub_env.guided_env.episode_log[:dialog_log_end_ptrs_guided[idx]])
        sub_env.reset_episode_log()

    free_dialogs = free_dialogs / total_dialogs
    guided_dialogs = guided_dialogs / total_dialogs
    
    if len(intent_accuracies) > 0:
        intent_accuracies = mean(intent_accuracies)
        intent_consistencies = mean(intent_consistencies)
        intent_tp = intent_tp * 2.0
        intent_f1 = intent_tp / (intent_tp + intent_fp + intent_fn)
    else:
        intent_accuracies = None
        intent_consistencies = None
        intent_f1 = None

    return episode_rewards_free, episode_rewards_guided, episode_lengths_free, episode_lengths_guided, percieved_lengths_free, percieved_lengths_guided, intent_f1, intent_accuracies, intent_consistencies, free_dialogs, guided_dialogs, dialog_log
