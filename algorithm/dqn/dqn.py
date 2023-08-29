
from collections import deque
from statistics import mean
from typing import Tuple, TypeVar, Union, Dict, Optional, Type, Any
from stable_baselines3 import DQN

import torch as th
import torch.nn.functional as F

from gymnasium import spaces
from stable_baselines3.dqn.policies import QNetwork
from stable_baselines3.common.type_aliases import Schedule

from typing import Any, Dict, Optional, Tuple, Type, Union

import numpy as np
import torch as th
from gymnasium import spaces
from torch.nn import functional as F

from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.type_aliases import GymEnv, Schedule
from stable_baselines3.dqn.policies import QNetwork
from algorithm.dqn.her import HindsightExperienceReplayWrapper

from algorithm.dqn.policy import CustomDQNPolicy
from algorithm.dqn.targets import DQNTarget, StandardTarget
from environment.old.her import OldHindsightExperienceReplayWrapper
from utils.utils import EnvInfo
import config as cfg

SelfDQN = TypeVar("SelfDQN", bound="CustomDQN")


class CustomDQN(DQN):
    """
    Deep Q-Network (DQN)

    Paper: https://arxiv.org/abs/1312.5602, https://www.nature.com/articles/nature14236
    Default hyperparameters are taken from the Nature paper,
    except for the optimizer and learning rate that were taken from Stable Baselines defaults.

    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: The learning rate, it can be a function
        of the current progress remaining (from 1 to 0)
    :param buffer_size: size of the replay buffer
    :param learning_starts: how many steps of the model to collect transitions for before learning starts
    :param batch_size: Minibatch size for each gradient update
    :param tau: the soft update coefficient ("Polyak update", between 0 and 1) default 1 for hard update
    :param gamma: the discount factor
    :param train_freq: Update the model every ``train_freq`` steps. Alternatively pass a tuple of frequency and unit
        like ``(5, "step")`` or ``(2, "episode")``.
    :param gradient_steps: How many gradient steps to do after each rollout (see ``train_freq``)
        Set to ``-1`` means to do as many gradient steps as steps done in the environment
        during the rollout.
    :param replay_buffer_class: Replay buffer class to use (for instance ``HerReplayBuffer``).
        If ``None``, it will be automatically selected.
    :param replay_buffer_kwargs: Keyword arguments to pass to the replay buffer on creation.
    :param optimize_memory_usage: Enable a memory efficient variant of the replay buffer
        at a cost of more complexity.
        See https://github.com/DLR-RM/stable-baselines3/issues/37#issuecomment-637501195
    :param target_update_interval: update the target network every ``target_update_interval``
        environment steps.
    :param exploration_fraction: fraction of entire training period over which the exploration rate is reduced
    :param exploration_initial_eps: initial value of random action probability
    :param exploration_final_eps: final value of random action probability
    :param max_grad_norm: The maximum value for the gradient clipping
    :param stats_window_size: Window size for the rollout logging, specifying the number of episodes to average
        the reported success rate, mean episode length, and mean reward over
    :param tensorboard_log: the log location for tensorboard (if None, no logging)
    :param policy_kwargs: additional arguments to be passed to the policy on creation
    :param verbose: Verbosity level: 0 for no output, 1 for info messages (such as device or wrappers used), 2 for
        debug messages
    :param seed: Seed for the pseudo random generators
    :param device: Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
    :param _init_setup_model: Whether or not to build the network at the creation of the instance
    """

    # Linear schedule will be defined in `_setup_model()`
    exploration_schedule: Schedule
    q_net: QNetwork
    q_net_target: QNetwork
    policy: CustomDQNPolicy

    def __init__(
        self,
        policy: Union[str, Type[CustomDQNPolicy]],
        env: Union[GymEnv, str],
        target: DQNTarget,
        learning_rate: Union[float, Schedule] = 1e-4,
        buffer_size: int = 1_000_000,  # 1e6
        learning_starts: int = 50000,
        batch_size: int = 32,
        tau: float = 1.0,
        gamma: float = 0.99,
        train_freq: Union[int, Tuple[int, str]] = 4,
        gradient_steps: int = 1,
        replay_buffer_class: Optional[Type[ReplayBuffer]] = None,
        replay_buffer_kwargs: Optional[Dict[str, Any]] = None,
        optimize_memory_usage: bool = False,
        target_update_interval: int = 10000,
        exploration_fraction: float = 0.1,
        exploration_initial_eps: float = 1.0,
        exploration_final_eps: float = 0.05,
        max_grad_norm: float = 10,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
        action_masking: bool = False,
        actions_in_state_space: bool = False,
    ) -> None:
        super().__init__(
            policy=policy,
            env=env,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            learning_starts=learning_starts,
            batch_size=batch_size,
            tau=tau,
            gamma=gamma,
            train_freq=train_freq,
            gradient_steps=gradient_steps,
            replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=replay_buffer_kwargs,
            optimize_memory_usage=optimize_memory_usage,
            target_update_interval=target_update_interval,
            exploration_fraction=exploration_fraction,
            exploration_initial_eps=exploration_initial_eps,
            exploration_final_eps=exploration_final_eps,
            max_grad_norm=max_grad_norm,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            seed=seed,
            device=device,
            _init_setup_model=_init_setup_model
        )
        self.action_masking = action_masking
        self.actions_in_state_space = actions_in_state_space
        self.target = target

        self.global_step = 0
        self.current_resets = 0

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update learning rate according to schedule
        self._update_learning_rate(self.policy.optimizer)

        td_losses = []
        intent_losses = []
        q_values = []
        for _ in range(gradient_steps):
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)  # type: ignore[union-attr]

            # Get current Q-values estimates
            current_q_values = self.q_net(replay_data.observations)
            # Handle intent prediction loss
            loss = 0.0
            if self.policy.intent_prediction:
                # split output of network into q values and intents
                current_q_values, current_intent_logits = current_q_values
                intent_labels = th.tensor([info[EnvInfo.IS_FAQ] for info in replay_data.infos], dtype=th.float, device=self.device)
                intent_loss = self.q_net.intent_loss_weight * F.binary_cross_entropy_with_logits(current_intent_logits, intent_labels, reduction="none")
                loss += intent_loss
                intent_losses.append(intent_loss.mean(-1).item())

            with th.no_grad():
                # Compute the next Q-values using the target network
                next_q_values = self.q_net_target(replay_data.next_observations)
                if self.policy.intent_prediction:
                    # split output of network into q values, ignore intents (position 1)
                    next_q_values = next_q_values[0]
                target_q_values = self.target.target(next_q_values=next_q_values, data=replay_data, q_old=current_q_values).squeeze()

            
            # Retrieve the q-values for the actions from the replay buffer
            current_q_values = th.gather(current_q_values, dim=1, index=replay_data.actions.long()).squeeze()
            q_values.extend(current_q_values.view(-1).tolist())

            # Compute Huber loss (less sensitive to outliers)
            # td_loss = F.smooth_l1_loss(current_q_values, target_q_values)
            td_loss = F.huber_loss(current_q_values, target_q_values, reduction='none')
            td_losses.append(td_loss.mean(-1).item())
            loss += td_loss
            if "prioritized" in self.replay_buffer.__class__.__name__.lower() or "hindsight" in self.replay_buffer.__class__.__name__.lower():
                # weight loss by priority
                loss = loss * replay_data.weights
                # update priorities
                td_error = th.abs(target_q_values - current_q_values)
                self.replay_buffer.update_weights(replay_data.indices, td_error)
            loss = loss.mean(-1) # reduce loss

            # Optimize the policy
            self.policy.optimizer.zero_grad()
            loss.backward()
            # Clip gradient norm
            th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()

            self.global_step += 1
        # Reset last transition indices
        self.replay_buffer.reset_last_transition_indices()

        # Increase update counter
        self._n_updates += gradient_steps

        self.logger.record("train/train_counter", self._n_updates)
        self.logger.record("train/episode_counter", self.env.current_episode)
        self.logger.record("train/global_step", self.global_step)
        self.logger.record("train/turn_counter", self.env.turn_counter)
        self.logger.record("train/td_loss", np.mean(td_losses))
        self.logger.record("train/max_goal_distance", cfg.INSTANCES[cfg.InstanceArgs.MAX_DISTANCE])
        self.logger.record("train/buffer_size", len(self.replay_buffer))
        self.logger.record("train/q_values", mean(q_values))
        self.logger.record("train/epsilon", self.exploration_rate)
        self.logger.record("train/resets", self.current_resets)
        if self.policy.intent_prediction:
            self.logger.record("train/intent_loss", np.mean(intent_losses))
        if self.replay_buffer_class in [HindsightExperienceReplayWrapper, OldHindsightExperienceReplayWrapper]:
            self.logger.record("rollout/total_aritificial_episodes", self.replay_buffer.artificial_episodes)
            self.logger.record("rollout/her_mean_reward_free", self.replay_buffer.artificial_mean_episode_reward_free)
            self.logger.record("rollout/hear_mean_reward_guided", self.replay_buffer.artificial_mean_episode_reward_guided)
            self.logger.record("rollout/her_mean_success_free", self.replay_buffer.replay_success_mean_free)
            self.logger.record("rollout/her_mean_success_guided", self.replay_buffer.replay_success_mean_guided)
 
    def _draw_random_actions(self, observation: th.Tensor) -> np.ndarray:
        if self.policy.is_vectorized_observation(observation):
            if isinstance(observation, dict):
                n_batch = observation[list(observation.keys())[0]].shape[0]
            else:
                n_batch = observation.shape[0]
            mask = None
            if self.actions_in_state_space:
                mask = (~(observation.abs().sum(-1) == 0.0)).to(th.int8).numpy() # batch x actions
            action = np.array([self.action_space.sample(mask=mask[batch_idx]) for batch_idx in range(n_batch)])
        else:
            action = np.array(self.action_space.sample())
        return action

    def predict(
        self,
        observation: Union[np.ndarray, Dict[str, np.ndarray]],
        state: Optional[Tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        """
        Overrides the base_class predict function to include epsilon-greedy exploration.

        :param observation: the input observation
        :param state: The last states (can be None, used in recurrent policies)
        :param episode_start: The last masks (can be None, used in recurrent policies)
        :param deterministic: Whether or not to return deterministic actions.
        :return: the model's action and the intent instead of the state (we can abuse state return here since we don't have recurrent policies)
        """
        # TODO action masking for actions not in state space
        if not deterministic and np.random.rand() < self.exploration_rate:
            # exploration
            action = self._draw_random_actions(observation)
        else:
            # exploitation
            # NOTE: masking is already built-in if actions_in_state_space = True
            action, state, intent_classes = self.policy.predict(observation, state, episode_start, deterministic)
            return action, intent_classes # abuse state return since we don't have recurrent policies
        return action, None

    def _sample_action(
        self,
        learning_starts: int,
        action_noise = None,
        n_envs: int = 1,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample an action according to the exploration policy.
        This is either done by sampling the probability distribution of the policy,
        or sampling a random action (from a uniform distribution over the action space)
        or by adding noise to the deterministic output.

        :param action_noise: Action noise that will be used for exploration
            Required for deterministic policy (e.g. TD3). This can also be used
            in addition to the stochastic policy for SAC.
        :param learning_starts: Number of steps before learning for the warm-up phase.
        :param n_envs:
        :return: action to take in the environment
            and scaled action that will be stored in the replay buffer.
            The two differs when the action space is not normalized (bounds are not [-1, 1]).
        """
        # Select action randomly or according to policy
        if self.num_timesteps < learning_starts and not (self.use_sde and self.use_sde_at_warmup):
            # Warmup phase
            unscaled_action = self._draw_random_actions(self._last_obs) # FIX to use our masks
        else:
            # Note: when using continuous actions,
            # we assume that the policy uses tanh to scale the action
            # We use non-deterministic action in the case of SAC, for TD3, it does not matter
            unscaled_action, _ = self.predict(self._last_obs, deterministic=False)

        # Rescale the action from [low, high] to [-1, 1]
        if isinstance(self.action_space, spaces.Box):
            scaled_action = self.policy.scale_action(unscaled_action)

            # Add noise to the action (improve exploration)
            if action_noise is not None:
                scaled_action = np.clip(scaled_action + action_noise(), -1, 1)

            # We store the scaled action in the buffer
            buffer_action = scaled_action
            action = self.policy.unscale_action(scaled_action)
        else:
            # Discrete case, no need to normalize or clip
            buffer_action = unscaled_action
            action = buffer_action
        return action, buffer_action
    
    def _update_current_progress_remaining(self, num_timesteps: int, total_timesteps: int) -> None:
        """
        Compute current progress remaining (starts from 1 and ends to 0)

        :param num_timesteps: current number of timesteps
        :param total_timesteps:
        """
        self._current_progress_remaining = 1.0 - float(num_timesteps % total_timesteps ) / float(total_timesteps)

    def reset_exploration(self, reset_idx: int, clear_buffer: bool):
        self.current_resets = reset_idx
        if reset_idx == 0:
            # first exploration round, don't need to reset anything
            return
        # second or later exploration round, reset things:
        # reset info + success buffers
        self.ep_info_buffer = deque(maxlen=self._stats_window_size)
        self.ep_success_buffer = deque(maxlen=self._stats_window_size)
        if clear_buffer:
            self.replay_buffer.clear()
        # trigger environment reset
        self._last_obs = None

    def learn(self,
        total_timesteps: int,
        reset_exploration_times: int = 0,
        clear_buffer_on_reset: bool = False,
        callback = None,
        log_interval: int = 4,
        progress_bar: bool = False):


        for reset_idx in range(reset_exploration_times+1):
            self.reset_exploration(reset_idx, clear_buffer_on_reset)
            end_timestep_of_reset = (reset_idx + 1) * total_timesteps

            total_timesteps, callback = self._setup_learn(
                end_timestep_of_reset,
                callback,
                reset_num_timesteps=False,
                tb_log_name="DQN",
                progress_bar=progress_bar,
            )

            callback.on_training_start(locals(), globals())

            while self.num_timesteps < end_timestep_of_reset:
                rollout = self.collect_rollouts(
                    self.env,
                    train_freq=self.train_freq,
                    action_noise=self.action_noise,
                    callback=callback,
                    learning_starts=self.learning_starts,
                    replay_buffer=self.replay_buffer,
                    log_interval=log_interval,
                )

                if rollout.continue_training is False:
                    break

                if self.num_timesteps > 0 and self.num_timesteps > self.learning_starts:
                    # If no `gradient_steps` is specified,
                    # do as many gradients steps as steps performed during the rollout
                    gradient_steps = self.gradient_steps if self.gradient_steps >= 0 else rollout.episode_timesteps
                    # Special case when the user passes `gradient_steps=0`
                    if gradient_steps > 0:
                        self.train(batch_size=self.batch_size, gradient_steps=gradient_steps)

            callback.on_training_end()
