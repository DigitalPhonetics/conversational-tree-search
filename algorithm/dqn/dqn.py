
import io
import pathlib
from statistics import mean
from typing import Iterable, Set, Tuple, TypeVar, Union, Dict, Optional, Type, Any
from stable_baselines3 import DQN

import torch as th
import torch.nn.functional as F

from gymnasium import spaces

from typing import Any, Dict, Optional, Tuple, Type, Union
from collections import deque 
from sortedcontainers import SortedList

import numpy as np
import torch as th
from gymnasium import spaces
from torch.nn import functional as F

from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.type_aliases import GymEnv, Schedule, TrainFreq, RolloutReturn, TrainFrequencyUnit
from stable_baselines3.dqn.policies import QNetwork
from algorithm.dqn.her import HindsightExperienceReplayWrapper
from stable_baselines3.common.save_util import recursive_getattr
from stable_baselines3.common.utils import should_collect_more_steps
from stable_baselines3.dqn.policies import QNetwork
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.noise import ActionNoise

from algorithm.dqn.policy import CustomDQNPolicy
from algorithm.dqn.targets import DQNTarget
from utils.loader import load_model, save_to_zip_file
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
        num_prediction_heads: int = 1,
        noise_std_init = 0.0,
        active_learning_args: Optional[cfg.ActiveLearningConfig] = None
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

        self.num_prediction_heads = num_prediction_heads
        self.current_prediction_head_mask = None

        self.noise_std_init = noise_std_init

        # Setup active learning, if configured
        self.active_learning_args = active_learning_args
        if active_learning_args.active:
            # load expert model here + config
            cfg, self.expert, state_enc = load_model(dqn_cls=CustomDQN, ckpt_path=active_learning_args.expert_model_ckpt, cfg_name=active_learning_args.expert_model_cfg,
                                     device=device,
                                     data=env.envs[0].data)
            self.active_learning_args = active_learning_args
            
            self.uncertanties = [deque([], maxlen=active_learning_args.reference_size) for _ in range(env.num_envs)]
            self.recent_uncertainties = [SortedList() for _ in range(env.num_envs)]
            self.query_budget = active_learning_args.query_budget # max. number of expert demonstrations (with T steps per demonstration)
            self.current_demonstration_step = active_learning_args.expert_turns # one demonstration consists of T steps
            self.current_demonstration_env_id = -1 # we only do demonstrations for max. 1 environment per step (don't use up whole budget at once)

            
        self.global_step = 0

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
                next_q_values = self.q_net_target(replay_data.next_observations) # batch (x num_heads) x actions
                if self.policy.intent_prediction:
                    # split output of network into q values, ignore intents (position 1)
                    next_q_values = next_q_values[0]
                target_q_values = self.target.target(next_q_values=next_q_values, data=replay_data, q_old=current_q_values).squeeze() # batch (x num_heads)
               
            # Retrieve the q-values for the actions from the replay buffer
            if next_q_values.dim() == 2:
                # standard model
                current_q_values = th.gather(current_q_values, dim=-1, index=replay_data.actions.long()).squeeze()
            else:
                # multihead model
                num_prediction_heads = next_q_values.size(1)
                current_q_values = current_q_values.gather(-1, replay_data.actions.view(batch_size,1).repeat(1, num_prediction_heads).unsqueeze(-1)) # batch x num_heads x 1
                current_q_values = current_q_values.squeeze(-1)     # batch x num_heads
                loss = loss.view(-1,1)
            q_values.extend(current_q_values.view(-1).tolist())

            # Compute Huber loss (less sensitive to outliers)
            # td_loss = F.smooth_l1_loss(current_q_values, target_q_values)
            td_loss = F.huber_loss(current_q_values, target_q_values, reduction='none')
            if th.is_tensor(replay_data.prediction_head_mask): # batch x num_heads
                # bootstrapped network with sampling probability < 1 per head -> we have to mask the heads
                td_loss = td_loss * replay_data.prediction_head_mask # batch x num_heads
            td_losses.append(td_loss.view(-1).mean(-1).item())
            loss = loss + td_loss
            if "prioritized" in self.replay_buffer.__class__.__name__.lower() or "hindsight" in self.replay_buffer.__class__.__name__.lower():
                # weight loss by priority
                if next_q_values.dim() == 2:
                    # standard model
                    loss = loss * replay_data.weights
                    td_error = th.abs(target_q_values - current_q_values)
                else:
                    # multihead model
                    loss = loss * replay_data.weights.view(-1,1)
                    td_error = th.abs(target_q_values.mean(-1) - current_q_values.mean(-1))
                # update priorities
                self.replay_buffer.update_weights(replay_data.indices, td_error)
            loss = loss.view(-1)
            loss = loss.mean(-1) # reduce loss

            # Optimize the policy
            self.policy.optimizer.zero_grad()
            loss.backward()

            if self.num_prediction_heads > 1:
                # scale shared gradients by 1/num_heads
                q_net_named_layers = dict(self.q_net.named_children())
                if "_orig_mod" in q_net_named_layers:
                    q_net_named_layers = dict(q_net_named_layers.pop('_orig_mod').named_children())
                for shared_layer_name in self.q_net.get_shared_module_names():
                    for module in q_net_named_layers[shared_layer_name]:
                        if hasattr(module, "weight") and module.weight.grad is not None:
                            module.weight.grad.data *= 1.0/float(self.num_prediction_heads)
                        if hasattr(module, "bias") and module.bias.grad is not None:
                            module.bias.grad.data *= 1.0/float(self.num_prediction_heads)
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
        if self.policy.intent_prediction:
            self.logger.record("train/intent_loss", np.mean(intent_losses))
        if self.replay_buffer_class in [HindsightExperienceReplayWrapper]:
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

    def active_learning_should_query_expert(self, obs: th.tensor) -> th.BoolTensor:
        """
        Args:
            obs: batch x obs_space
        """
        if (not self.active_learning_args.active) or self.query_budget == 0:
            # AL not activated, or no more queries left
            return False
        assert 0 <= self.active_learning_args.proportion_threshold <= 1
        self.logger.record("train/al_query_budget", self.query_budget)
 
        asking = False
        
        uncertainty = self.policy.q_net.uncertainty(obs=obs).tolist() # batch
        # find most uncertain env (cumulatively? or mean across all envs?)
        # also, update uncertainty reference windows for each env
        highest_uncertainty = uncertainty[0]
        most_uncertain_env_idx = 0
        for env_idx in range(obs.size(0)):
            if len(self.uncertanties[env_idx]) == self.active_learning_args.reference_size: # handle case that the uncertainties are still empty or not fully filled (initially)  
                step_idx = int(self.active_learning_args.proportion_threshold * len(self.uncertanties[env_idx])) - 1
                uncertainty_threshold = self.recent_uncertainties[env_idx][step_idx] # TODO: should this be sorted ascending, or descending?
                
                if uncertainty[env_idx] > uncertainty_threshold:
                    asking = True
                    if uncertainty[env_idx] > highest_uncertainty:
                        highest_uncertainty = uncertainty[env_idx]
                        most_uncertain_env_idx = env_idx

                # remove oldest uncertainty from buffer
                uncertainty_to_delete = self.uncertanties[env_idx].popleft()
                self.recent_uncertainties[env_idx].remove(uncertainty_to_delete)
            
            # add current uncertainty estimate to buffer
            self.uncertanties[env_idx].append(uncertainty[env_idx]) # append to the right side of the deque
            self.recent_uncertainties[env_idx].add(uncertainty[env_idx]) # put into sorted list


        if self.current_demonstration_step < self.active_learning_args.expert_turns:
            # there is a demonstration active right now, make sure to continue it until reaching defined number of expert steps
            asking = True
        else:
            # no running demonstration - reset parameters
            self.current_demonstration_env_id = most_uncertain_env_idx
            self.current_demonstration_step = self.active_learning_args.expert_turns if not asking else 0

        self.logger.record("train/al_highest_uncertainty", highest_uncertainty)
        self.logger.record("train/al_avg_uncertainty", mean(uncertainty))
        return asking

    @th.no_grad()
    def query_expert_action(self, obs: th.Tensor) -> int:
        # query only for most uncertain env
        env_obs = obs[self.current_demonstration_env_id].unsqueeze(0)
        action, _ = self.expert.predict(observation=env_obs, deterministic=True)

        self.query_budget -= 1
        self.current_demonstration_step += 1    

        return action

    def _predict_eps_greedy(
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
        if not deterministic and np.random.rand() < self.exploration_rate:
            # exploration
            action = self._draw_random_actions(observation)
        else:
            # exploitation
            # NOTE: masking is already built-in if actions_in_state_space = True
            action, state, intent_classes = self.policy.predict(observation, state, episode_start, deterministic)
            return action, intent_classes
        return action, None
    
    def _predict_multi_head(self,
        observation: Union[np.ndarray, Dict[str, np.ndarray]],
        state: Optional[Tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        # if deterministic, use majority vote
        # if random, greedily sample currently active exploration head (stored within multihead_prediction_head_mask)
        action, state, intent_classes = self.policy.predict(observation, state, episode_start, deterministic, prediction_head_mask=self.current_prediction_head_mask)     
        return action, intent_classes
    
    def _predict_noisy(self,
        observation: Union[np.ndarray, Dict[str, np.ndarray]],
        state: Optional[Tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        # predict noisy action (depending on deterministic=True/False)
        action, state, intent_classes = self.policy.predict(observation, state, episode_start, deterministic)     
        return action, intent_classes

    def predict(
        self,
        observation: Union[np.ndarray, Dict[str, np.ndarray]],
        state: Optional[Tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        """
        Overrides the base_class predict function to include various exploration modes.

        :param observation: the input observation
        :param state: The last states (can be None, used in recurrent policies)
        :param episode_start: The last masks (can be None, used in recurrent policies)
        :param deterministic: Whether or not to return deterministic actions.
        :return: the model's action and the intent instead of the state (we can abuse state return here since we don't have recurrent policies)
        """
        # TODO action masking for actions not in state space

        actions = None
        intents = None

        if self.num_prediction_heads > 1:
            # bootstrapped DQN
            actions, intents = self._predict_multi_head(observation=observation, state=state, episode_start=episode_start, deterministic=deterministic)
        elif self.noise_std_init > 0:
            # noisy DQN
            actions, intents =  self._predict_noisy(observation=observation, state=state, episode_start=episode_start, deterministic=deterministic)
        else:
            # normal DQN - eps greedy
            actions, intents = self._predict_eps_greedy(observation=observation, state=state, episode_start=episode_start, deterministic=deterministic)

        # should we query an expert (active learning only)?
        if self.active_learning_should_query_expert(obs=observation):
            if self.current_demonstration_env_id < observation.size(0):
                # TODO why is the bug (screenshot) happening?
                actions[self.current_demonstration_env_id] = self.query_expert_action(obs=observation).item()
            else:
                # prevent bug: reset 
                self.current_demonstration_step = self.active_learning_args.expert_turns

        return actions, intents

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

    def continue_learning(self,
        current_steps, 
        current_timesteps_at_start,
        current_episode_num,
        current_progress_remaining,
        current_n_updates,
        current_n_calls,
        current_exploration_rate,
        current_global_step,
        total_timesteps: int,
        callback = None,
        log_interval: int = 4,
        progress_bar: bool = False):

        self.num_timesteps = current_steps
        self._num_timesteps_at_start=current_timesteps_at_start
        self._episode_num = current_episode_num
        self._current_progress_remaining = current_progress_remaining
        self._n_updates = current_n_updates
        self._n_calls = current_n_calls
        self.exploration_rate = current_exploration_rate
        self.global_step = current_global_step

        _, callback = self._setup_learn(
            total_timesteps,
            callback,
            reset_num_timesteps=False,
            tb_log_name="DQN",
            progress_bar=progress_bar,
        )

        # reset total timesteps, because stable-baselines will increase it by the current number of steps - this would destroy the idea of the modulo operation in _update_current_progress_remaining  
        self.num_timesteps = current_steps
        self._num_timesteps_at_start=current_timesteps_at_start
        self._episode_num = current_episode_num
        self._current_progress_remaining = current_progress_remaining
        self._n_updates = current_n_updates
        self._n_calls = current_n_calls
        self.exploration_rate = current_exploration_rate
        self.global_step = current_global_step
        self._total_timesteps = total_timesteps

        callback.on_training_start(locals(), globals())

        while self.num_timesteps < total_timesteps:
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

        return self

    def _on_step(self) -> None:
        super()._on_step()
        
        # reset logs if we are trainig
        if self.policy.training:
            for env in self.env.envs:
                env.reset_episode_log()

    def save(
        self,
        path: Union[str, pathlib.Path, io.BufferedIOBase],
        exclude: Optional[Iterable[str]] = None,
        include: Optional[Set[str]] = None,
    ) -> None:
        """
        Save all the attributes of the object and the model parameters in a zip-file.

        :param path: path to the file where the rl agent should be saved
        :param exclude: name of parameters that should be excluded in addition to the default ones
        :param include: name of parameters that might be excluded but should be included anyway
        """
        save_to_zip_file(path, params=self.get_parameters(), replay_buffer=None)
        # save_to_zip_file(path, params=self.get_parameters(), replay_buffer=self.replay_buffer)

    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        train_freq: TrainFreq,
        replay_buffer: ReplayBuffer,
        action_noise: Optional[ActionNoise] = None,
        learning_starts: int = 0,
        log_interval: Optional[int] = None,
    ) -> RolloutReturn:
        """
        Collect experiences and store them into a ``ReplayBuffer``.

        :param env: The training environment
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param train_freq: How much experience to collect
            by doing rollouts of current policy.
            Either ``TrainFreq(<n>, TrainFrequencyUnit.STEP)``
            or ``TrainFreq(<n>, TrainFrequencyUnit.EPISODE)``
            with ``<n>`` being an integer greater than 0.
        :param action_noise: Action noise that will be used for exploration
            Required for deterministic policy (e.g. TD3). This can also be used
            in addition to the stochastic policy for SAC.
        :param learning_starts: Number of steps before learning for the warm-up phase.
        :param replay_buffer:
        :param log_interval: Log data every ``log_interval`` episodes
        :return:
        """
        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)

        num_collected_steps, num_collected_episodes = 0, 0

        assert isinstance(env, VecEnv), "You must pass a VecEnv"
        assert train_freq.frequency > 0, "Should at least collect one step or episode."

        if env.num_envs > 1:
            assert train_freq.unit == TrainFrequencyUnit.STEP, "You must use only one env when doing episodic training."

        if self.use_sde:
            self.actor.reset_noise(env.num_envs)

        # if we are using the multi-head model, initialize the mask for which head we are going to use for sampling per env
        if self.num_prediction_heads > 1 and isinstance(self.current_prediction_head_mask, type(None)):
            self.current_prediction_head_mask = th.randint(low=0, high=self.num_prediction_heads, size=(self.n_envs,)) # dim: n_envs

        callback.on_rollout_start()
        continue_training = True
        while should_collect_more_steps(train_freq, num_collected_steps, num_collected_episodes):
            if self.use_sde and self.sde_sample_freq > 0 and num_collected_steps % self.sde_sample_freq == 0:
                # Sample a new noise matrix
                self.actor.reset_noise(env.num_envs)

            # Select action randomly or according to policy
            actions, buffer_actions = self._sample_action(learning_starts, action_noise, env.num_envs)

            # Rescale and perform action
            new_obs, rewards, dones, infos = env.step(actions)

            self.num_timesteps += env.num_envs
            num_collected_steps += 1

            # Give access to local variables
            callback.update_locals(locals())
            # Only stop training if return value is False, not when it is None.
            if not callback.on_step():
                return RolloutReturn(num_collected_steps * env.num_envs, num_collected_episodes, continue_training=False)

            # Retrieve reward and episode length if using Monitor wrapper
            self._update_info_buffer(infos, dones)

            # Store data in replay buffer (normalized action and unnormalized observation)
            self._store_transition(replay_buffer, buffer_actions, new_obs, rewards, dones, infos)  # type: ignore[arg-type]

            self._update_current_progress_remaining(self.num_timesteps, self._total_timesteps)

            # For DQN, check if the target network should be updated
            # and update the exploration schedule
            # For SAC/TD3, the update is dones as the same time as the gradient update
            # see https://github.com/hill-a/stable-baselines/issues/900
            self._on_step()

            for idx, done in enumerate(dones):
                if done:
                    # Update stats
                    num_collected_episodes += 1
                    self._episode_num += 1

                    if action_noise is not None:
                        kwargs = dict(indices=[idx]) if env.num_envs > 1 else {}
                        action_noise.reset(**kwargs)

                    # Log training infos
                    if log_interval is not None and self._episode_num % log_interval == 0:
                        self._dump_logs()

                    # draw new sampling head for the finished env
                    if th.is_tensor(self.current_prediction_head_mask):
                        self.current_prediction_head_mask[idx] = th.randint(low=0, high=self.num_prediction_heads, size=(1,))
        callback.on_rollout_end()

        return RolloutReturn(num_collected_steps * env.num_envs, num_collected_episodes, continue_training)