from typing import Any, Dict, Union, Optional
import os 
import numpy as np
import warnings

from stable_baselines3.common.callbacks import BaseCallback, EventCallback
from stable_baselines3.common.vec_env import sync_envs_normalization
from stable_baselines3.common.evaluation import evaluate_policy

from environment.vec.vecenv import CustomVecEnv


class CustomEvalCallback(EventCallback):
    """
    Callback for evaluating an agent.

    .. warning::

      When using multiple environments, each call to  ``env.step()``
      will effectively correspond to ``n_envs`` steps.
      To account for that, you can use ``eval_freq = max(eval_freq // n_envs, 1)``

    :param eval_env: The environment used for initialization
    :param callback_on_new_best: Callback to trigger
        when there is a new best model according to the ``mean_reward``
    :param callback_after_eval: Callback to trigger after every evaluation
    :param n_eval_episodes: The number of episodes to test the agent
    :param eval_freq: Evaluate the agent every ``eval_freq`` call of the callback.
    :param log_path: Path to a folder where the evaluations (``evaluations.npz``)
        will be saved. It will be updated at each evaluation.
    :param best_model_save_path: Path to a folder where the best model
        according to performance on the eval env will be saved.
    :param deterministic: Whether the evaluation should
        use a stochastic or deterministic actions.
    :param render: Whether to render or not the environment during evaluation
    :param verbose: Verbosity level: 0 for no output, 1 for indicating information about evaluation results
    :param warn: Passed to ``evaluate_policy`` (warns if ``eval_env`` has not been
        wrapped with a Monitor wrapper)
    """

    def __init__(
        self,
        eval_env: CustomVecEnv,
        mode: str, # eval or test
        callback_on_new_best: Optional[BaseCallback] = None,
        callback_after_eval: Optional[BaseCallback] = None,
        n_eval_episodes: int = 5,
        eval_freq: int = 10000,
        log_path: Optional[str] = None,
        best_model_save_path: Optional[str] = None,
        deterministic: bool = True,
        render: bool = False,
        verbose: int = 1,
        warn: bool = True,
    ):
        super().__init__(callback_after_eval, verbose=verbose)

        self.callback_on_new_best = callback_on_new_best
        if self.callback_on_new_best is not None:
            # Give access to the parent
            self.callback_on_new_best.parent = self

        self.n_eval_episodes = n_eval_episodes
        self.eval_freq = eval_freq
        self.best_mean_reward = -np.inf
        self.last_mean_reward = -np.inf
        self.deterministic = deterministic
        self.render = render
        self.warn = warn
        self.mode = mode

        self.eval_env = eval_env
        self.best_model_save_path = best_model_save_path
        # Logs will be written in ``evaluations.npz``
        if log_path is not None:
            log_path = os.path.join(log_path, "evaluations")
        self.log_path = log_path
        self.evaluations_results = []
        self.evaluations_timesteps = []
        self.evaluations_length = []
        # For computing success rate
        self._is_success_buffer = []
        self.evaluations_successes = []
        self.goal_asked_free = []
        self.goal_asked_guided = []
        self.goal_reached_free = []
        self.goal_reached_guided = []
        self.goal_node_coverage_free = []
        self.goal_node_coverage_guided = []
        
        # For computing graph stats
        self.node_coverage = []
        self.synonym_coverage_questions = []
        self.synonym_coverage_answers = []
        # TODO self.variable_coverage = []
        self.actioncount_skips_invalid = []
        self.actioncount_ask_variable_irrelevant = []
        self.actioncount_ask_question_irrelevant = []
        self.actioncount_missingvariable = []



    def _init_callback(self) -> None:
        # Does not work in some corner cases, where the wrapper is not the same
        if not isinstance(self.training_env, type(self.eval_env)):
            warnings.warn("Training and eval env are not of the same type" f"{self.training_env} != {self.eval_env}")

        # Create folders if needed
        if self.best_model_save_path is not None:
            os.makedirs(self.best_model_save_path, exist_ok=True)
        if self.log_path is not None:
            os.makedirs(os.path.dirname(self.log_path), exist_ok=True)

        # Init callback called on new best model
        if self.callback_on_new_best is not None:
            self.callback_on_new_best.init_callback(self.model)

    def _log_success_callback(self, locals_: Dict[str, Any], globals_: Dict[str, Any]) -> None:
        """
        Callback passed to the  ``evaluate_policy`` function
        in order to log the success rate (when applicable),
        for instance when using HER.

        :param locals_:
        :param globals_:
        """
        info = locals_["info"]

        if locals_["done"]:
            maybe_is_success = info.get("is_success")
            if maybe_is_success is not None:
                self._is_success_buffer.append(maybe_is_success)

    def _on_step(self) -> bool:
        continue_training = True

        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            # Sync training and eval env if there is VecNormalize
            if self.model.get_vec_normalize_env() is not None:
                try:
                    sync_envs_normalization(self.training_env, self.eval_env)
                except AttributeError as e:
                    raise AssertionError(
                        "Training and eval env are not wrapped the same way, "
                        "see https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html#evalcallback "
                        "and warning above."
                    ) from e

            # Reset metrics
            if hasattr(self.eval_env.envs[0], "guided_env"):
                self.eval_env.envs[0].guided_env.reset_stats()
            if hasattr(self.eval_env.envs[0], 'free_env'):
                self.eval_env.envs[0].free_env.reset_stats()

            episode_rewards, episode_lengths = evaluate_policy(
                self.model,
                self.eval_env,
                n_eval_episodes=self.n_eval_episodes,
                render=self.render,
                deterministic=self.deterministic,
                return_episode_rewards=True,
                warn=self.warn,
                callback=self._log_success_callback,
            )

            if self.log_path is not None:
                self.evaluations_timesteps.append(self.num_timesteps)
                self.evaluations_results.append(episode_rewards)
                self.evaluations_length.append(episode_lengths)

                kwargs = {}
                # Save success log if present
                if len(self._is_success_buffer) > 0:
                    self.evaluations_successes.append(self._is_success_buffer)
                    kwargs = dict(successes=self.evaluations_successes)

                np.savez(
                    self.log_path,
                    timesteps=self.evaluations_timesteps,
                    results=self.evaluations_results,
                    ep_lengths=self.evaluations_length,
                    **kwargs,
                )

            mean_reward, std_reward = np.mean(episode_rewards), np.std(episode_rewards)
            mean_ep_length, std_ep_length = np.mean(episode_lengths), np.std(episode_lengths)
            self.last_mean_reward = mean_reward

            # create aggregate data in vec env, log here
            self.goal_asked_free.append(self.eval_env.stats_asked_goals_free())
            self.goal_asked_guided.append(self.eval_env.stats_asked_goals_guided())
            self.goal_reached_free.append(self.eval_env.stats_reached_goals_free())
            self.goal_reached_guided.append(self.eval_env.stats_reached_goals_guided())
            self.goal_node_coverage_free.append(self.eval_env.stats_goal_node_coverage_free())
            self.goal_node_coverage_guided.append(self.eval_env.stats_goal_node_coverage_guided())
            self.node_coverage.append(self.eval_env.stats_node_coverage())
            self.synonym_coverage_questions.append(self.eval_env.stats_synonym_coverage_questions())
            self.synonym_coverage_answers.append(self.eval_env.stats_synonym_coverage_answers())
            self.actioncount_skips_invalid.append(self.eval_env.stats_actioncount_skips_indvalid())
            self.actioncount_ask_variable_irrelevant.append(self.eval_env.stats_actioncount_ask_variable_irrelevant())
            self.actioncount_ask_question_irrelevant.append(self.eval_env.stats_actioncount_ask_question_irrelevant())
            self.actioncount_missingvariable.append(self.eval_env.stats_actioncount_missingvariable())

            self.logger.record(f"{self.mode}/goal_asked_free", self.goal_asked_free[-1])
            self.logger.record(f"{self.mode}/goal_asked_guided", self.goal_asked_guided[-1])
            self.logger.record(f"{self.mode}/goal_reached_free", self.goal_reached_free[-1])
            self.logger.record(f"{self.mode}/goal_reached_guided", self.goal_reached_guided[-1])
            self.logger.record(f"{self.mode}/goal_node_coverage_free", self.goal_node_coverage_free[-1])
            self.logger.record(f"{self.mode}/goal_node_coverage_guided", self.goal_node_coverage_guided[-1])
            self.logger.record(f"{self.mode}/node_coverage", self.node_coverage[-1])
            self.logger.record(f"{self.mode}/synonym_coverage_questions", self.synonym_coverage_questions[-1])
            self.logger.record(f"{self.mode}/synonym_coverage_answers", self.synonym_coverage_answers[-1])
            self.logger.record(f"{self.mode}/actioncount_skips_invalid", self.actioncount_skips_invalid[-1])
            self.logger.record(f"{self.mode}/actioncount_ask_variable_irrelevant", self.actioncount_ask_variable_irrelevant[-1])
            self.logger.record(f"{self.mode}/actioncount_ask_question_irrelevant", self.actioncount_ask_question_irrelevant[-1])
            self.logger.record(f"{self.mode}/actioncount_missingvariable", self.actioncount_missingvariable[-1])

            if self.verbose >= 1:
                print(f"Eval num_timesteps={self.num_timesteps}, " f"episode_reward={mean_reward:.2f} +/- {std_reward:.2f}")
                print(f"Episode length: {mean_ep_length:.2f} +/- {std_ep_length:.2f}")
            # Add to current Logger
            self.logger.record(f"{self.mode}/mean_reward", float(mean_reward))
            self.logger.record(f"{self.mode}/mean_ep_length", mean_ep_length)

            if len(self._is_success_buffer) > 0:
                success_rate = np.mean(self._is_success_buffer)
                if self.verbose >= 1:
                    print(f"Success rate: {100 * success_rate:.2f}%")
                self.logger.record(f"{self.mode}/success_rate", success_rate)

            # Dump log so the evaluation results are printed with the correct timestep
            self.logger.record(f"{self.mode}/time/total_timesteps", self.num_timesteps, exclude="tensorboard")
            self.logger.dump(self.num_timesteps)

            if mean_reward > self.best_mean_reward:
                if self.verbose >= 1:
                    print("New best mean reward!")
                if self.best_model_save_path is not None:
                    self.model.save(os.path.join(self.best_model_save_path, "best_model"))
                self.best_mean_reward = mean_reward
                # Trigger callback on new best model, if needed
                if self.callback_on_new_best is not None:
                    continue_training = self.callback_on_new_best.on_step()

            # Trigger callback after every evaluation, if needed
            if self.callback is not None:
                continue_training = continue_training and self._on_event()

        return continue_training

    def update_child_locals(self, locals_: Dict[str, Any]) -> None:
        """
        Update the references to the local variables.

        :param locals_: the local variables during rollout collection
        """
        if self.callback:
            self.callback.update_locals(locals_)