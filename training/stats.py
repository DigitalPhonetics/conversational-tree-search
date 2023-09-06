from statistics import mean
from typing import Any, Dict, Union, Optional
import os 
import numpy as np
import warnings

from stable_baselines3.common.callbacks import BaseCallback, EventCallback
from stable_baselines3.common.vec_env import sync_envs_normalization

from algorithm.evaluation import custom_evaluate_policy as evaluate_policy
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
        keep_checkpoints: int = 1,
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
        self.keep_checkpoints = keep_checkpoints
        self.checkpoint_handles = {} # map checkpoint scores to paths

        self.eval_env = eval_env
        self.best_model_save_path = best_model_save_path
        # # Logs will be written in ``evaluations.npz``
        self.log_path = log_path
        print("LOG PATH:", self.log_path)
        self.evaluations_results_free = []
        self.evaluations_results_guided = []
        self.evaluations_timesteps = []
        self.evaluations_length_free = []
        self.evaluations_length_guided = []
        self.evaluations_percieved_length_free = []
        self.evaluations_percieved_length_guided = []
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
        
        self.cumulative_coverage_nodes = set()
        self.cumulative_coverage_questions = set()
        self.cumulative_coverage_answers = set()

        # intent prediction
        self.intent_accuracies = []
        self.intent_consistencies = []

        self.free_dialogs = []
        self.guided_dialogs = []


    def _init_callback(self) -> None:
        # Does not work in some corner cases, where the wrapper is not the same
        if not isinstance(self.training_env, type(self.eval_env)):
            warnings.warn("Training and eval env are not of the same type" f"{self.training_env} != {self.eval_env}")

        # Create folders if needed
        if self.best_model_save_path is not None and not os.path.exists(self.best_model_save_path):
            os.makedirs(self.best_model_save_path)
        if self.log_path is not None and not os.path.exists(self.log_path):
            os.makedirs(self.log_path)

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
            for env in self.eval_env.envs:
                if hasattr(env, "guided_env"):
                    env.guided_env.reset_stats()
                if hasattr(env, 'free_env'):
                    env.free_env.reset_stats()

            episode_rewards_free, episode_rewards_guided, episode_lengths_free, episode_lengths_guided, percieved_lengths_free, percieved_lengths_guided, intent_accuracy, intent_consistency, free_dialogs, guided_dialogs, dialog_log = evaluate_policy(
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
                self.evaluations_results_free.append(episode_rewards_free)
                self.evaluations_results_guided.append(episode_rewards_guided)
                self.evaluations_length_free.append(episode_lengths_free)
                self.evaluations_length_guided.append(episode_lengths_guided)
                self.evaluations_percieved_length_free.append(episode_lengths_free)
                self.evaluations_percieved_length_guided.append(episode_lengths_guided)

                kwargs = {}
                # Save success log if present
                if len(self._is_success_buffer) > 0:
                    self.evaluations_successes.append(self._is_success_buffer)
                    kwargs = dict(successes=self.evaluations_successes)

                # TODO re-enable
                # np.savez(
                #     self.log_path,
                #     timesteps=self.evaluations_timesteps,
                #     results_free=self.evaluations_results_free,
                #     results_guided=self.evaluations_results_guided,
                #     ep_lengths_free=self.evaluations_length_free,
                #     ep_lengths_guided=self.evaluations_length_guided,
                #     goal_asked_free=self.goal_asked_free,
                #     goal_asked_guided=self.goal_asked_guided,
                #     goal_reached_free=self.goal_reached_free,
                #     goal_reached_guided=self.goal_reached_guided,
                #     goal_node_coverage_free=self.goal_node_coverage_free,
                #     goal_node_coverage_guided=self.goal_node_coverage_guided,
                #     node_coverage=self.node_coverage,
                #     synonym_coverage_questions=self.synonym_coverage_questions,
                #     synonym_coverage_answers=self.synonym_coverage_answers,
                #     actioncount_skips_invalid=self.actioncount_skips_invalid,
                #     actioncount_ask_variable_irrelevant=self.actioncount_ask_variable_irrelevant,
                #     actioncount_ask_question_irrelevant=self.actioncount_ask_question_irrelevant,
                #     actioncount_missingvariable=self.actioncount_missingvariable,
                #     intent_accuracies=self.intent_accuracies,
                #     intent_consistencies=self.intent_consistencies,
                #     free_dialogs=self.free_dialogs,
                #     guided_dialogs=self.guided_dialogs,
                #     **kwargs,
                # )

                # log dialogs
                with open(f"{self.log_path}/dialogs_{self.n_calls // self.eval_freq}.txt", "w") as f:
                    f.write(f"#LAST EPISODE: {self.eval_env.current_episode}")
                    f.writelines([log_line + "\n" for log_line in dialog_log])


            mean_reward, std_reward = np.mean(episode_rewards_free + episode_rewards_guided), np.std(episode_rewards_free + episode_rewards_guided)
            mean_ep_length, std_ep_length = np.mean(episode_lengths_free + episode_lengths_guided), np.std(episode_lengths_free + episode_lengths_guided)
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
            self.free_dialogs.append(free_dialogs)
            self.guided_dialogs.append(guided_dialogs)
            
            for env in self.eval_env.envs:
                if hasattr(env, "free_env"):
                    self.cumulative_coverage_nodes.update(env.free_env.node_coverage.keys())
                    self.cumulative_coverage_questions.update(env.free_env.coverage_question_synonyms.keys())
                    self.cumulative_coverage_answers.update(env.free_env.coverage_answer_synonyms.keys())
                if hasattr(env, 'guided_env'):
                    self.cumulative_coverage_nodes.update(env.guided_env.node_coverage.keys())
                    self.cumulative_coverage_answers.update(env.guided_env.coverage_answer_synonyms.keys())

            if not isinstance(intent_accuracy, type(None)):
                self.intent_accuracies.append(intent_accuracy)
                self.intent_consistencies.append(intent_consistency)
                self.logger.record(f"{self.mode}/intent_accuracy", self.intent_accuracies[-1])
                self.logger.record(f"{self.mode}/intent_consistency", self.intent_consistencies[-1])

            self.logger.record(f"{self.mode}/max_goal_distance", self.eval_env.envs[0].max_distance)
            self.logger.record(f"{self.mode}/free_dialog_percentage", self.free_dialogs[-1])
            self.logger.record(f"{self.mode}/guided_dialog_percentage", self.guided_dialogs[-1])
            self.logger.record(f"{self.mode}/goal_asked_free", self.goal_asked_free[-1])
            self.logger.record(f"{self.mode}/goal_asked_guided", self.goal_asked_guided[-1])
            self.logger.record(f"{self.mode}/goal_reached_free", self.goal_reached_free[-1])
            self.logger.record(f"{self.mode}/goal_reached_guided", self.goal_reached_guided[-1])
            self.logger.record(f"{self.mode}/goal_node_coverage_free", self.goal_node_coverage_free[-1])
            self.logger.record(f"{self.mode}/goal_node_coverage_guided", self.goal_node_coverage_guided[-1])
            self.logger.record(f"{self.mode}/epoch_node_coverage", self.node_coverage[-1])
            self.logger.record(f"{self.mode}/total_node_coverage", len(self.cumulative_coverage_nodes) / len(self.eval_env.envs[0].data.node_list))
            self.logger.record(f"{self.mode}/epoch_node_coverage", self.node_coverage[-1])
            self.logger.record(f"{self.mode}/total_coverage_questions", len(self.cumulative_coverage_questions) / len(self.eval_env.envs[0].data.question_list))
            self.logger.record(f"{self.mode}/epoch_coverage_questions", self.synonym_coverage_questions[-1])
            self.logger.record(f"{self.mode}/total_coverage_answers", len(self.cumulative_coverage_answers) / self.eval_env.envs[0].data.num_answer_synonyms)
            self.logger.record(f"{self.mode}/epoch_coverage_answers", self.synonym_coverage_answers[-1])
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
            if len(episode_lengths_free) > 0:
                self.logger.record(f"{self.mode}/ep_reward_free", mean(episode_rewards_free))
                self.logger.record(f"{self.mode}/ep_length_free", mean(episode_lengths_free))
                self.logger.record(f"{self.mode}/perceived_length_free", mean(percieved_lengths_free))
            if len(episode_rewards_guided) > 0:
                self.logger.record(f"{self.mode}/ep_reward_guided", mean(episode_rewards_guided))
                self.logger.record(f"{self.mode}/ep_length_guided", mean(episode_lengths_guided))
                self.logger.record(f"{self.mode}/perceived_length_guided", mean(percieved_lengths_guided))

            if len(self._is_success_buffer) > 0:
                success_rate = np.mean(self._is_success_buffer)
                if self.verbose >= 1:
                    print(f"Success rate: {100 * success_rate:.2f}%")
                self.logger.record(f"{self.mode}/success_rate", success_rate)

            # Dump log so the evaluation results are printed with the correct timestep
            self.logger.record(f"{self.mode}/time/total_timesteps", self.num_timesteps, exclude="tensorboard")
            self.logger.dump(self.num_timesteps)

          
            lowest_checkpoint_reward = 1000000000
            if len(self.checkpoint_handles) >= self.keep_checkpoints:
                # reached max. number of checkpoints to keep
                # -> find lowest currently stored checkpoint reward value
                for checkpoint_reward in self.checkpoint_handles:
                    if checkpoint_reward < lowest_checkpoint_reward:
                        lowest_checkpoint_reward = checkpoint_reward
            else:
                # we don't have reached the max. amount of checkpoints possible -> don't have to delete any other checkpoints
                lowest_checkpoint_reward = None
            if isinstance(lowest_checkpoint_reward, type(None)) or lowest_checkpoint_reward <= mean_reward:
                # current reward is better than lowest saved checkpoint (or we have stored less than we can)
                # -> delete lowest checkoint, then save current checkpoint
                if not isinstance(lowest_checkpoint_reward, type(None)):
                    os.remove(os.path.join(self.best_model_save_path, f"ckpt_{self.checkpoint_handles[lowest_checkpoint_reward]}.pt"))
                    os.remove(os.path.join(self.best_model_save_path, f"stats_{self.checkpoint_handles[lowest_checkpoint_reward]}.txt"))
                    del self.checkpoint_handles[lowest_checkpoint_reward]
                # save model and stats
                self.checkpoint_handles[mean_reward] = self.n_calls // self.eval_freq
                self.model.save(os.path.join(self.best_model_save_path, f"ckpt_{self.n_calls // self.eval_freq}"), include=set(['replay_buffer']))
                with open(os.path.join(self.best_model_save_path, f"stats_{self.n_calls // self.eval_freq}.txt"), "w") as f:
                    lines = [f"Eval num_timesteps={self.num_timesteps}\n",
                             f"{self.mode}/max_goal_distance: {self.eval_env.envs[0].max_distance}\n",
                             f"{self.mode}/free_dialog_percentage: {self.free_dialogs[-1]}\n",
                             f"{self.mode}/guided_dialog_percentage: {self.guided_dialogs[-1]}\n",
                             f"{self.mode}/goal_asked_free: {self.goal_asked_free[-1]}\n",
                             f"{self.mode}/goal_asked_guided: {self.goal_asked_guided[-1]}\n",
                             f"{self.mode}/goal_reached_free: {self.goal_reached_free[-1]}\n",
                             f"{self.mode}/goal_reached_guided: {self.goal_reached_guided[-1]}\n",
                             f"{self.mode}/goal_node_coverage_free: {self.goal_node_coverage_free[-1]}\n",
                             f"{self.mode}/goal_node_coverage_guided: {self.goal_node_coverage_guided[-1]}\n",
                             f"{self.mode}/epoch_node_coverage: {self.node_coverage[-1]}\n",
                             f"{self.mode}/total_node_coverage: {len(self.cumulative_coverage_nodes) / len(self.eval_env.envs[0].data.node_list)}\n",
                             f"{self.mode}/epoch_node_coverage: {self.node_coverage[-1]}\n",
                             f"{self.mode}/total_coverage_questions: {len(self.cumulative_coverage_questions) / len(self.eval_env.envs[0].data.question_list)}\n",
                             f"{self.mode}/epoch_coverage_question: { self.synonym_coverage_questions[-1]}\n",
                             f"{self.mode}/total_coverage_answers: {len(self.cumulative_coverage_answers) / self.eval_env.envs[0].data.num_answer_synonyms}\n",
                             f"{self.mode}/epoch_coverage_answers: {self.synonym_coverage_answers[-1]}\n",
                             f"{self.mode}/actioncount_skips_invalid: {self.actioncount_skips_invalid[-1]}\n",
                             f"{self.mode}/actioncount_ask_variable_irrelevant: {self.actioncount_ask_variable_irrelevant[-1]}\n",
                             f"{self.mode}/actioncount_ask_question_irrelevant: {self.actioncount_ask_question_irrelevant[-1]}\n",
                             f"{self.mode}/actioncount_missingvariable: {self.actioncount_missingvariable[-1]}\n",
                             f"Mean episode reward={mean_reward:.2f} +/- {std_reward:.2f}\n",
                             f"Mean episode length: {mean_ep_length:.2f} +/- {std_ep_length:.2f}\n"
                    ]
                    if not isinstance(intent_accuracy, type(None)):
                        lines += [f"{self.mode}/intent_accuracy: {self.intent_accuracies[-1]}\n",
                                  f"{self.mode}/intent_consistency: {self.intent_consistencies[-1]}\n"]
                    if len(episode_lengths_free) > 0:
                        lines += [f"{self.mode}/ep_reward_free: {mean(episode_rewards_free)}\n",
                                  f"{self.mode}/ep_length_free: {mean(episode_lengths_free)}\n",
                        ]
                    if len(episode_rewards_guided) > 0:
                        lines += [f"{self.mode}/ep_reward_guided: {mean(episode_rewards_guided)}\n",
                                  f"{self.mode}/ep_length_guided: {mean(episode_lengths_guided)}\n",
                                  f"{self.mode}/perceived_length_guided: {mean(percieved_lengths_guided)}\n"
                        ]
                    f.writelines(lines)
               
            if mean_reward > self.best_mean_reward:
                if self.verbose >= 1:
                    print("New best mean reward!")
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