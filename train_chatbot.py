from omegaconf import OmegaConf
import wandb

from copy import deepcopy
from statistics import mean
from typing import Any, Dict, List
from functools import reduce
from multiprocessing import Process
import os
import time
import random

import hydra
from hydra.core.config_store import ConfigStore

import numpy as np
import torch
import numpy as np
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_sequence

from config import ActionConfig, ConfigEntrypoint, DialogLogLevel, Experiment, WandbLogLevel, register_configs
from data.cache import Cache
from data.dataset import GraphDataset
from encoding.state import StateEncoding
from rl.utils import AutoSkipMode, AverageMetric, EnvInfo, EnvironmentMode, ExperimentLogging, _del_checkpoint, _get_file_hash, _munchausen_stable_logsoftmax, _save_checkpoint, safe_division
from simulation.dialogenv import ParallelDialogEnvironment


cs = ConfigStore.instance()
register_configs()


def to_class(path:str):
    from pydoc import locate
    class_instance = locate(path)
    return class_instance


class Trainer:
    def __init__(self, cfg: ConfigEntrypoint) -> None:
        self.cfg = cfg
        self.exp_name_prefix = "V9_stateencoding_newconfig"
   
        # set random seed
        random.seed(cfg.experiment.seed)
        np.random.seed(cfg.experiment.seed)
        torch.manual_seed(cfg.experiment.seed)
        torch.backends.cudnn.deterministic = cfg.experiment.cudnn_deterministic

        # load dialog tree
        self.train_data = GraphDataset(graph_path=cfg.experiment.training.dataset.graph_path, answer_path=cfg.experiment.training.dataset.answer_path, use_answer_synonyms=cfg.experiment.training.dataset.use_answer_synonyms)
        if cfg.experiment.validation:
            self.val_data = GraphDataset(graph_path=cfg.experiment.validation.dataset.graph_path, answer_path=cfg.experiment.validation.dataset.answer_path, use_answer_synonyms=cfg.experiment.validation.dataset.use_answer_synonyms)
        if cfg.experiment.testing:
            self.test_data = GraphDataset(graph_path=cfg.experiment.testing.dataset.graph_path, answer_path=cfg.experiment.testing.dataset.answer_path, use_answer_synonyms=cfg.experiment.testing.dataset.use_answer_synonyms)
        
        # prepare directories
        if cfg.experiment.logging.dialog_log == DialogLogLevel.FULL:
            self.exp_name = f"{self.exp_name_prefix}_{str(int(100 * cfg.experiment.environment.guided_free_ratio))}dialog_{'actionsinstatespace' if cfg.experiment.actions.in_state_space else 'actionsinactionspace'}"
            self.run_name = f"{self.exp_name}__{cfg.experiment.seed}__{int(time.time())}"
            os.makedirs(f"/mount/arbeitsdaten/asr-2/vaethdk/tmp_debugging_weights/{self.run_name}")
            log_to_file_test = f"/mount/arbeitsdaten/asr-2/vaethdk/tmp_debugging_weights/{self.run_name}/test_dialogs.txt"
            log_to_file_eval = f"/mount/arbeitsdaten/asr-2/vaethdk/tmp_debugging_weights/{self.run_name}/eval_dialogs.txt"
        else:
            log_to_file_test = None
            log_to_file_eval = None
            self.exp_name = self.exp_name_prefix

        self.cache = Cache(device=cfg.experiment.device, data=self.train_data, state_config=cfg.experiment.state, torch_compile=cfg.experiment.torch_compile)
        self.state_enc = StateEncoding(cache=self.cache, state_config=cfg.experiment.state, action_config=cfg.experiment.actions, data=self.train_data)

        self.train_env = ParallelDialogEnvironment(dialog_tree=self.train_data, state_enc=self.state_enc,mode=EnvironmentMode.TRAIN, max_steps=cfg.experiment.environment.max_steps, user_patience=cfg.experiment.environment.user_patience, normalize_rewards=cfg.experiment.environment.normalize_rewards, stop_when_reaching_goal=cfg.experiment.environment.stop_when_reaching_goal, dialog_faq_ratio=cfg.experiment.environment.guided_free_ratio, log_to_file=None,n_envs=cfg.experiment.environment.num_train_envs,auto_skip=cfg.experiment.environment.auto_skip, similarity_model=None)
        if cfg.experiment.validation:
            self.eval_env = ParallelDialogEnvironment(dialog_tree=self.val_data, state_enc=self.state_enc, mode=EnvironmentMode.EVAL, max_steps=cfg.experiment.environment.max_steps, user_patience=cfg.experiment.environment.user_patience, normalize_rewards=cfg.experiment.environment.normalize_rewards, stop_when_reaching_goal=cfg.experiment.environment.stop_when_reaching_goal, dialog_faq_ratio=cfg.experiment.environment.guided_free_ratio, log_to_file=log_to_file_eval,n_envs=cfg.experiment.environment.num_val_envs,auto_skip=cfg.experiment.environment.auto_skip, similarity_model=None)
        if cfg.experiment.testing:
            self.test_env = ParallelDialogEnvironment(dialog_tree=self.test_data, state_enc=self.state_enc, mode=EnvironmentMode.TEST, max_steps=cfg.experiment.environment.max_steps, user_patience=cfg.experiment.environment.user_patience, normalize_rewards=cfg.experiment.environment.normalize_rewards, stop_when_reaching_goal=cfg.experiment.environment.stop_when_reaching_goal, dialog_faq_ratio=cfg.experiment.environment.guided_free_ratio, log_to_file=log_to_file_test,n_envs=cfg.experiment.environment.num_test_envs,auto_skip=cfg.experiment.environment.auto_skip, similarity_model=None)
        
       
        if cfg.experiment.logging.wandb_log != WandbLogLevel.NONE:
            if cfg.experiment.logging.wandb_log == WandbLogLevel.OFFLINE:
                # TODO set wandb api key in env variable: "WANDB_API_KEY"
                os.environ["WANDB_MODE"] = "offline"
            # write code 
            wandb.init(project="cts_en_backport", config=cfg, save_code=True, name=self.exp_name, settings=wandb.Settings(code_dir="/mount/arbeitsdaten/asr-2/vaethdk/cts_en"))
            wandb.config.update({'datasetversion': _get_file_hash(cfg.experiment.training.dataset.graph_path)}) # log dataset version hash
            wandb.watch(self.model, log_freq=cfg.experiment.logging.log_interval)

        #
        # network setup
        #
        self.model = self._dqn_model_from_args(cfg.experiment)
        self.target_network = self._dqn_model_from_args(cfg.experiment)
        self.target_network.load_state_dict(self.model.state_dict())
        self.optimizer = self._optimizer_from_args(cfg.experiment, self.model)

        #
        # buffer setup
        #
        self.rb = self._buffer_from_args(args=cfg.experiment, state_enc=self.state_enc, train_data=self.train_data)

        # Setup train metrics
        self.train_episodic_return = AverageMetric(name='train/episodic_return', running_avg=25)
        self.train_episode_length = AverageMetric(name="train/episode_length", running_avg=25)
        self.train_success = AverageMetric(name="train/success", running_avg=25)
        self.train_goal_asked = AverageMetric(name='train/goal_asked', running_avg=25)

        self.last_save_step = 0
        self.savefile_goal_asked_score = {} # mapping from filename to goal_asked score from evaluation

    def _linear_schedule(self, start_e: float, end_e: float, duration: int, t: int):
        slope = (end_e - start_e) / duration
        return max(slope * t + start_e, end_e)

    def _beta_schedule(self, start_b: float, duration: int, t: int):
        slope = (1.0 - start_b) / duration
        return min(slope * t + start_b, 1.0)

    def _recurse_dict_to_cpu(self, state_dict: dict):
        copied = {}
        for key in state_dict:
            if isinstance(state_dict[key], dict):
                copied[key] = self._recurse_dict_to_cpu(state_dict[key])
            elif isinstance(state_dict[key], torch.Tensor):
                copied[key] = state_dict[key].clone().detach().cpu()
            else:
                copied[key] = deepcopy(state_dict[key])
        return copied

    def _save_checkpoint_with_timeout(self, goal_asked_score: float, global_step: int, episode_counter: int, train_counter: int, epsilon: float, timeout=None):
        if self.cfg.experiment.logging.keep_checkpoints > 0:
            self.last_save_step = global_step

            # find worst checkpoint
            worst_score_file = None
            if len(self.savefile_goal_asked_score) >= self.cfg.experiment.logging.keep_checkpoints:
                for filename in self.savefile_goal_asked_score:
                    if not worst_score_file:
                        worst_score_file = filename
                    if self.savefile_goal_asked_score[filename] < self.savefile_goal_asked_score[worst_score_file]:
                        worst_score_file = filename
                if self.savefile_goal_asked_score[worst_score_file] <= goal_asked_score:
                    # try deleting worst file
                    if timeout:
                        success = False
                        counter = 0
                        while not success and counter < 5:
                            p = Process(target=_del_checkpoint, args=(worst_score_file,))
                            p.start()
                            p.join(timeout=timeout)
                            counter += 1
                            if p.exitcode == 0:
                                success = True
                                del self.savefile_goal_asked_score[worst_score_file] 
                        if not success:
                            print(f"FAILED DELETING 5 times for checkpoint {worst_score_file}")
                    else:
                        _del_checkpoint(worst_score_file)
                        del self.savefile_goal_asked_score[worst_score_file] 

            # save new checkpoint
            if len(self.savefile_goal_asked_score) < self.cfg.experiment.logging.keep_checkpoints:
                if timeout:
                    success = False
                    counter = 0
                    while not success and counter < 5:
                        p = Process(target=_save_checkpoint, args=(global_step, episode_counter, train_counter, self.run_name,
                                                                    self._recurse_dict_to_cpu(self.model.state_dict()),
                                                                    self._recurse_dict_to_cpu(self.optimizer.state_dict()),
                                                                    epsilon,
                                                                    torch.get_rng_state().clone().detach().cpu(),
                                                                    np.random.get_state(),
                                                                    random.getstate()))
                        p.start()
                        p.join(timeout=timeout)
                        p.terminate()
                        counter += 1
                        if p.exitcode == 0:
                            success = True
                            self.savefile_goal_asked_score[f"/mount/arbeitsdaten/asr-2/vaethdk/tmp_debugging_weights/{self.run_name}/ckpt_{global_step}.pt"] = goal_asked_score
                    if not success:
                        print(f"FAILED SAVING 5 times for checkpoint at step {global_step}")
                else:
                    _save_checkpoint(global_step, episode_counter, train_counter, self.run_name,
                                                                    self._recurse_dict_to_cpu(self.model.state_dict()),
                                                                    self._recurse_dict_to_cpu(self.optimizer.state_dict()),
                                                                    epsilon,
                                                                    torch.get_rng_state().clone().detach().cpu(),
                                                                    np.random.get_state(),
                                                                    random.getstate())
                    self.savefile_goal_asked_score[f"/mount/arbeitsdaten/asr-2/vaethdk/tmp_debugging_weights/{self.run_name}/ckpt_{global_step}.pt"] = goal_asked_score
    

    def _dqn_model_from_args(self, args: Experiment):
        kwargs = {
            "state_enc": self.state_enc,
            "dropout_rate": args.policy.net_arch.dropout_rate,
            "activation_fn": to_class(args.policy.activation_fn),
            "normalization_layers": args.policy.net_arch.normalization_layers,
            "q_value_clipping": args.algorithm.dqn.q_value_clipping,
        }
        model_cls = to_class(args.policy.net_arch.net_cls)
        if 'dueling' in args.policy.net_arch.net_cls.lower():
            kwargs |= {
                "shared_layer_sizes": args.policy.net_arch.shared_layer_sizes,
                "advantage_layer_sizes": args.policy.net_arch.advantage_layer_sizes,
                "value_layer_sizes": args.policy.net_arch.value_layer_sizes,
            }
        else:
            kwargs |= {
                "hidden_layer_sizes": args.policy.net_arch.hidden_layer_sizes
            }
        model = model_cls(**kwargs).to(args.device)
        if args.torch_compile:
            model = torch.compile(model)
        return model

    def _optimizer_from_args(self, args: Experiment, model: torch.nn.Module):
        optim = to_class(args.optimizer.class_path)
        return optim(params=model.parameters(), lr=args.optimizer.lr)
    
    def _buffer_from_args(self, args: Experiment, state_enc: StateEncoding, train_data: GraphDataset):
        buffer_args = OmegaConf.to_container(args.algorithm.dqn.buffer)
        buffer_cls = to_class(buffer_args.pop('_target_'))
        kwargs = {
            "state_enc": state_enc
        } | buffer_args
        if args.algorithm.dqn.buffer._target_.endswith("HindsightExperienceReplay"):
            # HER buffer
            her_env = ParallelDialogEnvironment(dialog_tree=train_data, state_enc=state_enc,mode=EnvironmentMode.TRAIN, max_steps=args.environment.max_steps, user_patience=args.environment.user_patience, normalize_rewards=args.environment.normalize_rewards, stop_when_reaching_goal=args.environment.stop_when_reaching_goal, dialog_faq_ratio=args.environment.guided_free_ratio, log_to_file=None,n_envs=1,auto_skip=args.environment.auto_skip, similarity_model=None)
            kwargs |= {
                "envs": her_env,
                "dialog_tree": train_data,
                "answerParser": her_env.answer_template_parser,
                "logicParser": her_env.logicParser,
                "max_reward": her_env.max_reward,
                "experiment_logging": ExperimentLogging.NONE,
                "auto_skip": AutoSkipMode.NONE,
                "stop_when_reaching_goal": her_env.stop_when_reaching_goal,
                "similarity_model": None
            }
        return buffer_cls(**kwargs)

    @torch.no_grad()
    def eval(self, env: ParallelDialogEnvironment, eval_dialogs: int, eval_phase: int, prefix: str) -> float:
        """
        Returns:
            goal_asked score (float)
        """
        self.model.eval()

        if self.cfg.experiment.logging.dialog_log != ExperimentLogging.NONE and env.log_to_file:
            env.logger.info(f"=========== EVAL AT STEP {eval_dialogs}, PHASE {eval_phase} ============")
        
        eval_metrics = {
            "episode_return": [],
            "episode_length": [],
            "success": [],
            "goal_asked": [],
            "success_faq": [],
            "success_dialog": [],
            "goal_asked_faq": [],
            "goal_asked_dialog": [],
            "episode_skip_length_ratio": [],
            "skip_length_ratio_faq": [],
            "skip_length_ratio_dialog": [],
            "skipped_question_ratio": [],
            "skipped_variable_ratio": [],
            "skipped_info_ratio": [],
            "skipped_invalid_ratio": [],
            "faq_dialog_ratio": [],
            "ask_variable_irrelevant_ratio": [],
            "ask_question_irrelevant_ratio": [],
            "episode_missing_variable_ratio": [],
            "episode_history_wordcount": [],
            "max_history_wordcount": [0],
            "intentprediction_consistency": []
        }
        if 'intent' in self.cfg.experiment.policy.net_arch.net_cls.lower():
            intentprediction_tp = 0
            intentprediction_tn = 0
            intentprediction_fp = 0
            intentprediction_fn = 0
          
        batch_size = self.cfg.experiment.algorithm.dqn.batch_size
        num_dialogs = 0
        obs = env.reset()
        intent_history = [[] for _ in range(batch_size)]
        while num_dialogs < eval_dialogs:
            # state = [self.adapter.state_vector(env_obs) for env_obs in obs]
            state = self.state_enc.batch_state_vector_from_obs(obs, batch_size)
            node_keys = env.current_nodes_keys
            if self.algorithm == 'dqn':
                if self.state_enc.action_config.in_state_space:
                    actions, intent_classes = self.model.select_actions_eps_greedy(node_keys=node_keys, state_vectors=torch.cat(state, dim=0), epsilon=0.0)
                else:
                    actions, intent_classes = self.model.select_actions_eps_greedy(node_keys=node_keys, state_vectors=pack_sequence(state, enforce_sorted=False), epsilon=0.0)
            obs, rewards, dones, infos = env.step(actions)
            if torch.is_tensor(intent_classes):
                for idx, intent in enumerate(intent_classes.tolist()):
                    intent_history[idx].append(intent)

            for done_idx, done in enumerate(dones):
                if done and num_dialogs < eval_dialogs:
                    info = infos[done_idx]
                    env_instance = env.envs[done_idx]
                    # update metrics
                    eval_metrics["episode_return"].append(info[EnvInfo.EPISODE_REWARD])
                    eval_metrics["episode_length"].append(info[EnvInfo.EPISODE_LENGTH])
                    eval_metrics["success"].append(float(info[EnvInfo.REACHED_GOAL_ONCE]))
                    eval_metrics["goal_asked"].append(float(info[EnvInfo.ASKED_GOAL]))
                    if env_instance.is_faq_mode:
                        eval_metrics["success_faq"].append(1.0 if info[EnvInfo.REACHED_GOAL_ONCE] else 0.0)
                        eval_metrics["goal_asked_faq"].append(1.0 if info[EnvInfo.ASKED_GOAL] else 0.0)
                        eval_metrics["skip_length_ratio_faq"].append(env_instance.skipped_nodes / info[EnvInfo.EPISODE_LENGTH])
                    else:
                        eval_metrics["success_dialog"].append(info[EnvInfo.REACHED_GOAL_ONCE])
                        eval_metrics["goal_asked_dialog"].append(info[EnvInfo.ASKED_GOAL])
                        eval_metrics["skip_length_ratio_dialog"].append(env_instance.skipped_nodes / info[EnvInfo.EPISODE_LENGTH])
                    eval_metrics["episode_skip_length_ratio"].append(env_instance.skipped_nodes / info[EnvInfo.EPISODE_LENGTH])
                    eval_metrics["skipped_question_ratio"].append(safe_division(env_instance.actioncount_skip_question, env_instance.nodecount_question))
                    eval_metrics["skipped_variable_ratio"].append(safe_division(env_instance.actioncount_skip_variable, env_instance.nodecount_variable))
                    eval_metrics["skipped_info_ratio"].append(safe_division(env_instance.actioncount_skip_info, env_instance.nodecount_info))
                    eval_metrics["skipped_invalid_ratio"].append(safe_division(env_instance.actioncount_skip_invalid, env_instance.actioncount_skip))
                    eval_metrics["faq_dialog_ratio"].append(1.0 if env_instance.is_faq_mode else 0.0)
                    eval_metrics["ask_variable_irrelevant_ratio"].append(safe_division(env_instance.actioncount_ask_variable_irrelevant, env_instance.actioncount_ask_variable))
                    eval_metrics["ask_question_irrelevant_ratio"].append(safe_division(env_instance.actioncount_ask_question_irrelevant, env_instance.actioncount_ask_question))
                    eval_metrics["episode_missing_variable_ratio"].append(env_instance.actioncount_missingvariable)
                    hist_word_count = env_instance.get_history_word_count()
                    eval_metrics["episode_history_wordcount"].append(hist_word_count)
                    if hist_word_count > eval_metrics['max_history_wordcount'][0]:
                        eval_metrics['max_history_wordcount'] = [hist_word_count]
                    num_dialogs += 1

                    if torch.is_tensor(intent_classes):
                        intent_class_a_count = intent_history[done_idx].count(0)
                        intent_class_b_count = intent_history[done_idx].count(1)

                        # intent inconsistency: ratio number of intent classes in 1 dialog (1.0 if different each turn, 0.0 if same each turn)
                        # -> consistency: 1 - inconsistency
                        intent_inconsistency = intent_class_a_count / intent_class_b_count if intent_class_a_count < intent_class_b_count else intent_class_b_count / intent_class_a_count
                        eval_metrics["intentprediction_consistency"].append(1.0 - intent_inconsistency)

                        # calculate majority class (if more class 1 -> True, if more class 0 -> False)
                        majority_class = int(intent_class_b_count > intent_class_a_count) 
                        if info[EnvInfo.IS_FAQ] == False and majority_class == 0:
                            intentprediction_tn += 1
                        elif info[EnvInfo.IS_FAQ] == False and majority_class == 1:
                            intentprediction_fp += 1
                        elif info[EnvInfo.IS_FAQ] == True and majority_class == 1:
                            intentprediction_tp += 1
                        elif info[EnvInfo.IS_FAQ] == True and majority_class == 0:
                            intentprediction_fn += 1
                            
                    if self.cfg.experiment.logging.dialog_log != ExperimentLogging.NONE and env_instance.log_to_file:
                        env_instance.logger.info("\n".join(env_instance.episode_log))
                    
                    intent_history[done_idx] = [] # reset intent history
                    obs[done_idx] = env_instance.reset()

        # log metrics (averaged)
        log_dict = {
            f"{prefix}/coverage_faqs": env.get_coverage_faqs(),
            f"{prefix}/coverage_synonyms": env.get_coverage_synonyms(),
            f"{prefix}/coverage_variables": env.get_coverage_variables(),
            f"{prefix}/coverage_goal_nodes_free": env.get_goal_node_coverage_free(),
            f"{prefix}/coverage_goal_nodes_guided": env.get_goal_node_coverage_guided(),
            f"{prefix}/coverage_nodes": env.get_node_coverage(),
        }
        if 'intent' in self.cfg.experiment.policy.net_arch.net_cls.lower():
            eval_metrics["intentprediction_f1"] = [safe_division(intentprediction_tp, intentprediction_tp + 0.5 * (intentprediction_fp + intentprediction_fn))]
            eval_metrics["intentprediction_recall"] = [safe_division(intentprediction_tp, intentprediction_tp + intentprediction_fn)]
            eval_metrics["intentprediction_precision"] = [safe_division(intentprediction_tp, intentprediction_tp + intentprediction_fp)]
            eval_metrics["intentprediction_accuracy"] = [safe_division(intentprediction_tp + intentprediction_tn, num_dialogs)]
        for metric in eval_metrics:
            numerical_entries = [num for num in eval_metrics[metric] if num is not None]
            if len(numerical_entries) == 0:
                numerical_entries = [0.0]
            log_dict[f"{prefix}/{metric}"] = mean(numerical_entries)
        if self.cfg.experiment.logging.wandb_log != WandbLogLevel.NONE:
            wandb.log(log_dict, step=eval_phase)

        self.model.train()
        return mean(eval_metrics["goal_asked"])

    def log_train_step(self, global_step: int, train_step: int, epsilon: float, timesteps_per_reset: int, beta: float):
        if train_step % 50 == 0 and self.cfg.experiment.logging.wandb_log != WandbLogLevel.NONE:
            log_dict = {
                "train/learning_phase": global_step // timesteps_per_reset,
                "train/coverage_faqs": self.train_env.get_coverage_faqs(),
                "train/coverage_synonyms": self.train_env.get_coverage_synonyms(),
                "train/coverage_variables": self.train_env.get_coverage_variables(),
                "train/coverage_goal_nodes_free": self.train_env.get_goal_node_coverage_free(),
                "train/coverage_goal_nodes_guided": self.train_env.get_goal_node_coverage_guided(),
                "train/coverage_nodes": self.train_env.get_node_coverage(),
            }
            if self.algorithm == "dqn":
                log_dict["train/epsilon"] = epsilon
                
                if 'prioritized' in self.cfg.experiment.algorithm.dqn.buffer._target_.lower() or 'hindsight' in self.cfg.experiment.algorithm.dqn.buffer._target_.lower():
                    log_dict["train/priority_beta"] = beta
                log_dict["train/buffer_size"] = len(self.rb)
            if self.train_env.current_episode > 0:
                log_dict["train/faq_dialog_ratio"] = self.train_env.num_faqbased_dialogs / self.train_env.current_episode
            wandb.log(log_dict, step=global_step, commit=(global_step % 250) == 0)

    def store_dqn(self, observations: List[torch.FloatTensor], next_observations: List[torch.FloatTensor], actions: List[int], rewards: List[float], dones: List[bool], infos: List[dict], global_step: int):
        for env_id, (obs, next_obs, action, reward, done, info) in enumerate(zip(observations, next_observations, actions, rewards, dones, infos)):
            self.rb.add(env_id, obs, next_obs, action, reward, done, info, global_step)

    @torch.no_grad()
    def _munchausen_target(self, next_observations, data, q_prev: torch.FloatTensor):
        tau = self.cfg.experiment.algorithm.dqn.targets.tau
        q_next = self.target_network(next_observations)[0] # batch x actions
        mask = q_next > float('-inf')
        sum_term =  F.softmax(q_next / tau, dim=-1) * (q_next - _munchausen_stable_logsoftmax(q_next, tau)) # batch x actions
        log_policy = _munchausen_stable_logsoftmax(q_prev, tau).gather(-1, data.actions).view(-1) # batch x actions -> batch
        if self.cfg.experiment.algorithm.dqn.targets.clipping != 0:
            log_policy = torch.clip(log_policy, min=self.cfg.experiment.algorithm.dqn.targets.clipping, max=0)
        return data.rewards.flatten() + self.cfg.experiment.algorithm.dqn.targets.alpha*log_policy + self.cfg.experiment.algorithm.dqn.gamma * sum_term.masked_fill(~mask, 0.0).sum(-1) * (1.0 - data.dones.flatten())

    @torch.no_grad()
    def _td_target(self, next_observations, data):
        target_pred, _ = self.target_network(next_observations)
        target_max, _ = target_pred.max(dim=1) # output[1] would be predicted intent classes
        return data.rewards.flatten() + self.cfg.experiment.algorithm.dqn.gamma * target_max * (1 - data.dones.flatten())


    def train_step_dqn(self, global_step: int, train_counter: int):
        data = self.rb.sample(self.cfg.experiment.algorithm.dqn.batch_size)

        # observations = [self.adapter.state_vector({ key: data.observations[key][index] for key in data.observations}) for index in range(self.args['algorithm']["batch_size"])]
        # next_observations = [self.adapter.state_vector({ key: data.next_observations[key][index] for key in data.next_observations}) for index in range(self.args['algorithm']["batch_size"])]
        observations = self.state_enc.batch_state_vector(data.observations, self.cfg.experiment.algorithm.dqn.batch_size)
        next_observations = self.state_enc.batch_state_vector(data.next_observations, self.cfg.experiment.algorithm.dqn.batch_size)
        
        if self.state_enc.action_config.in_state_space:
            observations = torch.cat(observations, dim=0)
            next_observations = torch.cat(next_observations, dim=0)
        else:
            observations = pack_sequence(observations, enforce_sorted=False)
            next_observations = pack_sequence(next_observations, enforce_sorted=False)

        old_val, intent_logits = self.model(observations)
        if "MuenchausenTarget" in self.cfg.experiment.algorithm.dqn.targets._target_:
            td_target = self._munchausen_target(next_observations, data, old_val)
        else:
            td_target = self._td_target(next_observations, data)
        old_val = old_val.gather(1, data.actions).squeeze()

        # loss
        loss = F.huber_loss(td_target, old_val, reduction="none")
        intent_loss = 0 if not torch.is_tensor(intent_logits) else F.binary_cross_entropy_with_logits(intent_logits.view(-1), torch.tensor(data.infos[EnvInfo.IS_FAQ], dtype=torch.float, device=self.cfg.experiment.device), reduction="none")
        if 'prioritized' in self.cfg.experiment.algorithm.dqn.buffer._target_.lower() or 'hindsight' in self.cfg.experiment.algorithm.dqn.buffer._target_.lower():
            loss = loss * data.weights
            if not isinstance(intent_logits, type(None)):
                intent_loss = intent_loss * data.weights
            # update priorities
            with torch.no_grad():
                td_error = torch.abs(td_target - old_val)
                self.rb.update_weights(data.indices, td_error)
            # scale gradients by priority weights
        loss = loss.mean(-1) # reduce loss
        if not isinstance(intent_logits, type(None)):
            intent_loss = intent_loss.mean(-1) # reduce loss
        
        if self.cfg.experiment.logging.wandb_log != WandbLogLevel.NONE:
            log_dict = {"train/loss": loss.item(),
                        "train/q_values": old_val.mean().item()}
            if not isinstance(intent_logits, type(None)):
                log_dict['train/intent_loss'] = intent_loss.item()
            if 'prioritized' in self.cfg.experiment.algorithm.dqn.buffer._target_.lower() or 'hindsight' in self.cfg.experiment.algorithm.dqn.buffer._target_.lower():
                log_dict['train/priorization_weights'] = data.weights.mean().item()
            wandb.log(log_dict, step=global_step, commit=(train_counter % 250 == 0))

        # optimize the model
        loss += intent_loss
        self.optimizer.zero_grad()
        loss.backward()

        if self.cfg.experiment.algorithm.dqn.max_grad_norm:
            torch.nn.utils.clip_grad_value_(self.model.parameters(), self.cfg.experiment.algorithm.dqn.max_grad_norm)
        self.optimizer.step()

        # update the target network
        if train_counter % self.cfg.experiment.algorithm.dqn.target_network_update_frequency == 0:
            self.target_network.load_state_dict(self.model.state_dict())


    def train_loop_dqn(self):
        evaluation = not isinstance(self.cfg.experiment.validation, type(None))
        eval_every_train_timesteps = self.cfg.experiment.validation.every_steps
        eval_dialogs = self.cfg.experiment.validation.dialogs

        #
        # agent environment loop
        #
        timesteps_per_reset = self.cfg.experiment.algorithm.dqn.target_network_update_frequency
        learning_phases = self.cfg.experiment.algorithm.dqn.reset_exploration_times + 1 # 0 resets = 1 run
        total_timesteps = timesteps_per_reset * learning_phases

        self.model.train()
        obs: List[Dict[str, Any]] = self.train_env.reset()

        global_step = 0
        train_counter = 0
        episode_counter = 0

        # initial evaluation
        # self.eval(self.eval_env, eval_dialogs, global_step, prefix="eval")
        
        while global_step < total_timesteps:
            epsilon = self._linear_schedule(self.cfg.experiment.algorithm.dqn.eps_start,self.cfg.experiment.algorithm.dqn.eps_end, self.cfg.experiment.algorithm.dqn.exploration_fraction * timesteps_per_reset, global_step % timesteps_per_reset)
            beta = self._beta_schedule(self.cfg.experiment.algorithm.dqn.buffer.beta, self.cfg.experiment.algorithm.dqn.exploration_fraction * timesteps_per_reset, global_step % timesteps_per_reset)

            # state = [self.adapter.state_vector(env_obs) for env_obs in obs]
            state = self.state_enc.batch_state_vector_from_obs(obs, self.cfg.experiment.algorithm.dqn.batch_size)
            # choose and perform next action
            # state = [[env_obs[key] for key in env_obs if torch.is_tensor(env_obs[key])] for env_obs in obs]
            if self.state_enc.action_config.in_state_space:
                actions, _ = self.model.select_actions_eps_greedy(self.train_env.current_nodes_keys, torch.cat(state, dim=0), epsilon)
            else:
                actions, _ = self.model.select_actions_eps_greedy(self.train_env.current_nodes_keys, pack_sequence(state, enforce_sorted=False), epsilon)
                
            next_obs, rewards, dones, infos = self.train_env.step(actions)

            # update buffer and logs
            self.store_dqn(obs, next_obs, actions, rewards, dones, infos, global_step)
          
            obs = next_obs
            for done_idx, done in enumerate(dones):
                if done:
                    # restart finished environment & log results
                    episode_counter += 1
                    info = infos[done_idx]

                    self.train_episodic_return.log(info[EnvInfo.EPISODE_REWARD])
                    self.train_episode_length.log(info[EnvInfo.EPISODE_LENGTH])
                    self.train_success.log(float(info[EnvInfo.REACHED_GOAL_ONCE]))
                    self.train_goal_asked.log(float(info[EnvInfo.ASKED_GOAL]))

                    if self.cfg.experiment.logging.wandb_log != WandbLogLevel.NONE and episode_counter % self.train_episodic_return.running_avg == 0:
                        wandb.log({
                            self.train_episodic_return.name: self.train_episodic_return.eval(),
                            self.train_episode_length.name: self.train_episode_length.eval(),
                            self.train_success.name: self.train_success.eval(),
                            self.train_goal_asked.name: self.train_goal_asked.eval()
                        }, step=global_step, commit=(global_step % 250 == 0))
                    obs[done_idx] = self.train_env.reset_single(done_idx)

                global_step += 1
                #
                # Train
                #
             
                if len(self.rb) >= self.cfg.experiment.algorithm.dqn.learning_starts and global_step % self.cfg.experiment.training.every_steps == 0:
                    if 'prioritized' in self.cfg.experiment.algorithm.dqn.buffer._target_.lower() or 'hindsight' in self.cfg.experiment.algorithm.dqn.buffer._target_.lower():
                        self.rb.update_beta(beta)
                    self.train_step_dqn(global_step, train_counter)
                    train_counter += 1
                self.log_train_step(global_step=global_step, train_step=train_counter, epsilon=epsilon, timesteps_per_reset=timesteps_per_reset, beta=beta)

                #
                # Eval
                #
                if evaluation and global_step % eval_every_train_timesteps == 0:
                    eval_goal_asked_score = self.eval(self.eval_env, eval_dialogs, global_step, prefix="eval")
                    self.eval(self.test_env, eval_dialogs, global_step, prefix="test")
                    self._save_checkpoint_with_timeout(goal_asked_score=eval_goal_asked_score, global_step=global_step, episode_counter=episode_counter, train_counter=train_counter, epsilon=epsilon, timeout=300)
          
        self.train_env.close()

    def _concat_tensors(self, tensors: List[torch.Tensor]):
        if self.spaceadapter_config.action_config == ActionConfig.ACTIONS_IN_ACTION_SPACE:
            return torch.cat(tensors, dim=0).to(self.cfg.experiment.device)
        else:
            return pack_sequence([tensor.to(self.cfg.experiment.device) for tensor in tensors], enforce_sorted=False)

    def _flatten_list(self, multidim_list):
        return reduce(lambda sublist1, sublist2: sublist1 + sublist2, multidim_list)




@hydra.main(version_base=None, config_path="conf", config_name="default")
def load_cfg(cfg) -> None:
    trainer = Trainer(cfg)
    trainer.train_loop_dqn()


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    trainer = load_cfg()
    