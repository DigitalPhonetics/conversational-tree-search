"""
The BASELINE 1 consists of 
- external intent predictor
    - dialog component: handcrafted, similarity matching for NLU
    - faq component: similarity matching

Other possible baseline:
- BASELINE 2
    - external intent predictor
        - dialog component: RL (seperate)
        - faq component: RL (seperate)

Comparison to our system:
- RL 
    - predicts also intent


Metrics for comparison:
- Intent tracker accuracy / F1
- Success
    - Overall
    - Free
    - Guided
- Goal Asked
    - Overall
    - Free
    - Guided



Necessary switches:
- faq/dialog ratio
- train / test data
    - use synonyms
- noise level
"""
from dataclasses import dataclass
from statistics import mean
import os
from encoding.similiarity import AnswerSimilarityEncoding
from chatbot.adviser.app.faqPolicy import FAQPolicy, GuidedPolicy, IntentTracker, Intent

from simulation.dialogenv import DialogEnvironment, EnvironmentMode
from rl.utils import EMBEDDINGS, AutoSkipMode, EnvInfo, safe_division

import random
import numpy as np
import torch
import numpy as np

from data.dataset import GraphDataset

JOINT_DATA = False
RUN_SEEDS = [12345678, 89619, 7201944, 398842, 57063456]

@dataclass
class SimulatorConfig:
    mode: EnvironmentMode
    action_masking: bool
    use_answer_synonyms: bool
    max_steps: int
    user_patience: int
    dialog_faq_ratio: float
    dialogs: int
    dialog_faq_ratio: float
    stop_action: bool
    train_noise: float
    eval_noise: float
    test_noise: float
    auto_skip: AutoSkipMode

@dataclass
class FAQSettings:
    top_k: int


@dataclass
class Experiment:
    cudnn_deterministic: bool


class Evaluator:
    def setUp(self) -> None:
        self.device = "cuda:0" if len(os.environ["CUDA_VISIBLE_DEVICES"].strip()) > 0 else "cpu"

        self.exp_name_prefix = "TOP1_JOINTDATA_test_10noise_synonyms_intentpredictor_similarity"
   
        self.args = {
            "configuration": SimulatorConfig(
                mode = EnvironmentMode.TEST, # For eval: EnvironmentMode.TRAIN
                action_masking = True,
                use_answer_synonyms = True,
                max_steps = 50,
                user_patience = 3,
                dialogs = 500,
                dialog_faq_ratio = 0.5,
                stop_action=False,
                train_noise=0.0,
                eval_noise=0.0,
                test_noise=0.1,
                auto_skip=AutoSkipMode.SIMILARITY
            ),
            "faq_settings": FAQSettings(
                top_k = 1
            ),
            "experiment": Experiment(
                cudnn_deterministic = False,
            )
        }

        torch.backends.cudnn.deterministic = self.args["experiment"].cudnn_deterministic
        config : SimulatorConfig = self.args['configuration']
        print("MODE", config.mode.name) 


        # load text embedding
        text_embedding_name = "distiluse-base-multilingual-cased-v2"
        EMBEDDINGS[text_embedding_name]['args'].pop('cache_db_index')
        self.text_enc = EMBEDDINGS[text_embedding_name]['class'](device=self.device, **EMBEDDINGS[text_embedding_name]['args'])

        self.exp_name = f"BASELINE_{self.exp_name_prefix}"
        os.makedirs(f"/fs/scratch/users/vaethdk/adviser_reisekosten/newruns/{self.exp_name}")
        dialog_logfile = f"/fs/scratch/users/vaethdk/adviser_reisekosten/newruns/{self.exp_name}/dialogs.txt"
        # TODO save config file to this directory

        if JOINT_DATA:
            self.tree = GraphDataset(graph_path='resources/en/traintest_graph.json', answer_path='resources/en/traintest_answers.json', use_answer_synonyms=config.use_answer_synonyms)
        else:    
            self.tree = GraphDataset(graph_path='resources/en/train_graph.json', answer_path='resources/en/train_answers.json', use_answer_synonyms=config.use_answer_synonyms) if config.mode in [EnvironmentMode.TRAIN, EnvironmentMode.EVAL] else GraphDataset(graph_path='resources/en/test_graph.json', answer_path='resources/en/test_answers.json', use_answer_synonyms=config.use_answer_synonyms)

        # load models
        self.sentence_embeddings = AnswerSimilarityEncoding(model_name="distiluse-base-multilingual-cased-v2", dialog_tree=self.tree, device=self.device)
        self.similarity_model = self.sentence_embeddings.similarity_model
        self.intent_tracker = IntentTracker(device=self.device, ckpt_dir='./.models/intentpredictor')

        if config.mode == EnvironmentMode.TRAIN:
            noise = config.train_noise
        elif config.mode == EnvironmentMode.EVAL:
            noise = config.eval_noise
        else:
            noise = config.test_noise

        # load  env
        self.eval_env = DialogEnvironment(dialog_tree=self.tree, adapter=None, stop_action=config.stop_action, 
                                         mode=config.mode,
                                         train_noise=config.train_noise, eval_noise=config.eval_noise, test_noise=config.test_noise,
                                         max_steps=config.max_steps, user_patience=config.user_patience,
                                          auto_skip=AutoSkipMode.NONE, dialog_faq_ratio=config.dialog_faq_ratio,
                                          log_to_file=dialog_logfile, return_obs=False, normalize_rewards=True,
                                          stop_when_reaching_goal=True, similarity_model = self.sentence_embeddings)

        # load policies
        self.guided_policy = GuidedPolicy(similarity_model=self.sentence_embeddings, stop_action=config.stop_action, auto_skip=config.auto_skip, noise=noise)
        self.free_policy = FAQPolicy(dialog_tree=self.tree, similarity_model=self.similarity_model, top_k=self.args['faq_settings'].top_k, noise=noise)


    @torch.no_grad()
    def _play_free_episode(self):
        results = self.free_policy.top_k(query=self.eval_env.initial_user_utterance)

        # TODO: missing_variable? could be 1 if we draw an FAQ with a template
        mode_key = 'faq' if self.eval_env.is_faq_mode else 'dialog'
        for result in results:
            # check if any result matches the goal -> if so, success!
            if result.goal_node_key == self.eval_env.goal_node.key:
                return {
                    f"goal_asked_{mode_key}": 1.0,
                    f"success_{mode_key}": 1.0,
                    "episode_length": 1,
                    "faq_dialog_ratio": 1.0,
                    "ask_variable_irrelevant_ratio": 0.0,
                    "ask_question_irrelevant_ratio": 0.0,
                    "success": 1.0,
                    "goal_asked": 1.0
                }
        # no result matches the goal -> not successful
        return {
            f"goal_asked_{mode_key}": 0.0,
            f"success_{mode_key}": 0.0,
            "episode_length": 1,
            "faq_dialog_ratio": 1.0,
            "ask_variable_irrelevant_ratio": 0.0,
            "ask_question_irrelevant_ratio": 0.0,
            "success": 0.0,
            "goal_asked": 0.0
        }

    @torch.no_grad()
    def _play_guided_episode(self):
        self.guided_policy.reset()

        done = False
        info = None 
        while not done:
            action = self.guided_policy.get_action(self.eval_env.current_node, self.eval_env.current_user_utterance, self.eval_env.last_action_idx)
            _, reward, done, info = self.eval_env.step(action)

        mode_key = 'faq' if self.eval_env.is_faq_mode else 'dialog'
        return {
            "episode_length": float(info[EnvInfo.EPISODE_LENGTH]),
            "success": float(info[EnvInfo.REACHED_GOAL_ONCE]),
            "goal_asked": float(info[EnvInfo.ASKED_GOAL]),
            f"success_{mode_key}": float(info[EnvInfo.REACHED_GOAL_ONCE]),
            f"goal_asked_{mode_key}": float(info[EnvInfo.ASKED_GOAL]),
            "episode_skip_length_ratio": self.eval_env.skipped_nodes / info[EnvInfo.EPISODE_LENGTH],
            "skipped_question_ratio": safe_division(self.eval_env.actioncount_skip_question, self.eval_env.nodecount_question),
            "skipped_variable_ratio": safe_division(self.eval_env.actioncount_skip_variable, self.eval_env.nodecount_variable),
            "skipped_info_ratio": safe_division(self.eval_env.actioncount_skip_info, self.eval_env.nodecount_info),
            "skipped_invalid_ratio": safe_division(self.eval_env.actioncount_skip_invalid, self.eval_env.actioncount_skip),
            "faq_dialog_ratio": 0.0,
            "ask_variable_irrelevant_ratio": safe_division(self.eval_env.actioncount_ask_variable_irrelevant, self.eval_env.actioncount_ask_variable),
            "ask_question_irrelevant_ratio": safe_division(self.eval_env.actioncount_ask_question_irrelevant, self.eval_env.actioncount_ask_question),
            "episode_missing_variable_ratio": self.eval_env.actioncount_missingvariable,
        }


    @torch.no_grad()
    def eval(self, env: DialogEnvironment, eval_dialogs: int, seed) -> float:
        """
        Returns:
            goal_asked score (float)
        """

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        self.similarity_model.eval()

        eval_metrics = {
            # "episode_return": [],
            "episode_length": [],
            "success": [],
            "goal_asked": [],
            "success_faq": [],
            "success_dialog": [],
            "goal_asked_faq": [],
            "goal_asked_dialog": [],
            "episode_skip_length_ratio": [],
            "skipped_question_ratio": [],
            "skipped_variable_ratio": [],
            "skipped_info_ratio": [],
            "skipped_invalid_ratio": [],
            # "stop_prematurely_ratio": [],
            "faq_dialog_ratio": [],
            # "episode_stop_ratio": [],
            "ask_variable_irrelevant_ratio": [],
            "ask_question_irrelevant_ratio": [],
            "episode_missing_variable_ratio": [],
            # "episode_history_wordcount": [],
            # "max_history_wordcount": [0],
        }

        intentprediction_tp = 0
        intentprediction_tn = 0
        intentprediction_fp = 0
        intentprediction_fn = 0

          
        for i in range(eval_dialogs):
            # reset
            self.eval_env.reset()
            info = None

            # intent prediction

            # TODO do this per turn like real intent tracker in RL algorithm or just at the beginning?
            intent = self.intent_tracker.get_intent(self.eval_env.current_node, gen_user_utterance=self.eval_env.initial_user_utterance)
            self.eval_env.episode_log.append(f"Intent prediction: {intent.name}")

            if intent == Intent.FREE:
                # do free
                info = self._play_free_episode()
            else:
                # do guided
                info = self._play_guided_episode()
            assert info

            # evaluate intent tracker
            if self.eval_env.is_faq_mode == False and intent == Intent.FREE:
                intentprediction_fp += 1
            elif self.eval_env.is_faq_mode == False and intent == Intent.GUIDED:
                intentprediction_tn += 1
            elif self.eval_env.is_faq_mode == True and intent == Intent.FREE:
                intentprediction_tp += 1
            elif self.eval_env.is_faq_mode == True and intent == Intent.GUIDED:
                intentprediction_fn += 1
            
            # update global evaluation metrics with current dialog
            for metric in info:
                eval_metrics[metric].append(info[metric])

            self.eval_env.logger.info("\n".join(self.eval_env.episode_log))

        # log metrics (averaged)
        log_dict = {}
        eval_metrics["intentprediction_f1"] = [safe_division(intentprediction_tp, intentprediction_tp + 0.5 * (intentprediction_fp + intentprediction_fn))]
        eval_metrics["intentprediction_recall"] = [safe_division(intentprediction_tp, intentprediction_tp + intentprediction_fn)]
        eval_metrics["intentprediction_precision"] = [safe_division(intentprediction_tp, intentprediction_tp + intentprediction_fp)]
        eval_metrics["intentprediction_accuracy"] = [safe_division(intentprediction_tp + intentprediction_tn, eval_dialogs)]
        for metric in eval_metrics:
            numerical_entries = [num for num in eval_metrics[metric] if num is not None]
            if len(numerical_entries) == 0:
                numerical_entries = [0.0]
            log_dict[f"{metric}"] = mean(numerical_entries)
        print(log_dict)


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "true"

    evaluator = Evaluator()
    evaluator.setUp()

    # call eval() method 
    # NOTE: run seperately for eval / test setting (change config)
    for run, seed in enumerate(RUN_SEEDS):
        print(f"---- RUN {run} with seed {seed} ----")
        evaluator.eval(evaluator.eval_env, evaluator.args['configuration'].dialogs, seed)
    