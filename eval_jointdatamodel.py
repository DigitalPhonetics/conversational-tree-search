from copy import deepcopy
from statistics import mean
from typing import Any, Dict, List
import wandb
from functools import reduce
import redisai as rai
from multiprocessing import Process

import json
import os
from chatbot.adviser.app.answerTemplateParser import AnswerTemplateParser
from chatbot.adviser.app.encoding.similiarity import AnswerSimilarityEncoding
from chatbot.adviser.app.encoding.text import TextEmbeddingPooling

from chatbot.adviser.app.rl.dialogenv import EnvironmentMode, ParallelDialogEnvironment
from chatbot.adviser.app.rl.dialogtree import DialogTree
import chatbot.adviser.app.rl.dataset as Data
from chatbot.adviser.app.rl.layers.attention.attention_factory import AttentionActivationConfig, AttentionMechanismConfig, AttentionVectorAggregation
from chatbot.adviser.app.rl.spaceAdapter import AnswerSimilarityEmbeddingConfig, IntentEmbeddingConfig, SpaceAdapter, ActionConfig, SpaceAdapterAttentionInput, SpaceAdapterAttentionQueryInput, SpaceAdapterConfiguration, SpaceAdapterSpaceInput, TextEmbeddingConfig
from chatbot.adviser.app.rl.utils import EMBEDDINGS, AutoSkipMode, AverageMetric, EnvInfo, ExperimentLogging, _del_checkpoint, _get_file_hash, _munchausen_stable_logsoftmax, _munchausen_stable_softmax, _save_checkpoint, safe_division


import time
import random
import numpy as np
import torch
import numpy as np
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_sequence

# noise levels: 0.0, 0.1, 0.25, 0.5, 0,75, 1.0, 1.5, 2.0

EXPERIMENT_LOGGING = ExperimentLogging.ONLINE
CHECKPOINT_LOCATION = "/mount/arbeitsdaten/asr-2/vaethdk/adviser_reisekosten/newruns/V8_JOINTDATA_ROBERTA_10NOISE_25DROPOUT_ACTIONPOS_dqn_50dialog_1_cross-en-de-roberta-sentence-transformer_nouser_intent_prediction__12345678__1666219207/"
RUN_SEEDS = [12345678, 89619, 7201944, 398842, 57063456]
DIALOGS = 500

class Evaluator:
    def setUp(self) -> None:
        self.device = "cuda:0" if len(os.environ["CUDA_VISIBLE_DEVICES"].strip()) > 0 else "cpu"

        seed = 12345678
        self.run_name = "EVAL_0NOISE_V8_JOINTDATA_ROBERTA_10NOISE_25DROPOUT_ACTIONPOS"
   
        self.args = {
            "spaceadapter": {
                "configuration": SpaceAdapterConfiguration(
                    text_embedding="cross-en-de-roberta-sentence-transformer", #'distiluse-base-multilingual-cased-v2', # 'gbert-large' # 'cross-en-de-roberta-sentence-transformer',
                    action_config=ActionConfig.ACTIONS_IN_STATE_SPACE,
                    action_masking=True,
                    stop_action=False,
                    auto_skip=AutoSkipMode.NONE,
                    use_answer_synonyms=True
                ),
                "state": SpaceAdapterSpaceInput(
                    last_system_action=True,
                    beliefstate=True,
                    current_node_position=True,
                    current_node_type=True,
                    user_intent_prediction=IntentEmbeddingConfig(
                        active=False,
                        ckpt_dir='./.models/intentpredictor'
                    ),
                    answer_similarity_embedding=AnswerSimilarityEmbeddingConfig(
                        active=False,
                        model_name='distiluse-base-multilingual-cased-v2',
                        caching=True,
                    ),
                    dialog_node_text=TextEmbeddingConfig(
                        active=True,
                        pooling=TextEmbeddingPooling.MEAN,
                        caching=True,
                    ),
                    original_user_utterance=TextEmbeddingConfig(
                        active=True,
                        pooling=TextEmbeddingPooling.MEAN,
                        caching=True,
                    ),
                    current_user_utterance=TextEmbeddingConfig(
                        active=True,
                        pooling=TextEmbeddingPooling.MEAN,
                        caching=True,
                    ),
                    dialog_history=TextEmbeddingConfig(
                        active=True,
                        pooling=TextEmbeddingPooling.MEAN,
                        caching=False,
                    ),
                    action_text=TextEmbeddingConfig(
                        active=True,
                        pooling=TextEmbeddingPooling.MEAN,
                        caching=True,
                    ),
                    action_position=True
                ),
                "attention": [
                    SpaceAdapterAttentionInput(
                        active=False,
                        name="utterance_nodetext_attn",
                        queries=SpaceAdapterAttentionQueryInput(
                            input=['current_user_utterance',
                                    'original_user_utterance'],
                            pooling=TextEmbeddingPooling.CLS,
                            aggregation=AttentionVectorAggregation.SUM,
                            caching=True,
                            allow_noise=True
                        ),
                        matrix="dialog_node_text",
                        activation=AttentionActivationConfig.NONE,
                        attention_mechanism=AttentionMechanismConfig.ADDITIVE,
                        caching=False,
                        allow_noise=False
                    ),
                    SpaceAdapterAttentionInput(
                        active=False,
                        name="utterance_history_attn",
                        queries=SpaceAdapterAttentionQueryInput(
                            input=['current_user_utterance',
                                    'original_user_utterance'],
                            pooling=TextEmbeddingPooling.CLS,
                            aggregation=AttentionVectorAggregation.MAX,
                            caching=True,
                            allow_noise=True
                        ),
                        matrix="dialog_history",
                        activation=AttentionActivationConfig.NONE,
                        attention_mechanism=AttentionMechanismConfig.ADDITIVE,
                        caching=False,
                        allow_noise=False
                    )
                ]
            },
            "simulation": {
                "normalize_rewards": True,
                "max_steps": 50,
                "user_patience": 3,
                "stop_when_reaching_goal": True,
                "dialog_faq_ratio": 0.5,
                "parallel_train_envs": 128,
                "parallel_test_envs": 128,
                "train_noise": 0.0,
                "eval_noise": 0.0,
                "test_noise": 0.0
            },
            "experiment": {
                "seed": seed,
                "cudnn_deterministic": False,
                "keep": 5
            },
            "model": {
                "architecture": "new_dueling", # 'dueling', 'vanilla', "new_dueling"
                "shared_layer_sizes": [8096, 4096, 4096],
                "value_layer_sizes": [2048, 1024],
                "advantage_layer_sizes": [4096, 2048, 1024],
                "hidden_layer_sizes": [4096, 2048, 1024],
                "dropout": 0.25,
                "activation_fn": "SELU",
                "normalization_layers": False,
                "intentprediction": True # True # False
            },
            "optimizer": {
                "name": "Adam",
                "lr": 0.0001
            },
            "algorithm": {
                "timesteps_per_reset": 1500000,
                "reset_exploration_times": 0,
                "max_grad_norm": 1.0,
                "batch_size": 128,
                "gamma": 0.99,
                "algorithm": "dqn", # "ppo", "dqn"
            },
            "ppo": {
                "T": 4, # timesteps per actor (<< episode length) included in one minibatch => parallel actors = batch_size // T2,
                'update_epochs': 10,
                'minibatch_size': 64
            },
            "dqn": {
                "buffer_size": 100000,
                "buffer_type": "HER", # "prioritized", "LAP", # "uniform", # "HER"
                "priority_replay_alpha": 0.6,
                "priority_replay_beta": 0.4,
                "exploration_fraction": 0.99,
                "eps_start": 0.6,
                "eps_end": 0.0,
                "train_frequency": 3,
                "learning_starts": 1280,
                "target_network_frequency": 15,
                "q_value_clipping": 10.0,
                "munchausen_targets": True,
                "munchausen_tau": 0.03,
                "munchausen_alpha": 0.9,
                "munchausen_clipping": -1
            },
            "evaluation": {
                "evaluation": True,
                "every_train_timesteps": 10000,
                "dialogs": 500
            }
        }


        # set random seed
        random.seed(self.args["experiment"]["seed"])
        np.random.seed(self.args["experiment"]["seed"])
        torch.manual_seed(self.args["experiment"]["seed"])
        torch.backends.cudnn.deterministic = self.args["experiment"]["cudnn_deterministic"]

        # load dialog tree
        self.tree = DialogTree(version=0)
        self.eval_tree = DialogTree(version=1)

        # load text embedding
        text_embedding_name = self.args['spaceadapter']['configuration'].text_embedding
        self.cache_conn = rai.Client(host='localhost', port=64123, db=EMBEDDINGS[text_embedding_name]['args'].pop('cache_db_index'))
        self.text_enc = EMBEDDINGS[text_embedding_name]['class'](device=self.device, **EMBEDDINGS[text_embedding_name]['args'])

        # post-init spaceadapter 
        self.spaceadapter_config: SpaceAdapterConfiguration = self.args['spaceadapter']['configuration']
        self.spaceadapter_state: SpaceAdapterSpaceInput = self.args['spaceadapter']['state']
        self.spaceadapter_attention: List[SpaceAdapterAttentionInput] = self.args['spaceadapter']['attention']
        self.spaceadapter_config.post_init(tree=self.tree)
        self.spaceadapter_state.post_init(device=self.device, tree=self.tree, text_embedding=self.text_enc, action_config=self.spaceadapter_config.action_config, action_masking=self.spaceadapter_config.action_masking, stop_action=self.spaceadapter_config.stop_action, cache_connection=self.cache_conn)
        for attn in self.spaceadapter_attention:
            attn.post_init(device=self.device, tree=self.tree, text_embedding=self.text_enc, action_config=self.spaceadapter_config.action_config, action_masking=self.spaceadapter_config.action_masking, cache_connection=self.cache_conn)

        # prepare directories
        spaceadapter_json = self.spaceadapter_config.toJson() | self.spaceadapter_state.toJson() | {"attention": [attn.toJson() for attn in self.spaceadapter_attention]}
        if EXPERIMENT_LOGGING != ExperimentLogging.NONE:
            os.makedirs(f"/fs/scratch/users/vaethdk/adviser_reisekosten/newruns/{self.run_name}")
            self.args['simulation']["log_to_file"] = f"/fs/scratch/users/vaethdk/adviser_reisekosten/newruns/{self.run_name}/dialogs.txt"
        else:
            self.args['simulation']["log_to_file"] = None

        # init spaceadapter
        self.adapter = SpaceAdapter(device=self.device, dialog_tree=self.tree, **self.args["spaceadapter"])
        self.algorithm = self.args['algorithm']['algorithm']

        if self.algorithm == 'dqn':
            self.n_train_envs = self.args['simulation'].pop('parallel_train_envs')
            self.n_test_envs = self.args['simulation'].pop('parallel_test_envs')
            # assert self.args['algorithm']['batch_size'] > self.args['dqn']['train_frequency'], "Training batch size should be larger than the train frequency to avoid bias to most recent transitions only"
        else:
            assert False, f"Unknown algorithm: {self.algorithm}"

        # init auto-skip model
        similarity_model = None
        if self.spaceadapter_config.auto_skip != AutoSkipMode.NONE:
            if not isinstance(self.adapter.stateinput.answer_similarity_embedding, type(None)):
                similarity_model = self.adapter.stateinput.encoders['action_answer_similarity_embedding']
            else:
                similarity_model = AnswerSimilarityEncoding(model_name="distiluse-base-multilingual-cased-v2", dialog_tree=self.tree, device=self.device, caching=True)
       
        dialog_faq_ratio = self.args['simulation'].pop('dialog_faq_ratio')
        self.eval_env = ParallelDialogEnvironment(dialog_tree=self.tree, adapter=self.adapter, stop_action=self.adapter.configuration.stop_action, use_answer_synonyms=self.adapter.configuration.use_answer_synonyms, mode=EnvironmentMode.TEST, n_envs=self.n_test_envs, auto_skip=self.spaceadapter_config.auto_skip, dialog_faq_ratio=0.5, similarity_model=similarity_model, use_joint_dataset=True, **self.args['simulation'])
    
        #
        # network setup
        #
        if self.algorithm == 'dqn':
            self.model = self._dqn_model_from_args(self.args).to(self.device)
            # self.experiment.set_model_graph(str(self.model))
        self.adapter.set_model(self.model)


    def _parse_activation_fn(self, activation_fn_name: str):
        if activation_fn_name == "ReLU":
            return torch.nn.ReLU
        elif activation_fn_name  == "tanh":
            return torch.nn.Tanh
        elif activation_fn_name == "SELU":
            return torch.nn.SELU
        else:
            assert False, f"unknown activation function name: {activation_fn_name}"
   
    def _dqn_model_from_args(self, args: dict):
        q_value_clipping = args['dqn']['q_value_clipping'] if 'q_value_clipping' in args['dqn'] else 0

        kwargs = {
            "adapter": self.adapter,
            "dropout_rate": args['model']['dropout'],
            "activation_fn": self._parse_activation_fn(args['model']['activation_fn']),
            "normalization_layers": args['model']['normalization_layers'],
            "q_value_clipping": q_value_clipping,
        }
        if 'dueling' in args['model']['architecture']:
            kwargs |= {
                "shared_layer_sizes": args['model']['shared_layer_sizes'],
                "advantage_layer_sizes": args["model"]["advantage_layer_sizes"],
                "value_layer_sizes": args['model']['value_layer_sizes'],
            }
            if args['model']['intentprediction'] == False:
                from chatbot.adviser.app.rl.dqn.dqn import DuelingDQN
                model = DuelingDQN(**kwargs)
            else:
                if args['model']['architecture'] == "dueling":
                    from chatbot.adviser.app.rl.dqn.dqn import DuelingDQNWithIntentPredictionHead
                    model = DuelingDQNWithIntentPredictionHead(**kwargs)
                elif args['model']['architecture'] == "new_dueling":
                    from chatbot.adviser.app.rl.dqn.dqn import NewDuelingDQNWithIntentPredictionHead
                    model = NewDuelingDQNWithIntentPredictionHead(**kwargs)
        elif args['model']['architecture'] == 'vanilla':
            from chatbot.adviser.app.rl.dqn.dqn import DQN
            model = DQN(hidden_layer_sizes=args["model"]["hidden_layer_sizes"], **kwargs)
        assert model, f"unknown model architecture {args['model']['architecture']}"

        return model


    @torch.no_grad()
    def eval(self, env: ParallelDialogEnvironment, eval_dialogs: int, seed: int) -> float:
        """
        Returns:
            goal_asked score (float)
        """
        self.model.eval()

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

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
            "stop_prematurely_ratio": [],
            "faq_dialog_ratio": [],
            "episode_stop_ratio": [],
            "ask_variable_irrelevant_ratio": [],
            "ask_question_irrelevant_ratio": [],
            "episode_missing_variable_ratio": [],
            "episode_history_wordcount": [],
            "max_history_wordcount": [0],
            "intentprediction_consistency": []
        }
        if self.args['model']['intentprediction'] == True:
            intentprediction_tp = 0
            intentprediction_tn = 0
            intentprediction_fp = 0
            intentprediction_fn = 0
          
        batch_size = self.args['algorithm']["batch_size"]
        num_dialogs = 0
        obs = env.reset()
        intent_history = [[] for _ in range(batch_size)]
        while num_dialogs < eval_dialogs:
            # state = [self.adapter.state_vector(env_obs) for env_obs in obs]
            state = self.adapter.batch_state_vector_from_obs(obs, batch_size)
            node_keys = env.current_nodes_keys
            if self.algorithm == 'dqn':
                if self.adapter.configuration.action_config == ActionConfig.ACTIONS_IN_ACTION_SPACE:
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
                    eval_metrics["stop_prematurely_ratio"].append(env_instance.actioncount_stop_prematurely)
                    eval_metrics["faq_dialog_ratio"].append(1.0 if env_instance.is_faq_mode else 0.0)
                    eval_metrics["episode_stop_ratio"].append(env_instance.actioncount_stop)
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
                            
                    if EXPERIMENT_LOGGING != ExperimentLogging.NONE:
                        env_instance.logger.info("\n".join(env_instance.episode_log))
                    
                    intent_history[done_idx] = [] # reset intent history
                    obs[done_idx] = env_instance.reset()

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

def load_ckpt(evaluator: Evaluator):
    # load checkpoint
    print(f"Loading checkpoint from {CHECKPOINT_LOCATION}...")
    ckpt = torch.load(CHECKPOINT_LOCATION, map_location=torch.device('cpu'))
    evaluator.model.load_state_dict(ckpt['model'])
    evaluator.model.to(evaluator.device)

    global_step = ckpt['global_step']

    torch.set_rng_state(ckpt['torch_rng'])
    np.random.set_state(ckpt['numpy_rng'])
    random.setstate(ckpt['rand_rng'])

    print("Checkpoint step", global_step)


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "true"

    Data.objects[0] = Data.Dataset.fromJSON('traintest_graph.json', version=0)
    Data.objects[1] = Data.Dataset.fromJSON('traintest_graph.json', version=1)

    evaluator = Evaluator()
    evaluator.setUp()
    load_ckpt(evaluator)
   
    for run, seed in enumerate(RUN_SEEDS):
        print(f"---- RUN {run} with seed {seed} ----")
        evaluator.eval(evaluator.eval_env, DIALOGS, seed)
    