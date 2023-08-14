LANGAUGE = "en" # "de"
import chatbot.adviser.app.rl.dataset as Data
Data.LANGUAGE = LANGAUGE

from copy import deepcopy
import logging
from statistics import mean
from typing import Any, Dict, List
from functools import reduce
from multiprocessing import Process
import json
import os
from enum import Enum
import time
import random

import torch
import numpy as np
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_sequence
import wandb

from chatbot.adviser.app.answerTemplateParser import AnswerTemplateParser
from chatbot.adviser.app.encoding.similiarity import AnswerSimilarityEncoding
from chatbot.adviser.app.encoding.text import TextEmbeddingPooling

from chatbot.adviser.app.rl.dialogenv import EnvironmentMode, ParallelDialogEnvironment
from chatbot.adviser.app.rl.dialogtree import DialogTree

from chatbot.adviser.app.rl.layers.attention.attention_factory import AttentionActivationConfig, AttentionMechanismConfig, AttentionVectorAggregation
from chatbot.adviser.app.rl.spaceAdapter import AnswerSimilarityEmbeddingConfig, IntentEmbeddingConfig, SpaceAdapter, ActionConfig, SpaceAdapterAttentionInput, SpaceAdapterAttentionQueryInput, SpaceAdapterConfiguration, SpaceAdapterSpaceInput, TextEmbeddingConfig
from chatbot.adviser.app.rl.utils import EMBEDDINGS, AutoSkipMode, AverageMetric, EnvInfo, ExperimentLogging, _del_checkpoint, _get_file_hash, _munchausen_stable_logsoftmax, _munchausen_stable_softmax, _save_checkpoint, safe_division



torch.set_num_threads(8) # default on server: 32
torch.set_num_interop_threads(8) # default on server: 32


EXPERIMENT_LOGGING = ExperimentLogging.ONLINE


class AugmentationMode(Enum):
    NO_AUGMENTATION="NONE"
    ONLY_AUGMENTATION="ONLY"
    COMBINED="COMBINED"

class Trainer:
    def setUp(self) -> None:
        self.device = "cuda:0" if len(os.environ["CUDA_VISIBLE_DEVICES"].strip()) > 0 else "cpu"

        # REMOVE AUTOSKIP ARG FROM SIMULATION
        # ADD stop_action ARG TO CONFIGURATION
        # ADD noise ARG TO STATE TEXT INPUTSAi
        seed = 9546370
        self.exp_name_prefix = "V3_GENERATEDONLY"
   
        self.args = {
            "language": LANGAUGE,
            "data": {
               "augmentation": AugmentationMode.ONLY_AUGMENTATION,
               "augmentation_version": 2
            },
            "spaceadapter": {
                "configuration": SpaceAdapterConfiguration(
                    text_embedding="all-mpnet-base-v2", #'distiluse-base-multilingual-cased-v2', # 'gbert-large' # 'cross-en-de-roberta-sentence-transformer', 'all-mpnet-base-v2'
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
                        model_name='all-mpnet-base-v2',
                    ),
                    dialog_node_text=TextEmbeddingConfig(
                        active=True,
                        pooling=TextEmbeddingPooling.MEAN,
                    ),
                    original_user_utterance=TextEmbeddingConfig(
                        active=True,
                        pooling=TextEmbeddingPooling.MEAN,
                    ),
                    current_user_utterance=TextEmbeddingConfig(
                        active=True,
                        pooling=TextEmbeddingPooling.MEAN,
                    ),
                    dialog_history=TextEmbeddingConfig(
                        active=True,
                        pooling=TextEmbeddingPooling.MEAN,
                    ),
                    action_text=TextEmbeddingConfig(
                        active=True,
                        pooling=TextEmbeddingPooling.MEAN,
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
                            allow_noise=True
                        ),
                        matrix="dialog_node_text",
                        activation=AttentionActivationConfig.NONE,
                        attention_mechanism=AttentionMechanismConfig.ADDITIVE,
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
                            allow_noise=True
                        ),
                        matrix="dialog_history",
                        activation=AttentionActivationConfig.NONE,
                        attention_mechanism=AttentionMechanismConfig.ADDITIVE,
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
                "parallel_train_envs": 256,
                "parallel_test_envs": 256,
                "train_noise": 0.1,
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
                "shared_layer_sizes": [4096, 4096, 4096],
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
                "lr": 0.0001,
                "intent_loss_weighting": 0.1
            },
            "algorithm": {
                "timesteps_per_reset": 1000000,
                "reset_exploration_times": 0,
                "max_grad_norm": 1.0,
                "batch_size": 256,
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
                "eps_start": 0.9,
                "eps_end": 0.0,
                "train_frequency": 3,
                "learning_starts": 2560,
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
        print("Setting random seeds...")
        random.seed(self.args["experiment"]["seed"])
        np.random.seed(self.args["experiment"]["seed"])
        torch.manual_seed(self.args["experiment"]["seed"])
        torch.backends.cudnn.deterministic = self.args["experiment"]["cudnn_deterministic"]

        # Load data
        Data.LANGUAGE = self.args['language']
        Data.objects[0] = Data.Dataset.fromJSON(f"resources/{self.args['language']}/train_graph.json", answer_synonyms=self.args["spaceadapter"]['configuration'].use_answer_synonyms, version=0)
        Data.objects[1] = Data.Dataset.fromJSON(f"resources/{self.args['language']}/test_graph.json", answer_synonyms=self.args["spaceadapter"]['configuration'].use_answer_synonyms, version=1)

        # load augmentation data
        augmentation_mode = self.args['data']['augmentation']
        if augmentation_mode == AugmentationMode.NO_AUGMENTATION:
            print("DATA AUGMENTATION: NONE")
        elif augmentation_mode.ONLY_AUGMENTATION:
            print("ONLY AUGMENTATION")
            # remove all existing questions
            for node in Data.objects[0].nodes_by_type(Data.NodeType.INFO.value):
                # remove existing questions
                for question in node.faq_questions:
                    Data.objects[0]._faq_list.remove(question)
                    del Data.objects[0]._faq_by_key[question.key]
                node.faq_questions = []
            # remove all existing answer synonyms
            Data.objects[0].answer_synonyms = {}
        else:
            print("MIXED AUGMENTATION")
        if augmentation_mode in [AugmentationMode.ONLY_AUGMENTATION, AugmentationMode.COMBINED]:
            # add generated questions 
            with open(f'resources/{LANGAUGE}/augmentation/train_questions_v{self.args["data"]["augmentation_version"]}.json', "r") as f:
                data = json.load(f)
                for key in data:
                    # locate node object
                    node_entry = data[key]
                    node_key = node_entry["dialog_node_key"]
                    node_obj: Data.DialogNode = Data.objects[0].node_by_key(node_key)
                    # add new question (only if info node)
                    if node_obj.node_type == Data.NodeType.INFO.value:
                        new_question = Data.FAQQuestion(key=key, text=node_entry['text'], dialog_node_key=node_key, version=0)
                        node_obj.faq_questions.append(new_question)
                        Data.objects[0]._faq_list.append(new_question)
                        Data.objects[0]._faq_by_key[new_question.key] = new_question
                # update number of questions
                Data.objects[0]._num_faq_nodes = sum([1 for node in Data.objects[0]._node_list if len(node.faq_questions) > 0])
            # answer canadidates
            if self.args["spaceadapter"]['configuration'].use_answer_synonyms:
                with open(f'resources/{LANGAUGE}/augmentation/train_answers.json', 'r') as f:
                    new_answer_synonyms = json.load(f)
                    for key in new_answer_synonyms:
                        if augmentation_mode == AugmentationMode.ONLY_AUGMENTATION:
                            Data.objects[0].answer_synonyms[key] = []
                        Data.objects[0].answer_synonyms[key].extend(new_answer_synonyms[key])
                    # update number of answers
                    Data.objects[0]._num_answer_synonyms = sum([len(Data.objects[0].answer_synonyms[answer]) for answer in Data.objects[0].answer_synonyms])

        # load dialog tree
        print("Loading data...")
        self.tree = DialogTree(version=0)
        self.eval_tree = DialogTree(version=1)

        # load text embedding
        print("Loading embeddings...")
        text_embedding_name = self.args['spaceadapter']['configuration'].text_embedding
        self.text_enc = EMBEDDINGS[text_embedding_name]['class'](device=self.device, **EMBEDDINGS[text_embedding_name]['args'])

        # post-init spaceadapter 
        print("Configuration of Space Adapter....")
        self.spaceadapter_config: SpaceAdapterConfiguration = self.args['spaceadapter']['configuration']
        self.spaceadapter_state: SpaceAdapterSpaceInput = self.args['spaceadapter']['state']
        self.spaceadapter_attention: List[SpaceAdapterAttentionInput] = self.args['spaceadapter']['attention']
        self.spaceadapter_config.post_init(tree=self.tree)
        self.spaceadapter_state.post_init(device=self.device, tree=self.tree, text_embedding=self.text_enc, action_config=self.spaceadapter_config.action_config, action_masking=self.spaceadapter_config.action_masking, stop_action=self.spaceadapter_config.stop_action)
        for attn in self.spaceadapter_attention:
            attn.post_init(device=self.device, tree=self.tree, text_embedding=self.text_enc, action_config=self.spaceadapter_config.action_config, action_masking=self.spaceadapter_config.action_masking)

        # prepare directories
        spaceadapter_json = self.spaceadapter_config.toJson() | self.spaceadapter_state.toJson() | {"attention": [attn.toJson() for attn in self.spaceadapter_attention]}
        if EXPERIMENT_LOGGING != ExperimentLogging.NONE:
            self.exp_name = f"{self.exp_name_prefix}_{self.args['algorithm']['algorithm']}_{str(int(100 * self.args['simulation']['dialog_faq_ratio']))}dialog_{self.spaceadapter_config.action_config.value}_{self.spaceadapter_config.text_embedding}"
            for key in spaceadapter_json['state']:
                if isinstance(spaceadapter_json['state'][key], bool):
                    if not spaceadapter_json['state'][key]:
                        self.exp_name += f"_no{key}"
                else:
                    if not spaceadapter_json['state'][key]['active']:
                        self.exp_name += f"_no{key}"
            self.run_name = f"{self.exp_name}__{seed}__{int(time.time())}"
            os.makedirs(f"/mount/arbeitsdaten/asr-2/vaethdk/adviser_reisekosten/newruns_en/{self.run_name}")
            os.makedirs(f"/fs/scratch/users/vaethdk/cts_english/newruns_en/{self.run_name}")
            log_to_file_eval = f"/fs/scratch/users/vaethdk/cts_english/newruns_en/{self.run_name}/eval_dialogs.txt"
            print("Logging EVAL dialogs to file", log_to_file_eval)
            log_to_file_test = f"/fs/scratch/users/vaethdk/cts_english/newruns_en/{self.run_name}/test_dialogs.txt"
            print("Logging TEST dialogs to file", log_to_file_test)
            
            self.eval_logger = logging.getLogger("env" + EnvironmentMode.EVAL.name)
            self.eval_logger.setLevel(logging.INFO)
            self.eval_file_handler = logging.FileHandler(log_to_file_eval, mode='w')
            self.eval_file_handler.setLevel(logging.INFO)
            self.eval_logger.addHandler(self.eval_file_handler)
            self.eval_logger.info("EVAL LOGGER")

            self.test_logger = logging.getLogger("env" + EnvironmentMode.TEST.name)
            self.test_logger.setLevel(logging.INFO)
            test_file_handler = logging.FileHandler(log_to_file_test, mode='w')
            test_file_handler.setLevel(logging.INFO)
            self.test_logger.addHandler(test_file_handler)
            self.test_logger.info("TEST LOGGER")
        else:
            self.eval_logger = None
            self.test_logger = None

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
                similarity_model = AnswerSimilarityEncoding(model_name="distiluse-base-multilingual-cased-v2", dialog_tree=self.tree, device=self.device)
       
        dialog_faq_ratio = self.args['simulation'].pop('dialog_faq_ratio')

        print("Starting environments...")
        self.train_env = ParallelDialogEnvironment(dialog_tree=self.tree, adapter=self.adapter, stop_action=self.adapter.configuration.stop_action, mode=EnvironmentMode.TRAIN, n_envs=self.n_train_envs, auto_skip=self.spaceadapter_config.auto_skip, dialog_faq_ratio=dialog_faq_ratio, similarity_model=similarity_model, log_to_file=None, **self.args['simulation'])
        self.eval_env = ParallelDialogEnvironment(dialog_tree=self.tree, adapter=self.adapter, stop_action=self.adapter.configuration.stop_action, mode=EnvironmentMode.EVAL, n_envs=self.n_test_envs, auto_skip=self.spaceadapter_config.auto_skip, dialog_faq_ratio=0.5, similarity_model=similarity_model, log_to_file=self.eval_logger, **self.args['simulation'])
        self.test_env = ParallelDialogEnvironment(dialog_tree=self.eval_tree, adapter=self.adapter, stop_action=self.adapter.configuration.stop_action, mode=EnvironmentMode.TEST, n_envs=self.n_test_envs, auto_skip=self.spaceadapter_config.auto_skip, dialog_faq_ratio=0.5, similarity_model=similarity_model, log_to_file=self.test_logger, **self.args['simulation'])
        
        print("Setup logging...")

        if EXPERIMENT_LOGGING == ExperimentLogging.OFFLINE:
            # TODO set wandb api key in env variable: "WANDB_API_KEY"
            os.environ["WANDB_MODE"] = "offline"
        
        args = {key: self.args[key] for key in self.args if key != 'spaceadapter'}
        args['data'] = {"augmentation": self.args['data']['augmentation'].value}
        if EXPERIMENT_LOGGING != ExperimentLogging.NONE:
            # write code 
            wandb.init(project="cts_en", config=(spaceadapter_json | args), save_code=True, name=self.exp_name, settings=wandb.Settings(code_dir="/fs/scratch/users/vaethdk/cts_english/chatbot/management/commands"))
            wandb.config.update({'datasetversion': _get_file_hash(f'resources/{Data.LANGUAGE}/train_graph.json')}) # log dataset version hash

        #
        # network setup
        #
        print("Loading model...")
        if self.algorithm == 'dqn':
            self.model = torch.compile(self._dqn_model_from_args(self.args).to(self.device))
            self.target_network = torch.compile(self._dqn_model_from_args(self.args).to(self.device))
            self.target_network.load_state_dict(self.model.state_dict())
            self.target_network.eval()
            # self.experiment.set_model_graph(str(self.model))
        self.optimizer = self._optimizer_from_args(self.args, self.model)
        self.adapter.set_model(self.model)

        if EXPERIMENT_LOGGING != ExperimentLogging.NONE:
            wandb.watch(self.model, log_freq=100)

        #
        # buffer setup
        #
        print("Setting up replay buffer...")
        if self.algorithm == "dqn":
            if not "buffer_type" in self.args['dqn'] or self.args['dqn']['buffer_type'] == 'uniform':
                from chatbot.adviser.app.rl.dqn.replay_uniform import UniformReplayBuffer
                self.rb = UniformReplayBuffer(self.args['dqn']['buffer_size'], self.adapter, device=self.device)
            elif self.args['dqn']['buffer_type'] == 'prioritized':
                from chatbot.adviser.app.rl.dqn.replay_prioritized import PrioritizedReplayBuffer
                self.rb = PrioritizedReplayBuffer(
                    buffer_size=self.args['dqn']['buffer_size'], adapter=self.adapter, device=self.device,
                        alpha=self.args['dqn']['priority_replay_alpha'], beta=self.args['dqn']['priority_replay_beta']
                )
            elif self.args['dqn']['buffer_type'] == 'LAP':
                from chatbot.adviser.app.rl.dqn.replay_prioritized import PrioritizedLAPReplayBuffer
                self.rb = PrioritizedLAPReplayBuffer( buffer_size=self.args['dqn']['buffer_size'], adapter=self.adapter, device=self.device,
                        alpha=self.args['dqn']['priority_replay_alpha'], beta=self.args['dqn']['priority_replay_beta']
                )
            elif self.args['dqn']['buffer_type'] == 'HER':
                from chatbot.adviser.app.rl.dqn.replay_her import HindsightExperienceReplay
                self.rb = HindsightExperienceReplay(envs=self.train_env, buffer_size=self.args['dqn']['buffer_size'], adapter=self.adapter,
                                                    train_noise=self.args['simulation']['train_noise'],
                                                    dialog_tree=self.tree, answerParser=AnswerTemplateParser(), logicParser=self.train_env.logicParser,
                                                    dialog_faq_ratio=0.0, max_reward=self.train_env.max_reward,
                                                    alpha=self.args['dqn']['priority_replay_alpha'], beta=self.args['dqn']['priority_replay_beta'],
                                                    device=self.device, experiment_logging=EXPERIMENT_LOGGING, auto_skip=self.spaceadapter_config.auto_skip,
                                                    stop_when_reaching_goal=self.args['simulation']['stop_when_reaching_goal'],
                                                    similarity_model=similarity_model)
        # write experiment config file
        if EXPERIMENT_LOGGING != ExperimentLogging.NONE:
            with open(f"/mount/arbeitsdaten/asr-2/vaethdk/adviser_reisekosten/newruns_en/{self.run_name}/config.json", "w") as f:
                json.dump({'spaceadapter': spaceadapter_json} | args, f)


        # Setup train metrics
        print("Setting up metrics...")
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

    def _flattened_args_dict(self, args: dict, outer_key: str = ""):
        flattened = {}
        for key in args:
            new_outer_key = f"{outer_key}_{key}" if len(outer_key) > 0 else key
            if isinstance(args[key], dict):
                flattened = flattened | self._flattened_args_dict(args[key], new_outer_key)
            else:
                flattened[new_outer_key] = args[key]
        return flattened

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
        keep_checkpoints = self.args['experiment']['keep'] if "keep" in self.args['experiment'] else 5
        if EXPERIMENT_LOGGING != ExperimentLogging.NONE:
            self.last_save_step = global_step

            # check if we should save at all (or if the current checkpoint is worse than all others that we have already)
            worst_score_so_far = min(self.savefile_goal_asked_score.values()) if len(self.savefile_goal_asked_score) > 0 else -1.0
            if worst_score_so_far > goal_asked_score:
                # new checkpoint is worse than all the ones we have so far - don't save it
                return

            # find worst checkpoint
            worst_score_file = None
            if len(self.savefile_goal_asked_score) >= keep_checkpoints:
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
            if len(self.savefile_goal_asked_score) < keep_checkpoints:
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
                            self.savefile_goal_asked_score[f"/mount/arbeitsdaten/asr-2/vaethdk/adviser_reisekosten/newruns_en/{self.run_name}/ckpt_{global_step}.pt"] = goal_asked_score
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
                    self.savefile_goal_asked_score[f"/mount/arbeitsdaten/asr-2/vaethdk/adviser_reisekosten/newruns_en/{self.run_name}/ckpt_{global_step}.pt"] = goal_asked_score
    

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

    def _optimizer_from_args(self, args: dict, model: torch.nn.Module):
        if args['optimizer']['name'] in ["Adam", "AMSGrad"]:
            optim = torch.optim.Adam(model.parameters(), lr=args['optimizer']['lr'], amsgrad=args['optimizer']['name'] == "AMSGrad")
        elif args['optimizer']['name'] == "AdamW":
            optim = torch.optim.AdamW(model.parameters(), lr=args['optimizer']['lr'])
        assert optim, "unknown optimizer"

        return optim

    @torch.no_grad()
    def eval(self, env: ParallelDialogEnvironment, eval_dialogs: int, eval_phase: int, prefix: str) -> float:
        """
        Returns:
            goal_asked score (float)
        """
        self.model.eval()
        
        

        if EXPERIMENT_LOGGING != ExperimentLogging.NONE:
            f = open(f"/fs/scratch/users/vaethdk/cts_english/newruns_en/{self.run_name}/{prefix}_dialogs_{eval_phase}.txt", "w")
            f.write(f"=========== EVAL AT STEP {eval_dialogs}, PHASE {eval_phase} ============\n")
        
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
                            
                    if EXPERIMENT_LOGGING != ExperimentLogging.NONE and not isinstance(env_instance.logger, type(None)):
                        f.writelines("\n".join(env_instance.episode_log))
                    
                    intent_history[done_idx] = [] # reset intent history
                    obs[done_idx] = env_instance.reset()

        # log metrics (averaged)
        log_dict = {
            f"{prefix}/coverage_faqs": env.get_coverage_faqs(),
            f"{prefix}/coverage_synonyms": env.get_coverage_synonyms(),
            f"{prefix}/coverage_variables": env.get_coverage_variables(),
            f"{prefix}/count_seen_countries": env.count_seen_countries(),
            f"{prefix}/count_seen_cities": env.count_seen_cities(),
            f"{prefix}/coverage_goal_nodes_free": env.get_goal_node_coverage_free(),
            f"{prefix}/coverage_goal_nodes_guided": env.get_goal_node_coverage_guided(),
            f"{prefix}/coverage_nodes": env.get_node_coverage(),
        }
        if self.args['model']['intentprediction'] == True:
            eval_metrics["intentprediction_f1"] = [safe_division(intentprediction_tp, intentprediction_tp + 0.5 * (intentprediction_fp + intentprediction_fn))]
            eval_metrics["intentprediction_recall"] = [safe_division(intentprediction_tp, intentprediction_tp + intentprediction_fn)]
            eval_metrics["intentprediction_precision"] = [safe_division(intentprediction_tp, intentprediction_tp + intentprediction_fp)]
            eval_metrics["intentprediction_accuracy"] = [safe_division(intentprediction_tp + intentprediction_tn, num_dialogs)]
        for metric in eval_metrics:
            numerical_entries = [num for num in eval_metrics[metric] if num is not None]
            if len(numerical_entries) == 0:
                numerical_entries = [0.0]
            log_dict[f"{prefix}/{metric}"] = mean(numerical_entries)
        if EXPERIMENT_LOGGING != ExperimentLogging.NONE:
            wandb.log(log_dict, step=eval_phase)
            f.close()

        self.model.train()
        return mean(eval_metrics["goal_asked"])

    def log_train_step(self, global_step: int, train_step: int, episode_counter: int, turn_counter: int,epsilon: float, timesteps_per_reset: int, beta: float):
        if train_step % 50 == 0 and EXPERIMENT_LOGGING != ExperimentLogging.NONE:
            log_dict = {
                "train/learning_phase": global_step // timesteps_per_reset,
                "train/global_step": global_step,
                "train/train_counter": train_step,
                "train/episode_counter": episode_counter,
                "train/turn_counter": turn_counter,
                "train/coverage_faqs": self.train_env.get_coverage_faqs(),
                "train/coverage_synonyms": self.train_env.get_coverage_synonyms(),
                "train/coverage_variables": self.train_env.get_coverage_variables(),
                "train/count_seen_countries": self.train_env.count_seen_countries(),
                "train/count_seen_cities": self.train_env.count_seen_cities(),
                "train/coverage_goal_nodes_free": self.train_env.get_goal_node_coverage_free(),
                "train/coverage_goal_nodes_guided": self.train_env.get_goal_node_coverage_guided(),
                "train/coverage_nodes": self.train_env.get_node_coverage(),
            }
            if self.algorithm == "dqn":
                log_dict["train/epsilon"] = epsilon
                if 'buffer_type' in self.args['dqn'] and self.args['dqn']['buffer_type'].lower() in ['prioritized', 'her']:
                    log_dict["train/priority_beta"] = beta
                log_dict["train/buffer_size"] = len(self.rb)
            if self.train_env.current_episode > 0:
                log_dict["train/faq_dialog_ratio"] = self.train_env.num_faqbased_dialogs / self.train_env.current_episode
            log_dict["train/actioncount_stop_prematurely"] = self.train_env.actioncount_stop_prematurely
            wandb.log(log_dict, step=global_step, commit=(global_step % 250) == 0)

    def store_dqn(self, observations: List[torch.FloatTensor], next_observations: List[torch.FloatTensor], actions: List[int], rewards: List[float], dones: List[bool], infos: List[dict], global_step: int):
        for env_id, (obs, next_obs, action, reward, done, info) in enumerate(zip(observations, next_observations, actions, rewards, dones, infos)):
            self.rb.add(env_id, obs, next_obs, action, reward, done, info, global_step)

    @torch.no_grad()
    def _munchausen_target(self, next_observations, data, q_prev: torch.FloatTensor):
        tau = self.args['dqn']['munchausen_tau']
        q_next = self.target_network(next_observations)[0] # batch x actions
        mask = q_next > float('-inf')
        sum_term = F.softmax(q_next / tau, dim=-1) * (q_next - _munchausen_stable_logsoftmax(q_next, tau)) # batch x actions
        log_policy = _munchausen_stable_logsoftmax(q_prev, tau).gather(-1, data.actions).view(-1) # batch x actions -> batch
        if self.args['dqn']['munchausen_clipping'] != 0:
            log_policy = torch.clip(log_policy, min=self.args['dqn']['munchausen_clipping'], max=1)
        return data.rewards.flatten() + self.args['dqn']['munchausen_alpha']*log_policy + self.args['algorithm']["gamma"] * sum_term.masked_fill(~mask, 0.0).sum(-1) * (1.0 - data.dones.flatten())

    @torch.no_grad()
    def _td_target(self, next_observations, data):
        target_pred, _ = self.target_network(next_observations)
        target_max, _ = target_pred.max(dim=1) # output[1] would be predicted intent classes
        return data.rewards.flatten() + self.args['algorithm']["gamma"] * target_max * (1 - data.dones.flatten()*torch.tensor(data.infos[EnvInfo.IS_FAQ], dtype=torch.float, device=self.device))


    def train_step_dqn(self, global_step: int, train_counter: int):
        data = self.rb.sample(self.args['algorithm']["batch_size"])

        # observations = [self.adapter.state_vector({ key: data.observations[key][index] for key in data.observations}) for index in range(self.args['algorithm']["batch_size"])]
        # next_observations = [self.adapter.state_vector({ key: data.next_observations[key][index] for key in data.next_observations}) for index in range(self.args['algorithm']["batch_size"])]
        observations = self.adapter.batch_state_vector(data.observations, self.args['algorithm']["batch_size"])
        next_observations = self.adapter.batch_state_vector(data.next_observations, self.args['algorithm']["batch_size"])
        
        if self.adapter.configuration.action_config == ActionConfig.ACTIONS_IN_ACTION_SPACE:
            observations = torch.cat(observations, dim=0)
            next_observations = torch.cat(next_observations, dim=0)
        else:
            observations = pack_sequence(observations, enforce_sorted=False)
            next_observations = pack_sequence(next_observations, enforce_sorted=False)

        old_val, intent_logits = self.model(observations)
        if 'munchausen_targets' in self.args['dqn'] and self.args['dqn']['munchausen_targets'] == True:
            td_target = self._munchausen_target(next_observations, data, old_val)
        else:
            td_target = self._td_target(next_observations, data)
        old_val = old_val.gather(1, data.actions).squeeze()

        # loss
        loss = F.huber_loss(old_val, td_target, reduction="none")
        intent_loss = 0 if not torch.is_tensor(intent_logits) else F.binary_cross_entropy_with_logits(intent_logits.view(-1), torch.tensor(data.infos[EnvInfo.IS_FAQ], dtype=torch.float, device=self.device), reduction="none")
        if 'buffer_type' in self.args['dqn'] and self.args['dqn']['buffer_type'].lower() in ['prioritized', 'her']:
            loss = loss * data.weights
            if not isinstance(intent_logits, type(None)):
                intent_loss = intent_loss * data.weights
            # update priorities
            td_error = torch.abs(td_target - old_val)
            self.rb.update_weights(data.indices, td_error)
            # scale gradients by priority weights
        loss = loss.mean(-1) # reduce loss
        if not isinstance(intent_logits, type(None)):
            intent_loss = intent_loss.mean(-1) # reduce loss
        
        if EXPERIMENT_LOGGING != ExperimentLogging.NONE:
            log_dict = {"train/loss": loss.item(),
                        "train/q_values": old_val.mean().item()}
            if not isinstance(intent_logits, type(None)):
                log_dict['train/intent_loss'] = intent_loss.item()
            if 'buffer_type' in self.args['dqn'] and self.args['dqn']['buffer_type'].lower() in ['prioritized', 'her']:
                log_dict['train/priorization_weights'] = data.weights.mean().item()
            wandb.log(log_dict, step=global_step, commit=(train_counter % 250 == 0))

        # optimize the model
        loss += intent_loss * self.args['optimizer']['intent_loss_weighting']
        self.optimizer.zero_grad()
        loss.backward()

        if self.args['algorithm']["max_grad_norm"] > 0:
            torch.nn.utils.clip_grad_value_(self.model.parameters(), self.args['algorithm']["max_grad_norm"])
        self.optimizer.step()

        # update the target network
        if train_counter % self.args['dqn']["target_network_frequency"] == 0:
            self.target_network.load_state_dict(self.model.state_dict())


    def train_loop(self):
        evaluation = self.args['evaluation']["evaluation"]
        eval_every_train_timesteps = self.args["evaluation"]["every_train_timesteps"]
        eval_dialogs = self.args['evaluation']['dialogs']

        #
        # agent environment loop
        #
        timesteps_per_reset = self.args['algorithm']['timesteps_per_reset']
        learning_phases = self.args['algorithm']['reset_exploration_times'] + 1 # 0 resets = 1 run
        total_timesteps = timesteps_per_reset * learning_phases

        self.model.train()
        obs: List[Dict[str, Any]] = self.train_env.reset()

        global_step = 0
        train_counter = 0
        episode_counter = 0
        turn_counter = 0

        # initial evaluation
        # self.eval(self.eval_env, eval_dialogs, global_step, prefix="eval")
        
        while global_step < total_timesteps:
            epsilon = self._linear_schedule(self.args['dqn']['eps_start'], self.args['dqn']['eps_end'], self.args['dqn']['exploration_fraction'] * timesteps_per_reset, global_step % timesteps_per_reset)
            beta = self._beta_schedule(self.args['dqn']['priority_replay_beta'], self.args['dqn']['exploration_fraction'] * timesteps_per_reset, global_step % timesteps_per_reset)

            # state = [self.adapter.state_vector(env_obs) for env_obs in obs]
            state = self.adapter.batch_state_vector_from_obs(obs, self.args['algorithm']["batch_size"])
            # choose and perform next action
            # state = [[env_obs[key] for key in env_obs if torch.is_tensor(env_obs[key])] for env_obs in obs]
            if self.adapter.configuration.action_config == ActionConfig.ACTIONS_IN_ACTION_SPACE:
                actions, _ = self.model.select_actions_eps_greedy(self.train_env.current_nodes_keys, torch.cat(state, dim=0), epsilon)
            else:
                actions, _ = self.model.select_actions_eps_greedy(self.train_env.current_nodes_keys, pack_sequence(state, enforce_sorted=False), epsilon)
                
            next_obs, rewards, dones, infos = self.train_env.step(actions)
            turn_counter += len(actions)

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

                    if EXPERIMENT_LOGGING != ExperimentLogging.NONE and episode_counter % self.train_episodic_return.running_avg == 0:
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
             
                if self.algorithm == 'dqn' and len(self.rb) >= self.args['dqn']['learning_starts'] and global_step % self.args['dqn']["train_frequency"] == 0:
                    if self.args['dqn']['buffer_type'].lower() in ['prioritized', 'her']:
                        self.rb.update_beta(beta)
                    self.train_step_dqn(global_step, train_counter)
                    train_counter += 1
                self.log_train_step(global_step=global_step, train_step=train_counter, episode_counter=episode_counter, turn_counter=turn_counter, epsilon=epsilon, timesteps_per_reset=timesteps_per_reset, beta=beta)

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
            return torch.cat(tensors, dim=0).to(self.device)
        else:
            return pack_sequence([tensor.to(self.device) for tensor in tensors], enforce_sorted=False)

    def _flatten_list(self, multidim_list):
        return reduce(lambda sublist1, sublist2: sublist1 + sublist2, multidim_list)





if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "true"

    trainer = Trainer()
    trainer.setUp()
    if trainer.algorithm == "dqn":
        trainer.train_loop()
    