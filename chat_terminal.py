import os

from chatbot.adviser.app.rl.dqn.dqn import DQNNetwork
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

MODEL_PATH = "/mount/arbeitsdaten/asr-2/vaethdk/adviser_reisekosten/newruns/V9_ALTERNATIVESEED_ROBERTA_NEWARCH_dqn_50dialog_1_cross-en-de-roberta-sentence-transformer_nouser_intent_prediction_dqn_50dialog_1_cross-en-de-roberta-sentence-transformer_nouser_intent_prediction__9546370__1672918719"
CKPT = "760000"


import torch
from torch.nn.utils.rnn import pack_sequence
device = "cuda:0"

from typing import List
from copy import deepcopy
import itertools
import traceback

from chatbot.adviser.app.rl.dialogtree import DialogTree
import chatbot.adviser.app.rl.dataset as Data
from chatbot.adviser.app.rl.spaceAdapter import AnswerSimilarityEmbeddingConfig, IntentEmbeddingConfig, SpaceAdapter, ActionConfig, SpaceAdapterAttentionInput, SpaceAdapterAttentionQueryInput, SpaceAdapterConfiguration, SpaceAdapterSpaceInput, TextEmbeddingConfig
from chatbot.adviser.app.rl.utils import EMBEDDINGS, AutoSkipMode, StateEntry, AverageMetric, EnvInfo, ExperimentLogging, _del_checkpoint, _get_file_hash, _munchausen_stable_logsoftmax, _munchausen_stable_softmax, _save_checkpoint, safe_division
from chatbot.adviser.app.rl.layers.attention.attention_factory import AttentionActivationConfig, AttentionMechanismConfig, AttentionVectorAggregation
from chatbot.adviser.app.encoding.text import TextEmbeddingPooling, TextEmbeddings



def load_config() -> dict:
    seed = 12345678
    return {
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
                    caching=False,
                ),
                dialog_node_text=TextEmbeddingConfig(
                    active=True,
                    pooling=TextEmbeddingPooling.MEAN,
                    caching=False,
                ),
                original_user_utterance=TextEmbeddingConfig(
                    active=True,
                    pooling=TextEmbeddingPooling.MEAN,
                    caching=False,
                ),
                current_user_utterance=TextEmbeddingConfig(
                    active=True,
                    pooling=TextEmbeddingPooling.MEAN,
                    caching=False,
                ),
                dialog_history=TextEmbeddingConfig(
                    active=True,
                    pooling=TextEmbeddingPooling.MEAN,
                    caching=False,
                ),
                action_text=TextEmbeddingConfig(
                    active=True,
                    pooling=TextEmbeddingPooling.MEAN,
                    caching=False,
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
                        caching=False,
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
                        caching=False,
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
            "train_noise": 0.1,
            "eval_noise": 0.0,
            "test_noise": 0.0
        },
        "experiment": {
            "seed": seed,
            "cudnn_deterministic": True,
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
            "timesteps_per_reset": 1000000,
            "reset_exploration_times": 0,
            "max_grad_norm": 1.0,
            "batch_size": 3,
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


def load_data() -> DialogTree:
    print("Loading data...")
    Data.objects[0] = Data.Dataset.fromJSON('train_graph.json', version=0)
    # Data.objects[1] = Data.Dataset.fromJSON('test_graph.json', version=1)

    tree = DialogTree(version=0)
    print(" - Tree depth", tree.get_max_tree_depth())
    print(" - Tree max. degree", tree.get_max_node_degree())

    return tree

def get_text_embedding(args: dict) -> TextEmbeddings:
    print("Loading Text Embedding...")
    text_embedding_name = args['spaceadapter']['configuration'].text_embedding
    EMBEDDINGS[text_embedding_name]['args'].pop('cache_db_index')
    return EMBEDDINGS[text_embedding_name]['class'](device=device, **EMBEDDINGS[text_embedding_name]['args'])


def setup_space_adapter(text_enc: TextEmbeddings) -> SpaceAdapter:
    print("Configuring Space Adapter...")
    cache_conn = None # no caching for testing
    spaceadapter_config: SpaceAdapterConfiguration = args['spaceadapter']['configuration']
    spaceadapter_state: SpaceAdapterSpaceInput = args['spaceadapter']['state']
    spaceadapter_attention: List[SpaceAdapterAttentionInput] = args['spaceadapter']['attention']
    spaceadapter_config.post_init(tree=tree)
    spaceadapter_state.post_init(device=device, tree=tree, text_embedding=text_enc, action_config=spaceadapter_config.action_config, action_masking=spaceadapter_config.action_masking, stop_action=spaceadapter_config.stop_action, cache_connection=cache_conn)
    for attn in spaceadapter_attention:
        attn.post_init(device=device, tree=tree, text_embedding=text_enc, action_config=spaceadapter_config.action_config, action_masking=spaceadapter_config.action_masking, cache_connection=cache_conn)
    return SpaceAdapter(device=device, dialog_tree=tree, **args["spaceadapter"])

def init_model(args: dict) -> DQNNetwork:
    print("Initializing model...")
    if args['model']['activation_fn'] == "ReLU":
        acivation_fn = torch.nn.ReLU
    elif args['model']['activation_fn']  == "tanh":
        acivation_fn =  torch.nn.Tanh
    elif args['model']['activation_fn'] == "SELU":
        acivation_fn =  torch.nn.SELU
    else:
        assert False, f"unknown activation function name: {args['model']['activation_fn']}"
    q_value_clipping = args['dqn']['q_value_clipping'] if 'q_value_clipping' in args['dqn'] else 0
    kwargs = {
        "adapter": adapter,
        "dropout_rate": args['model']['dropout'],
        "activation_fn": acivation_fn,
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

def load_weigths(model: DQNNetwork):
    print("Loading weights...")
    weights = f"{MODEL_PATH}/ckpt_{CKPT}.pt"
    model.load_state_dict(torch.load(weights, map_location="cpu")['model'])
    model.to(device)
    adapter.set_model(model)
    model.eval()


def print_system(node_key, system_utterances_history):
    node = Data.objects[0].node_by_key(node_key)
    print("System: ", node.content.text)
    system_utterances_history.append(deepcopy(node.content.text))
    for answer in node.answers:
        print(" - ", answer.content.text)

def handle_skips(current_node: Data.DialogNode, system_utterances_history: List[str], user_utterances_history: List[str]) -> Data.DialogNode:
    if current_node.node_type == Data.NodeType.INFO.value:
        print_system(current_node.key, system_utterances_history)
        input("Please press any key to skip...")
        user_utterances_history.append([""])
        current_node = Data.objects[0].node_by_key(current_node.connected_node_key)
    return current_node

def _transform_dialog_history(user_utterances_history, system_utterances_history):
        # interleave system and user utterances
        # Returns: List[Tuple(sys: str, usr: str)]
        usr_history = user_utterances_history
        if len(usr_history) < len(system_utterances_history):
            usr_history = usr_history + [""]
        assert len(usr_history) == len(system_utterances_history)
        return list(itertools.chain(zip([utterance for utterance in system_utterances_history], [utterance for utterance in user_utterances_history])))

def get_obs(current_node, initial_user_utterance, current_user_utterance, system_utterances_history, user_utterances_history, bst, last_action_idx):
    obs = {
        StateEntry.DIALOG_NODE.value: current_node,
        StateEntry.DIALOG_NODE_KEY.value: current_node.key,
        StateEntry.ORIGINAL_USER_UTTERANCE.value: deepcopy(initial_user_utterance),
        StateEntry.CURRENT_USER_UTTERANCE.value: deepcopy(current_user_utterance),
        StateEntry.SYSTEM_UTTERANCE_HISTORY.value: deepcopy(system_utterances_history),
        StateEntry.USER_UTTERANCE_HISTORY.value: deepcopy(user_utterances_history),
        StateEntry.DIALOG_HISTORY.value: _transform_dialog_history(user_utterances_history, system_utterances_history),
        StateEntry.BST.value: deepcopy(bst),
        StateEntry.LAST_SYSACT.value: last_action_idx,
        StateEntry.NOISE.value: 0.0
    }
    return adapter.encode(obs)

def action_to_text(node, index: int):
    # decode action offsets: ASK=0, SKIP_1=1, ..., SKIP_N=N
    if index == 0:
        return "ASK"
    else:
        return node.answer_by_index(index - 1).content.text

def next_action(current_node, obs):
    # print("Batch size", args['algorithm']["batch_size"])
    state = adapter.batch_state_vector_from_obs([obs, obs, obs], args['algorithm']["batch_size"])
    state = pack_sequence([s.to(device) for s in state], enforce_sorted=False)
    q_values, intent_logits = model(state) # batch x num_max_actions
    ### TEST ###
    # q_values = q_values.view(-1, 7)
    print(intent_logits.size())
    ###
    print("Intent logits", intent_logits)
    print("Q values:", q_values)
    if adapter.configuration.action_masking:
        print("MASKING")
        q_values = torch.masked_fill(q_values, adapter.get_action_masks(node_keys=[current_node.key])[:,:q_values.size(-1)], float('-inf'))
        print(" Masked Q values:", q_values)
    next_action_indices = q_values.argmax(-1).tolist()
    print("Next action", [(action, action_to_text(current_node, action)) for action in next_action_indices])
    intent_classes = None if isinstance(intent_logits, type(None)) else (torch.sigmoid(intent_logits).view(-1) > 0.5).long().tolist()
    print("Intent class", intent_classes)
    print(["FAQ" if intent == 1 else "DIALOG" for intent in intent_classes])
    return next_action_indices
                
args = load_config()
torch.backends.cudnn.deterministic = args["experiment"]["cudnn_deterministic"]

tree = load_data()
text_enc = get_text_embedding(args)
adapter = setup_space_adapter(text_enc)
model = init_model(args)
load_weigths(model)

loop = True
while loop:
    restart = False
    
    # initialize dialog
    turn = 0
    current_node = Data.objects[0].node_by_key(tree.get_start_node().connected_node_key)
    initial_user_utterance = None
    bst = {}
    last_action_idx = 1  # TODO last action index has to start with 0 IF THERE IS NOT STOP ACTION, otherwise 1
    user_utterances_history = []
    system_utterances_history = []

    try:
        while True:
            # system turn
            print_system(current_node.key, system_utterances_history)

            if turn == 0 or last_action_idx == 0:   # TODO last action index has to start with 0 IF THERE IS NOT STOP ACTION, otherwise 1 -> remove turn == 0 from this statement
                initial_user_utterance = input(">>")
                current_user_utterance = deepcopy(initial_user_utterance)
            else:
                current_user_utterance = ""
            user_utterances_history.append(deepcopy(current_user_utterance))
            if current_user_utterance == "exit":
                # Exit system
                loop = False
                break
            elif current_user_utterance == "restart":
                # restart dialog
                break 

            # choose next action
            last_action_idx = next_action(current_node, get_obs(current_node, initial_user_utterance, current_user_utterance, system_utterances_history, user_utterances_history, bst, last_action_idx))[0]

            # progress system
            if last_action_idx > 0:
                current_node = Data.objects[0].node_by_key(current_node.answer_by_index(last_action_idx - 1).connected_node_key)
            # AUTO SKIP MODE current_node = handle_skips(current_node, system_utterances_history, user_utterances_history)

            turn += 1
    except:
        traceback.print_exc()
        
    