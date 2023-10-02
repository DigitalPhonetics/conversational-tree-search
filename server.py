import hashlib
import logging
import multiprocessing
from copy import deepcopy
import os

import tornado
import tornado.httpserver
from tornado.web import Application

from typing import Tuple
from data.dataset import GraphDataset, DataAugmentationLevel
from data.parsers.parserValueProvider import ReimbursementRealValueBackend
from data.parsers.answerTemplateParser import AnswerTemplateParser
from data.parsers.systemTemplateParser import SystemTemplateParser
from data.parsers.logicParser import LogicTemplateParser
from server.formattedDataset import FormattedReimburseGraphDataset
from server.handlers import AuthenticatedWebSocketHandler, BaseHandler, ChatIndex, DataAgreement, LogPostSurvey, LogPreSurvey, LoginHandler, PostSurvey, PreSurvey, ThankYou, KnownEntry
from tornado.web import RequestHandler

from utils.utils import AutoSkipMode, to_class
from algorithm.dqn.dqn import CustomDQN
import torch
from data.cache import Cache
from gymnasium import Env
from encoding.state import StateEncoding
from server.nlu import NLU

from algorithm.dqn.buffer import CustomReplayBuffer
import gymnasium as gym

from hydra import compose, initialize
from omegaconf import DictConfig, OmegaConf
from hydra.core.config_store import ConfigStore
from config import register_configs, DialogLogLevel, WandbLogLevel

DEBUG = True
NUM_GOALS = 2
HARD_GOALS = [
    ("You are trying to figure out how much money you get for booking somewhere to stay on your trip. <ul><li>Your trip is to Tokyo, Japan</li><li>Your trip should take 10 days</li><li>You plan to stay in a hotel</li></ul>", 1234),
    ("You want to figure out how much money you can get reimbursed for your travel. <ul><li>You used your own car</li><li>Your trip was 20km and lasted 8 hours</li><li>You took two colleagues with you</li></ul>", 1235),
    ("You want to know how much money you can get be reimbursed for for your accommodations. <ul><li>You are traveling to France for your next trip</li><li>You plan to stay with your brother in his apartment. </li></ul>", 1236),
    ("You want to know how high the per diem is for food on your next trip. <ul><li>You are traveling to London, England</li><li>You will be there for 72 hours</li></ul>", 1237),
    ("You want to know...", 1238)
]
EASY_GOALS = [
    ("You want to know if you can get reimbursed if you reserve a seat for yourself on the train", 1239),
    ("You are traveling with another colleague and want to know if you have to share a room or if each of you can book your own", 1230),
    ("You are planning on attending a conference and want to know if the membership fee can be reimbursed", 1231),
    ("You have just gotten sick and cannot travel anymore. You want to know if the money you have already paid can be reimbursed", 1232),
    ("You want to know if you can be reimbursed if you need to book a taxi during your trip.", 1233)
    ]
GROUP_ASSIGNMENTS = {"hdc": [], "faq": [], "cts": []}
USER_GOAL_GROUPS = {}
CHAT_ENGINES = {}
DEVICE = "cuda:0"

# on start, check if we have an assignment file, if so, load it and pre-fill group assignments with content
if os.path.isfile("user_log.txt"):
    with open("user_log.txt", "r") as assignments:
        for line in assignments:
            if "GROUP" in line:
                user, group = line.split("||")
                user = user.split(":")[1].strip()
                group = group.split(":")[1].strip()
                GROUP_ASSIGNMENTS[group].append(user)
            elif "GOAL_INDEX" in line:
                user, goal_group = line.split("||")
                user = user.split(":")[1].strip()
                goal_group = goal_group.split(":")[1].strip()
                if goal_group not in USER_GOAL_GROUPS:
                    USER_GOAL_GROUPS[goal_group] = []
                USER_GOAL_GROUPS[goal_group].append(user)


chat_logger = logging.getLogger("chat")
chat_logger.setLevel(logging.INFO)
chat_log_file_handler = logging.FileHandler("chat_log.txt")
chat_log_file_handler.setLevel(logging.INFO)
chat_logger.addHandler(chat_log_file_handler)

survey_logger = logging.getLogger("survey")
survey_logger.setLevel(logging.INFO)
survey_log_file_handler = logging.FileHandler("survey_log.txt")
survey_log_file_handler.setLevel(logging.INFO)
survey_logger.addHandler(survey_log_file_handler)

user_logger = logging.getLogger("user_info")
user_logger.setLevel(logging.INFO)
user_log_file_handler = logging.FileHandler("user_log.txt")
user_log_file_handler.setLevel(logging.INFO)
user_logger.addHandler(user_log_file_handler)

cs = ConfigStore.instance()
register_configs()

## NOTE: assumes already unzipped checkpoint at cfg_path!
cfg_name = "reimburse_realdata_terminalobs"
ckpt_path = '/mount/arbeitsdaten/asr-2/vaethdk/cts_newcodebase_weights/run_1694965093/best_eval/weights/tmp'

multiprocessing.set_start_method("spawn")


def load_model(ckpt_path: str, cfg_name: str, device: str, data: GraphDataset) -> Tuple[DictConfig, CustomDQN, StateEncoding]:
    # load config
    cfg_path = "./conf/"

    with initialize(version_base=None, config_path=cfg_path):
        # parse config
        print("Parsing config...")
        cfg = compose(config_name=cfg_name)
        # print(OmegaConf.to_yaml(cfg))

        # disable logging
        cfg.experiment.logging.dialog_log = DialogLogLevel.NONE
        cfg.experiment.logging.wandb_log = WandbLogLevel.NONE
        cfg.experiment.logging.log_interval = 9999999
        cfg.experiment.logging.keep_checkpoints = 9

        # load encodings
        print("Loading encodings...")
        state_cfg = cfg.experiment.state
        action_cfg = cfg.experiment.actions
        cache = Cache(device=device, data=data, state_config=state_cfg, torch_compile=False)
        encoding = StateEncoding(cache=cache, state_config=state_cfg, action_config=action_cfg, data=data)

        # setup spaces
        action_space = gym.spaces.Discrete(encoding.space_dims.num_actions)
        if encoding.action_config.in_state_space == True:
            # state space: max. node degree (#actions) x state dim
            observation_space = gym.spaces.Box(low=float('-inf'), high=float('inf'), shape=(encoding.space_dims.num_actions, encoding.space_dims.state_vector,)) #, dtype=np.float32)
        else:
            observation_space = gym.spaces.Box(low=float('-inf'), high=float('inf'), shape=(encoding.space_dims.state_vector,)) #, dtype=np.float32)

        class CustomEnv(Env):
            def __init__(self, observation_space, action_space) -> None:
                self.observation_space = observation_space
                self.action_space = action_space
        dummy_env = CustomEnv(observation_space=observation_space, action_space=action_space)

        # setup model
        print("Settung up model...")
        net_arch = OmegaConf.to_container(cfg.experiment.policy.net_arch)
        net_arch['state_dims'] = encoding.space_dims # patch arguments
        optim = OmegaConf.to_container(cfg.experiment.optimizer)
        optim_class = to_class(optim.pop('class_path'))
        lr = optim.pop('lr')
        print("Optim ARGS:", optim_class, lr, optim)
        policy_kwargs = {
            "activation_fn": to_class(cfg.experiment.policy.activation_fn),   
            "net_arch": net_arch,
            "torch_compile": cfg.experiment.torch_compile,
            "optimizer_class": optim_class,
            "optimizer_kwargs": optim
        }
        replay_buffer_kwargs = {
            "num_train_envs": cfg.experiment.environment.num_train_envs,
            "batch_size": cfg.experiment.algorithm.dqn.batch_size,
            "dataset": data,
            "append_ask_action": False,
            # "state_encoding": state_encoding,
            "auto_skip": AutoSkipMode.NONE,
            "normalize_rewards": True,
            "stop_when_reaching_goal": cfg.experiment.environment.stop_when_reaching_goal,
            "stop_on_invalid_skip": cfg.experiment.environment.stop_on_invalid_skip,
            "max_steps": cfg.experiment.environment.max_steps,
            "user_patience": cfg.experiment.environment.user_patience,
            "sys_token": cfg.experiment.environment.sys_token,
            "usr_token": cfg.experiment.environment.usr_token,
            "sep_token": cfg.experiment.environment.sep_token,
            "alpha": cfg.experiment.algorithm.dqn.buffer.backend.alpha,
            "beta": cfg.experiment.algorithm.dqn.buffer.backend.beta,
            "use_lap": cfg.experiment.algorithm.dqn.buffer.backend.use_lap ,
        }
        replay_buffer_class = CustomReplayBuffer
        dqn_target_cls =  to_class(cfg.experiment.algorithm.dqn.targets._target_)
        dqn_target_args = {'gamma': cfg.experiment.algorithm.dqn.gamma}
        dqn_target_args.update(cfg.experiment.algorithm.dqn.targets) 
        model = CustomDQN(policy=to_class(cfg.experiment.policy._target_), policy_kwargs=policy_kwargs,
                    target=dqn_target_cls(**dqn_target_args),
                    seed=cfg.experiment.seed,
                    env=dummy_env, 
                    batch_size=cfg.experiment.algorithm.dqn.batch_size,
                    verbose=1, device=cfg.experiment.device,  
                    learning_rate=lr, 
                    exploration_initial_eps=cfg.experiment.algorithm.dqn.eps_start, exploration_final_eps=cfg.experiment.algorithm.dqn.eps_end, exploration_fraction=cfg.experiment.algorithm.dqn.exploration_fraction,
                    buffer_size=1, # we don't need to store experience, will only increase RAM usage 
                    learning_starts=cfg.experiment.algorithm.dqn.warmup_turns,
                    gamma=cfg.experiment.algorithm.dqn.gamma,
                    train_freq=1, # how many rollouts to perform before training once (one rollout = num_train_envs steps)
                    gradient_steps=max(cfg.experiment.environment.num_train_envs // cfg.experiment.training.every_steps, 1),
                    target_update_interval=cfg.experiment.algorithm.dqn.target_network_update_frequency * cfg.experiment.environment.num_train_envs,
                    max_grad_norm=cfg.experiment.algorithm.dqn.max_grad_norm,
                    tensorboard_log=None,
                    replay_buffer_class=replay_buffer_class,
                    optimize_memory_usage=False,
                    replay_buffer_kwargs=replay_buffer_kwargs,
                    action_masking=cfg.experiment.actions.action_masking,
                    actions_in_state_space=cfg.experiment.actions.in_state_space
                )
        
        # restore weights
        print("Restoring weights...")
        ckpt_params = torch.load(f"{ckpt_path}/policy.pth", map_location=device)
        model.policy.load_state_dict(ckpt_params)
        model.policy.set_training_mode(False)
        model.policy.eval()
    return cfg, model, encoding


# setup data
nlu = NLU()
data = FormattedReimburseGraphDataset('en/reimburse/test_graph.json', 'en/reimburse/test_answers.json', use_answer_synonyms=True, augmentation=DataAugmentationLevel.NONE, resource_dir='resources')
# setup data & parsers
answerParser = AnswerTemplateParser()
logicParser = LogicTemplateParser()
sysParser = SystemTemplateParser()
valueBackend = ReimbursementRealValueBackend(a1_laender=data.a1_countries, data=data.hotel_costs)
# setup model and encoding
cfg, model, state_encoding = load_model(ckpt_path=ckpt_path, cfg_name=cfg_name, device=DEVICE, data=data)


class CheckLogin(RequestHandler):
    def post(self):
        username = self.get_body_argument("username").encode()
        h = hashlib.shake_256(username)
        user_id = h.hexdigest(15)
        known_users = set()
        for group in GROUP_ASSIGNMENTS:
            known_users.update(GROUP_ASSIGNMENTS[group])
        if user_id in known_users:
            self.redirect("/known_entry")
        else:
            self.set_secure_cookie("user", user_id)
            self.redirect("/data_agreement")


class UserAgreed(BaseHandler):
    def post(self):
        global GROUP_ASSIGNMENTS
        logging.getLogger("user_info").info(f"USER: {self.current_user} || AGREED: True")

        # If user is not assigned, assign to group with fewest participants
        if not self.get_cookie("group_assignment"):
            group = sorted(GROUP_ASSIGNMENTS.items(), key=lambda item: len(item[1]))[0][0]
            self.set_cookie("group_assignment", group)
            GROUP_ASSIGNMENTS[group].append(self.current_user)

            # add assignment to file
            logging.getLogger("user_info").info(f"USER: {self.current_user} || GROUP: {group}")
            logging.getLogger("chat").info(f"USER: {self.current_user} || GROUP: {group}")

        # Assign the 
        if not self.get_cookie_("goal_group"):
            goal_group = sorted(USER_GOAL_GROUPS.items(), key=lambda item: len(item[1]))[0][0]
            self.set_cookie("goal_group", goal_group)
            USER_GOAL_GROUPS[goal_group].append(self.current_user)
            logging.getLogger("user_info").info(f"USER: {self.current_user} || GOAL_INDEX: {goal_group}")

        self.redirect(f"/pre_survey")
        

class UserChatSocket(AuthenticatedWebSocketHandler):
    def open(self):
        global CHAT_ENGINES
        print(f"Opened socket for user: {self.current_user}")
        print(f"starting dialog system for user {self.current_user}")
        logging.getLogger("chat").info(f"==== NEW DIALOG STARTED FOR USER {self.current_user} ====")
        # TODO initialise new dialog for user
        if not self.current_user in CHAT_ENGINES:
            # TODO swap out chat engine! (CTS, Free or guided baselines)
            CHAT_ENGINES[self.current_user] = ChatEngine(self.current_user, self)
        else:
            CHAT_ENGINES[self.current_user].socket = self

        # choose a goal
        if not self.get_cookie("goal_counter"):
            self.set_cookie("goal_counter", 0)
        goal_group = int(self.get_cookie("goal_group"))
        goal_counter = int(self.set_cookie("goal_counter"))
        if goal_counter == 0:
            goal, node_id = HARD_GOALS[goal_group]
        else:
            goal, node_id = EASY_GOALS[goal_group]
        logging.getLogger("chat").info(f"USER: {self.current_user} || GOAL: {goal} || NODE_ID: {node_id}")
        self.write_message({"EVENT": "NEW_GOAL", "VALUE": goal})            
        CHAT_ENGINES[self.current_user].start_dialog()

    def on_message(self, message):
        global NUM_GOALS
        global CHAT_ENGINES
        data = tornado.escape.json_decode(message)
        event = data["EVENT"]
        value = data["VALUE"]
        if event == "MSG":
            # forward message to (correct) dialog system
            print(f"MSG for user {self.current_user}: {message}")
            logging.getLogger("chat").info(f"MSG USER ({self.current_user}): {value}")
            CHAT_ENGINES[self.current_user].user_reply(value)
        elif event == "RESTART":
            # restart dialog
            self.write_message({"EVENT": "RESTART", "VALUE": True})
            logging.getLogger("chat").info(f"USER ({self.current_user} NEW DIALOG)")
            CHAT_ENGINES[self.current_user].start_dialog()
        elif"NEXT_GOAL" in event:
            # update the goal counter
            goal_counter = int(self.get_cookie("goal_counter"))
            goal_counter += 1
            self.set_cookie("goal_counter", goal_counter)

            # Log success/failure
            if event == "NEXT_GOAL_CORRECT":
                logging.getLogger("chat").info(f"USER: {self.current_user} || DIALOG END: SUCCESS")
            elif event == "NEXT_GOAL_INCORRECT":
                logging.getLogger("chat").info(f"USER: {self.current_user} || DIALOG END: FAILURE")
            else:
                logging.getLogger("chat").info(f"USER: {self.current_user} || DIALOG END: UNKNOWN CONDITION")
            with open("test_log.txt", "a") as f:
                f.writelines([line + "\n" for line in CHAT_ENGINES[self.current_user].user_env.episode_log])
            # Interaction over, redirect to the post-survey
            if goal_counter >= NUM_GOALS:
                self.write_message({"EVENT": "EXPERIMENT_OVER", "VALUE": True})
                # Start a new dialog
                self.write_message({"EVENT": "RESTART", "VALUE": True})
            else:  # choose a new goal
                # There are only two goals chosen, so if we get here, we know we're already on the second gaol
                goal_group = int(self.get_cookies("goal_group"))
                next_goal, node_id = EASY_GOALS[goal_group]
                self.write_message({"EVENT": "NEW_GOAL", "VALUE": next_goal})
                logging.getLogger("chat").info(f"USER: {self.current_user} || GOAL: {next_goal} || NODE_ID: {node_id}")
                logging.getLogger("chat").info(f"USER ({self.current_user} NEW DIALOG)")
                CHAT_ENGINES[self.current_user].start_dialog()
       
    def on_close(self):
        print(f"Closing connection for user {self.current_user}")

if __name__ == "__main__":
    settings = {
        "login_url": "/",
        "cookie_secret": "YOUR_SECRET_KEY",
        "debug": DEBUG,
        "static_path": "./node_modules"
    }
    print("settings created")
    app = Application([
        (r"/", LoginHandler),
        (r"/post_survey", PostSurvey),
        (r"/pre_survey", PreSurvey),
        (r"/check_login", CheckLogin),
        (r"/chat", ChatIndex),
        (r"/channel", UserChatSocket),
        (r"/log_pre_survey", LogPreSurvey),
        (r"/log_post_survey", LogPostSurvey),
        (r"/data_agreement", DataAgreement),
        (r"/thank_you", ThankYou),
        (r"/known_entry", KnownEntry),
        (r"/agreed_to_data_collection", UserAgreed)
    ], **settings)
    print("created app")
    http_server = tornado.httpserver.HTTPServer(app) #, ssl_options = ssl_ctx)
    http_server.listen(44123)
    print("set up server address")

    io_loop = tornado.ioloop.IOLoop.current()
    print("created io loop")
    io_loop.start()