import asyncio
import logging
import hashlib
import multiprocessing

import tornado
import tornado.httpserver
from tornado.web import RequestHandler, Application
from tornado.websocket import WebSocketHandler

from environment.realuser import RealUserEnvironment
from typing import List, Tuple
from data.dataset import GraphDataset, ReimburseGraphDataset, DataAugmentationLevel
from data.parsers.parserValueProvider import ReimbursementRealValueBackend
from data.parsers.answerTemplateParser import AnswerTemplateParser
from data.parsers.systemTemplateParser import SystemTemplateParser
from data.parsers.logicParser import LogicTemplateParser
from utils.utils import AutoSkipMode, to_class
from algorithm.dqn.dqn import CustomDQN
import torch
from data.cache import Cache
from gymnasium import Env
from encoding.state import StateEncoding

from config import ActionType, DialogLogLevel, WandbLogLevel
from algorithm.dqn.her import HindsightExperienceReplayWrapper
import gymnasium as gym

from hydra import compose, initialize
from omegaconf import DictConfig, OmegaConf
from hydra.core.config_store import ConfigStore
from config import register_configs

DEBUG = True
NUM_GOALS = 2
GROUP_ASSIGNMENTS = {"hdc": [], "faq": [], "cts": []}
USER_GOAL_NUM = {}
CHAT_ENGINES = {}
DEVICE = "cpu"

# on start, check if we have an assignment file, if so, load it and pre-fill group assignments with content
with open("user_log.txt", "a+") as assignments:
    print(assignments)
    print(type(assignments))
    for line in assignments:
        if "GROUP" in line:
            user, group = line.split("||")
            user = user.split(":")[1].strip()
            group = group.split(":")[1].strip()
            GROUP_ASSIGNMENTS[group].append(user)


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
cfg_name = "reimburse_generated_v1_terminalobs"
ckpt_path = '/mount/arbeitsdaten/asr-2/vaethdk/cts_newcodebase_weights/run_1695028356/best_eval/weights/test'

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
            "use_lap": cfg.experiment.algorithm.dqn.buffer.backend.use_lap 
        }
        replay_buffer_class = HindsightExperienceReplayWrapper
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
                    buffer_size=cfg.experiment.algorithm.dqn.buffer.backend.buffer_size, 
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

def load_env(data: GraphDataset, answer_parser, logic_parser, value_backend) -> RealUserEnvironment:
    # setup env
    env = RealUserEnvironment(dataset=data, 
                        sys_token="SYSTEM", usr_token="USER", sep_token="",
                        max_steps=50, max_reward=150, user_patience=2,
                        answer_parser=answerParser, logic_parser=logicParser, value_backend=valueBackend,
                        auto_skip=AutoSkipMode.NONE, stop_on_invalid_skip=False)
    return env

# setup data
data = ReimburseGraphDataset('en/reimburse/test_graph.json', 'en/reimburse/test_answers.json', use_answer_synonyms=True, augmentation=DataAugmentationLevel.NONE, resource_dir='resources')
# setup data & parsers
answerParser = AnswerTemplateParser()
logicParser = LogicTemplateParser()
sysParser = SystemTemplateParser()
valueBackend = ReimbursementRealValueBackend(a1_laender=data.a1_countries, data=data.hotel_costs)
# setup model and encoding
cfg, model, state_encoding = load_model(ckpt_path=ckpt_path, cfg_name=cfg_name, device=DEVICE, data=data)


def choose_user_goal(user_id: str):
    global USER_GOAL_NUM
    if USER_GOAL_NUM[user_id] == 1:
        return "You are trying to figure out how much money you get for booking somewhere to stay on your trip. <ul><li>Your trip is to Tokyo, Japan</li><li>You plan to stay in a hotel</li></ul>"
    else:
        return "You want to figure out how much money you can get reimbursed for your travel. <ul><li>You used your own car</li><li>Your trip was 20km and lasted 8 hours</li><li>You took two colleagues with you</li></ul>"

## TODO: write new GUI module
## - BST
## - Conversation history
## - GC
## - Logging?
## - Survey processing?
class ChatEngine:
    def __init__(self, user_id: str, socket) -> None:
        self.user_id = user_id
        self.user_env = load_env(data, answerParser, logicParser, valueBackend) # contains .bst, .reset()
        self.current_obs = None
        self.socket = socket

    def start_dialog(self) -> None:
        self.current_obs = self.reset()

    def user_reply(self, user_utterance: str):
        # TODO is it enough to set current user utterance ?
        self.user_env.current_user_utterance = user_utterance

    def system_reply(self) -> List[str]:
        # encode observation
        s = state_encoding.batch_encode(observation=[self.user_env.get_obs()], sys_token=cfg.experiment.environment.sys_token, usr_token=cfg.experiment.environment.usr_token, sep_token=cfg.experiment.environment.sep_token) 
        # predict action & intent
        action, intent = model.predict(observation=s, deterministic=True)
        action = int(action)
        intent = intent.item()

        msgs = []

        if action == ActionType.ASK:
            # get system message from action
            msgs.append(self.user_env.current_node.text)

            # TODO continue automatic looping after certain nodes
            # - info node (because there will be no expected user input)
        elif action >= ActionType.SKIP:
            # continue skipping
            msgs.extend(self.system_reply())

        for msg in msgs:
            self.socket.write_message(msg)


class BaseHandler(RequestHandler):
    def get_current_user(self):
        return tornado.escape.to_unicode(self.get_secure_cookie("user"))    

class AuthenticatedWebSocketHandler(WebSocketHandler):
    def get_current_user(self):
        return tornado.escape.to_unicode(self.get_secure_cookie("user"))



class LoginHandler(BaseHandler):
    def get(self):
        if self.current_user:
            self.redirect("/data_agreement")
        else:
            self.render("server/templates/login.html")


class CheckLogin(RequestHandler):
    def post(self):
        username = self.get_body_argument("username").encode()
        h = hashlib.shake_256(username)
        self.set_secure_cookie("user", h.hexdigest(15))
        self.redirect("/data_agreement")

class LogPreSurvey(BaseHandler):
    def post(self):
        global GROUP_ASSIGNMENTS
        results = self.request.body_arguments
        results = {key : str(results[key][0])[2:-1] for key in results}
        logging.getLogger("survey").info(f"USER: {self.current_user} || PRE-SURVEY: {results}")
        self.redirect(f"/chat")

class LogPostSurvey(BaseHandler):
    def post(self):
        results = self.request.body_arguments
        results = {key : str(results[key][0])[2:-1] for key in results}
        logging.getLogger("survey").info(f"USER: {self.current_user} || POST-SURVEY: {results}")
        self.redirect(f"/thank_you")

class UserAgreed(BaseHandler):
    def post(self):
        logging.getLogger("user_info").info(f"USER: {self.current_user} || AGREED: True")

        # If user is not assigned, assign to group with fewest participants
        if not self.get_cookie("group_assignment"):
            group = sorted(GROUP_ASSIGNMENTS.items(), key=lambda item: len(item[1]))[0][0]
            self.set_cookie("group_assignment", group)

            # add assignment to file
            logging.getLogger("user_info").info(f"USER: {self.current_user} || GROUP: {group}")
            logging.getLogger("chat").info(f"USER: {self.current_user} || GROUP: {group}")

        self.redirect(f"/pre_survey")
        

class ChatIndex(BaseHandler):
    @tornado.web.authenticated
    def get(self):
        global USER_GOAL_NUM
        # TODO: actually choose a goal
        goal = choose_user_goal(self.current_user)
        logging.getLogger("chat").info(f"USER: {self.current_user} || GOAL: {goal}")
        self.render("server/templates/chat.html", goal=goal)

class DataAgreement(BaseHandler):
    @tornado.web.authenticated
    def get(self):
        self.render("server/templates/data_agreement.html")

class PostSurvey(BaseHandler):
    @tornado.web.authenticated
    def get(self):
        self.render("server/templates/post_survey.html")

class PreSurvey(BaseHandler):
    @tornado.web.authenticated
    def get(self):
        self.render("server/templates/pre_survey.html")

class ThankYou(BaseHandler):
    @tornado.web.authenticated
    def get(self):
        self.render("server/templates/thank_you.html", completion_key=self.current_user)


class UserChatSocket(AuthenticatedWebSocketHandler):
    def open(self):
        global CHAT_ENGINES
        print(f"Opened socket for user: {self.current_user}")
        print(f"starting dialog system for user {self.current_user}")
        logging.getLogger("chat").info(f"==== NEW DIALOG STARTED FOR USER {self.current_user} ====")
        # TODO initialise new dialog for user
        if not self.current_user in CHAT_ENGINES:
            CHAT_ENGINES[self.get_current_user] = ChatEngine(self.current_user, self)
        CHAT_ENGINES[self.get_current_user].start_dialog()

    def on_message(self, message):
        global NUM_GOALS
        global USER_GOAL_NUM
        global CHAT_ENGINES
        data = tornado.escape.json_decode(message)
        event = data["EVENT"]
        value = data["VALUE"]
        if event == "MSG":
            # forward message to (correct) dialog system
            print(f"MSG for user {self.current_user}: {message}")
            logging.getLogger("chat").info(f"MSG USER ({self.current_user}): {value}")
            CHAT_ENGINES[self.current_user].user_reply(message)
        elif event == "RESTART":
            logging.getLogger("chat").info(f"USER ({self.current_user} FINISHED DIALOG)")
            # restart dialog
            CHAT_ENGINES[self.current_user].start_dialog()
        elif event == "NEXT_GOAL":
            # TODO choose a new goal
            if USER_GOAL_NUM[self.current_user] >= NUM_GOALS:
                self.write_message({"EVENT": "EXPERIMENT_OVER", "VALUE": True})
            else:
                USER_GOAL_NUM[self.current_user] += 1
                next_goal = choose_user_goal(self.current_user)
                self.write_message({"EVENT": "NEW_GOAL", "VALUE": next_goal})
                logging.getLogger("chat").info(f"USER: {self.current_user} || GOAL: {next_goal}")
       
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
        (r"/agreed_to_data_collection", UserAgreed)
    ], **settings)
    print("created app")
    http_server = tornado.httpserver.HTTPServer(app) #, ssl_options = ssl_ctx)
    http_server.listen(44123)
    print("set up server address")

    io_loop = tornado.ioloop.IOLoop.current()
    print("created io loop")
    io_loop.start()