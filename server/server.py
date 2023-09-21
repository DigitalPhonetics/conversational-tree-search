import asyncio
import logging
import multiprocessing

import tornado
import tornado.httpserver
from tornado.web import RequestHandler, Application
from tornado.websocket import WebSocketHandler

from environment.realuser import RealUserEnvironment
from typing import Tuple
from data.dataset import GraphDataset, ReimburseGraphDataset, DataAugmentationLevel
from data.parsers.parserValueProvider import RealValueBackend
from data.parsers.answerTemplateParser import AnswerTemplateParser
from data.parsers.systemTemplateParser import SystemTemplateParser
from data.parsers.logicParser import LogicTemplateParser
from utils.utils import AutoSkipMode
from algorithm.dqn.dqn import CustomDQN
import torch
from data.cache import Cache
from gymnasium import Env
from encoding.state import StateEncoding

from config import DialogLogLevel, WandbLogLevel
from algorithm.dqn.her import HindsightExperienceReplayWrapper
import gymnasium as gym

from hydra import compose, initialize
from omegaconf import DictConfig, OmegaConf
from hydra.core.config_store import ConfigStore
from config import register_configs

cs = ConfigStore.instance()
register_configs()

## NOTE: assumes already unzipped checkpoint at cfg_path!
cfg_name = "reimburse_generated_v1_terminalobs"
ckpt_path = '/mount/arbeitsdaten/asr-2/vaethdk/cts_newcodebase_weights/run_1695028356/best_eval/weights/test'

multiprocessing.set_start_method("spawn")

logger = logging.getLogger("chat")
logger.setLevel(logging.INFO)
log_file_handler = logging.FileHandler("chat_log.txt")
log_file_handler.setLevel(logging.INFO)
logger.addHandler(log_file_handler)

DEBUG = True
NUM_CONDITIONS = 2

CURRENT_USER_CONNECTION = None 

def to_class(path:str):
    from pydoc import locate
    class_instance = locate(path)
    return class_instance



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

def load_env(data: GraphDataset) -> RealUserEnvironment:
    # setup data & parsers
    answerParser = AnswerTemplateParser()
    logicParser = LogicTemplateParser()
    sysParser = SystemTemplateParser()
    valueBackend = RealValueBackend(a1_laender=data.a1_countries, data=data.hotel_costs)

    # setup env
    env = RealUserEnvironment(dataset=data, 
                        sys_token="SYSTEM", usr_token="USER", sep_token="",
                        max_steps=50, max_reward=150, user_patience=2,
                        answer_parser=answerParser, logic_parser=logicParser, value_backend=valueBackend,
                        auto_skip=AutoSkipMode.NONE, stop_on_invalid_skip=False)
    return env

data = ReimburseGraphDataset('en/reimburse/test_graph.json', 'en/reimburse/test_answers.json', use_answer_synonyms=True, augmentation=DataAugmentationLevel.NONE, resource_dir='resources')
cfg, model, state_encoding = load_model(ckpt_path=ckpt_path, cfg_name=cfg_name, device='cpu', data=data)

def next_action(env: RealUserEnvironment) -> Tuple[int, bool]:
    # encode observation
    s = state_encoding.batch_encode(observation=[env.get_obs()], sys_token=cfg.experiment.environment.sys_token, usr_token=cfg.experiment.environment.usr_token, sep_token=cfg.experiment.environment.sep_token) 
    # predict action & intent
    action, intent = model.predict(observation=s, deterministic=True)
    action = int(action)
    intent = intent.item()

    return action, intent



## TODO 
class GUIInterface(Service):
    def __init__(self, domain = None, logger =  None):
        Service.__init__(self, domain=domain)
        self.logger = logger
        self.loopy_loop = asyncio.new_event_loop()

    @PublishSubscribe(sub_topics=['sys_utterance'])
    def send_sys_output(self, user_id: str = "default", sys_utterance: str = ""):
        asyncio.set_event_loop(self.loopy_loop)
        print("SYS MSG:", sys_utterance)
        if self.get_state(user_id, 'socket'):
            logging.getLogger("chat").info(f"MSG SYSTEM ({user_id}): {sys_utterance}")
            self.get_state(user_id, 'socket').write_message({"EVENT": "MSG", "VALUE": sys_utterance})
        else:
            print("NOT CONNECTED")

    # TODO: update once we have speech and not text
    @PublishSubscribe(pub_topics=["user_utterance"])
    def forward_to_dialog_system(self, user_id, user_utterance):
        print("USER", user_id)
        print("PUBLISH TO DIALOG SYSTEM")
        return {"user_utterance": user_utterance}
    

# TODO 
gui_interface = GUIInterface(domain=domain)
bst = HandcraftedBST(domain=domain)
ds = DialogSystem([gui_interface, nlg, nlu, bst, policy, adaptivity])
error_free = ds.is_error_free_messaging_pipeline()
if not error_free:
    ds.print_inconsistencies()

class BaseHandler(RequestHandler):
    def get_current_user(self):
        return tornado.escape.to_unicode(self.get_secure_cookie("user"))    

class AuthenticatedWebSocketHandler(WebSocketHandler):
    def get_current_user(self):
        return tornado.escape.to_unicode(self.get_secure_cookie("user"))



class LoginHandler(BaseHandler):
    def get(self):
        if self.current_user:
            if not self.get_cookie("seen_conditions"):
                self.set_cookie("seen_conditions", "0")
            self.redirect("/chat")
        else:
            self.render("./templates/login.html")


class CheckLogin(RequestHandler):
    def post(self):
        username = self.get_body_argument("username")
        self.set_secure_cookie("user", username)
        if not self.get_cookie("seen_conditions"):
            self.set_cookie("seen_conditions", "0")
        self.redirect("/chat")

class LogSurvey(BaseHandler):
    def post(self):
        global NUM_CONDITIONS
        seen_conditions = int(self.get_cookie('seen_conditions')) + 1
        self.set_cookie("seen_conditions", str(seen_conditions))
        results = self.request.body_arguments
        results = {key : str(results[key][0])[2:-1] for key in results}
        logging.getLogger("chat").info(f"USER {self.current_user}; SURVEY: {results}")
        if seen_conditions < NUM_CONDITIONS:
            self.redirect(f"/chat")
        else:
            self.redirect(f"/thank_you")
        

class ChatIndex(BaseHandler):
    @tornado.web.authenticated
    def get(self):
        global NUM_CONDITIONS
        self.render("./templates/chat.html", total_conditions=NUM_CONDITIONS)

class Survey(BaseHandler):
    @tornado.web.authenticated
    def get(self):
        self.render("./templates/survey.html")

class PreSurvey(BaseHandler):
    @tornado.web.authenticated
    def get(self):
        self.render("./templates/pre_survey.html")

class ThankYou(BaseHandler):
    @tornado.web.authenticated
    def get(self):
        self.render("./templates/thank_you.html")


class UserChatSocket(AuthenticatedWebSocketHandler):
    def open(self):
        global gui_interface
        global ds
        print(f"Opened socket for user: {self.current_user}")
        gui_interface.set_state(user_id=self.current_user, attribute_name='socket', attribute_value=self)
        ds._start_dialog(start_signals={"user_utterance/GasStations": ""}, user_id=self.current_user)
        print(f"starting dialog system for user {self.current_user}")
        logging.getLogger("chat").info(f"==== NEW DIALOG STARTED FOR USER {self.current_user} ====")

    def on_message(self, message):
        global gui_interface
        data = tornado.escape.json_decode(message)
        event = data["EVENT"]
        value = data["VALUE"]
        if event == "MSG":
            # TODO: might need to update for audio files
            print(f"MSG for user {self.current_user}: {message}")
            logging.getLogger("chat").info(f"MSG USER ({self.current_user}): {value}")
            gui_interface.forward_to_dialog_system(self.current_user, value)
        if event == "END_DIALOG":
            logging.getLogger("chat").info(f"USER ({self.current_user} FINISHED DIALOG)")
       
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
        (r"/post_survey", Survey),
        (r"/pre_survey", PreSurvey),
        (r"/check_login", CheckLogin),
        (r"/chat", ChatIndex),
        (r"/channel", UserChatSocket),
        (r"/log_survey", LogSurvey),
        (r"/thank_you", ThankYou)
    ], **settings)
    print("created app")
    http_server = tornado.httpserver.HTTPServer(app) #, ssl_options = ssl_ctx)
    http_server.listen(8000)
    print("set up server address")

    io_loop = tornado.ioloop.IOLoop.current()
    print("created io loop")
    io_loop.start()
    
    ds.print_inconsistencies()
    # ds._start_dialog(start_signals={"user_utterance/gasstations": ""}, user_id="default")
    # gui_interface.forward_to_dialog_system("Hi")

