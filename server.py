import multiprocessing
if not multiprocessing.get_start_method(allow_none=True):
    multiprocessing.set_start_method("spawn")
import pickle

from sentence_transformers import SentenceTransformer
import torch

import hashlib
import logging

import os
import random
import traceback
import ssl

import tornado
import tornado.httpserver
from tornado.web import Application

from data.dataset import DataAugmentationLevel, NodeType
from data.parsers.parserValueProvider import ReimbursementRealValueBackend
from data.parsers.answerTemplateParser import AnswerTemplateParser
from data.parsers.systemTemplateParser import SystemTemplateParser
from data.parsers.logicParser import LogicTemplateParser
from server.cts_llm_policy import LLMPolicy, calculate_node_text_embeddings, get_node_candidate_list_by_type, parallel_path_computation
from server.cts_llm_policy import system_1 as sys_prompt_1
from server.formattedDataset import FormattedReimburseGraphDataset
from server.handlers import AuthenticatedWebSocketHandler, BaseHandler, ChatIndex, DataAgreement, LogPostSurvey, LogPreSurvey, LoginHandler, PostSurvey, PreSurvey, ThankYou, KnownEntry
from tornado.web import RequestHandler

from server.nlu import NLU

MODEL = "gpt-4o-2024-08-06" # or "gpt-4o", "gpt-4o-mini"
TEMPERATURE = 0.0
SEED = 43
TOP_K = 15
MAX_STEPS = 100
USER_PATIENCE = 5
RETRIEVAL_NODE_TYPES = [NodeType.INFO, NodeType.QUESTION]
VERBOSE = False


DEBUG = False
NUM_GOALS = 3
HARD_GOALS = [
    ("You are trying to figure out how much money you get for booking somewhere to stay on your trip. <ul><li>Your trip is to Tokyo, Japan</li><li>Your trip should take 10 days</li><li>You plan to stay in a hotel</li></ul>", 16365521324065600),
    ("You want to figure out how much money you can get for your travel. <ul><li>You used your own car</li><li>You took two colleagues with you</li></ul>", 16460328708250870),
    ("You want to know how much money you can get for your accommodations. <ul><li>You are traveling to France for your next trip</li><li>You plan to stay with your brother in his apartment. </li></ul>", 16378349334755637),
    ("You want to know how the reimbursement process works for research semester.<ul><li>You plan to bring your family with you</li></ul>", 16370483534787100),
    ("You want to know how to book your flight.<ul><li>You plan to extend your stay with private vacation before flying back</li></ul>", 16363755463439219)
]
EASY_GOALS = [
    ("You want to know if you can get reimbursed if you reserve a seat for yourself on the train", 16363756478730906),
    ("You are traveling with another colleague and want to know if you have to share a room or if each of you can book your own", 16363834594338823),
    ("You are booking a trip where you plan to attend a conference and want to know if the conference fee can be reimbursed", 16384329210117153),
    ("You have to cancel your trip. You want to know if the money you have already paid can be reimbursed", 16457053159041482),
    ("You are planning to book a trip, and want to know If you can get reimbursed for taking a taxi during your trip.", 16365525829145685)
    ]

OPEN_GOALS = [
    ("You want to know how to book a hotel", 16387868859695624),
    ("You want to know how to book a flight", 16387868859695624),
    ("You want more information about how to plan a research semester.", 16387868859695624),
    ("You want to inform yourself what to do in case of an emergency during travel.", 16387868859695624)
    ]
POLICY_ASSIGNMENT = {"cts_llm": []}
USER_GOAL_GROUPS_OPEN = {i: [] for i in range(len(OPEN_GOALS))}
USER_GOAL_GROUPS_EASY = {i: [] for i in range(len(EASY_GOALS))}
USER_GOAL_GROUPS_HARD = {i: [] for i in range(len(HARD_GOALS))}
CHAT_ENGINES = {}


class CheckLogin(RequestHandler):
    def post(self):
        username = self.get_body_argument("username").encode()
        h = hashlib.shake_256(username)
        user_id = h.hexdigest(15)
        if user_id in POLICY_ASSIGNMENT["cts_llm"]:
            self.redirect("/known_entry")
        else:
            self.set_secure_cookie("user", user_id)
            self.redirect("/data_agreement")


class UserAgreed(BaseHandler):
    def post(self):
        global POLICY_ASSIGNMENT, USER_GOAL_GROUPS_HARD, USER_GOAL_GROUPS_EASY, USER_GOAL_GROUPS_OPEN
        logging.getLogger("user_info").info(f"USER: {self.current_user} || AGREED: True")

        # assign to cts_llm group
        group = "cts_llm"
        POLICY_ASSIGNMENT[group].append(self.current_user)

        # add assignment to file
        logging.getLogger("user_info").info(f"USER: {self.current_user} || GROUP: {group}")
        logging.getLogger("chat").info(f"USER: {self.current_user} || GROUP: {group}")

        # Assign the goal group
        if not self.get_cookie("goal_groups"):
            goal_group_open = random.choices(list(range(len(USER_GOAL_GROUPS_OPEN))), weights=[1./(len(USER_GOAL_GROUPS_OPEN[i]) + 1) for i in USER_GOAL_GROUPS_OPEN], k=1)[0]
            goal_group_easy = random.choices(list(range(len(USER_GOAL_GROUPS_EASY))), weights=[1./(len(USER_GOAL_GROUPS_EASY[i]) + 1) for i in USER_GOAL_GROUPS_EASY], k=1)[0]
            goal_group_hard = random.choices(list(range(len(USER_GOAL_GROUPS_HARD))), weights=[1./(len(USER_GOAL_GROUPS_HARD[i]) + 1) for i in USER_GOAL_GROUPS_HARD], k=1)[0]
            self.set_cookie("goal_groups", f"{goal_group_open},{goal_group_easy},{goal_group_hard}")
            USER_GOAL_GROUPS_OPEN[goal_group_open].append(self.current_user)
            USER_GOAL_GROUPS_EASY[goal_group_easy].append(self.current_user)
            USER_GOAL_GROUPS_HARD[goal_group_hard].append(self.current_user)
            logging.getLogger("user_info").info(f"USER: {self.current_user} || GOAL_INDICES: {goal_group_open},{goal_group_easy},{goal_group_hard}")

        self.redirect(f"/pre_survey")
        

class UserChatSocket(AuthenticatedWebSocketHandler):
    def open(self):
        global CHAT_ENGINES
        print(f"Opened socket for user: {self.current_user}")
        print(f"starting dialog system for user {self.current_user}")
        logging.getLogger("chat").info(f"==== NEW DIALOG STARTED FOR USER {self.current_user} ====")
        if not self.current_user in CHAT_ENGINES:
            # Create policy for group assignment and user
            CHAT_ENGINES[self.current_user] = LLMPolicy(node_list=node_list, node_embedding=node_text_embeddings, bi_encoder=bi_encoder,
                                                        user_id=self.current_user, socket=self,
                                                        path_cache=path_cache,
                                                        data=data, nlu=nlu, sysParser=sysParser, answerParser=answerParser, logicParser=logicParser, value_backend=valueBackend,
                                                        model=MODEL, temperature=TEMPERATURE, seed=SEED, top_k=TOP_K,
                                                        verbose=VERBOSE, strict=False,
                                                        sys_prompt=sys_prompt_1)
        else:
            CHAT_ENGINES[self.current_user].socket = self

        # choose a goal
        if not self.get_cookie("goal_counter"):
            self.set_cookie("goal_counter", str(0))
        goal_groups = [int(group) for group in self.get_cookie("goal_groups").split(",")]
        goal_counter = int(self.get_cookie("goal_counter"))
        goal_group = goal_groups[goal_counter]
        if goal_counter == 0:
            goal, node_id = OPEN_GOALS[goal_group]
        elif goal_counter == 1:
            goal, node_id = EASY_GOALS[goal_group]
        else:
            goal, node_id = HARD_GOALS[goal_group]
        logging.getLogger("chat").info(f"USER: {self.current_user} || GOAL: {goal} || NODE_ID: {node_id}")
        self.write_message({"EVENT": "NEW_GOAL", "VALUE": goal})            
        CHAT_ENGINES[self.current_user].reset(node_id)

    def on_message(self, message):
        global NUM_GOALS
        global CHAT_ENGINES
        data = tornado.escape.json_decode(message)
        event = data["EVENT"]
        value = data["VALUE"]
        if event == "MSG":
            # forward message to (correct) dialog system
            # print(f"MSG for user {self.current_user}: {message}")
            # logging.getLogger("chat").info(f"MSG USER ({self.current_user}): {value}")
            try:
                CHAT_ENGINES[self.current_user].user_reply(value)
            except:
                traceback.print_exc()
                logging.getLogger("chat").error(traceback.format_exc())
                self.write_message({"EVENT": "MSG", "VALUE": "Sorry, but the system encountered an error. Please restart the dialog / reload the page, and if that leads to an error again, please end this dialog by click on the <b>Finished Dialog</b> button on the right.",  "CANDIDATES": [], "NODE_TYPE": "infoNode" })
        elif event == "RESTART":
            # restart dialog
            self.write_message({"EVENT": "RESTART", "VALUE": True})
            if not CHAT_ENGINES[self.current_user].done:
                logging.getLogger("chat").info(f'{self.current_user}-{CHAT_ENGINES[self.current_user].current_episode}$=> REACHED GOAL ONCE: {CHAT_ENGINES[self.current_user].reached_goal_once}')
                logging.getLogger("chat").info(f'{self.current_user}-{CHAT_ENGINES[self.current_user].current_episode}$=> ASKED GOAL ONCE: {CHAT_ENGINES[self.current_user].asked_goal_once}')
                logging.getLogger("chat").info(f'{self.current_user}-{CHAT_ENGINES[self.current_user].current_episode}$=> PERCIEVED LENGTH: {CHAT_ENGINES[self.current_user].perceived_length}')
            logging.getLogger("chat").info(f"=== USER ({self.current_user}) RESTART === ")
            CHAT_ENGINES[self.current_user].reset(None)
        elif event == "NEXT_GOAL":
            # update the goal counter
            goal_counter = value["goal_counter"]
            # log user rating for current dialog
            if not CHAT_ENGINES[self.current_user].done:
                logging.getLogger("chat").info(f'{self.current_user}-{CHAT_ENGINES[self.current_user].current_episode}$=> REACHED GOAL ONCE: {CHAT_ENGINES[self.current_user].reached_goal_once}')
                logging.getLogger("chat").info(f'{self.current_user}-{CHAT_ENGINES[self.current_user].current_episode}$=> ASKED GOAL ONCE: {CHAT_ENGINES[self.current_user].asked_goal_once}')
                logging.getLogger("chat").info(f'{self.current_user}-{CHAT_ENGINES[self.current_user].current_episode}$=> PERCIEVED LENGTH: {CHAT_ENGINES[self.current_user].perceived_length}')
            logging.getLogger("chat").info(f"USER: {self.current_user} || QUALITY: {value['quality']}")
            logging.getLogger("chat").info(f"USER: {self.current_user} || LENGTH: {value['length']}")

            # Interaction over, redirect to the post-survey
            if goal_counter >= NUM_GOALS:
                self.write_message({"EVENT": "EXPERIMENT_OVER", "VALUE": True})
                # Start a new dialog
                self.write_message({"EVENT": "RESTART", "VALUE": True})
            else:  # choose a new goal
                goal_groups = [int(group) for group in self.get_cookie("goal_groups").split(",")]
                goal_group = goal_groups[goal_counter]
                if goal_counter == 1:
                    next_goal, node_id = EASY_GOALS[goal_group]
                else:
                    next_goal, node_id = HARD_GOALS[goal_group]
                self.write_message({"EVENT": "NEW_GOAL", "VALUE": next_goal})
                logging.getLogger("chat").info(f"==== NEW DIALOG STARTED FOR USER {self.current_user} ====")
                logging.getLogger("chat").info(f"USER: {self.current_user} || GOAL: {next_goal} || NODE_ID: {node_id}")
                CHAT_ENGINES[self.current_user].reset(node_id)

    def on_close(self):
        logging.getLogger("chat").info(f"Closing connection for user {self.current_user}")
        print(f"Closing connection for user {self.current_user}")

if __name__ == "__main__":
    # on start, check if we have an assignment file, if so, load it and pre-fill group assignments with content
    if os.path.isfile("user_log.txt"):
        with open("user_log.txt", "r") as assignments:
            for line in assignments:
                if "GROUP" in line:
                    user, group = line.split("||")
                    user = user.split(":")[1].strip()
                    group = group.split(":")[1].strip()
                    POLICY_ASSIGNMENT[group].append(user)
                elif "GOAL_INDICES" in line:
                    user, goal_groups = line.split("||")
                    user = user.split(":")[1].strip()
                    goal_groups = [int(group) for group in goal_groups.split(":")[1].strip().split(",")]
                    USER_GOAL_GROUPS_OPEN[goal_groups[0]].append(user)
                    USER_GOAL_GROUPS_EASY[goal_groups[1]].append(user)
                    USER_GOAL_GROUPS_HARD[goal_groups[2]].append(user)


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



    # setup data
    nlu = NLU()
    data = FormattedReimburseGraphDataset('en/reimburse/test_graph.json', 'en/reimburse/test_answers.json', use_answer_synonyms=True, augmentation=DataAugmentationLevel.NONE, resource_dir='resources')
    print("Loading path cache...")
    if not os.path.isfile("path_cache.pkl"):
        print("BUILDING PATH CACHE (THIS SHOULD HAPPEN ONLY ON FIRST STARTUP)")
        path_cache = parallel_path_computation(num_workers=16, data=data)
        with open("path_cache.pkl", "wb") as f:
            pickle.dump(path_cache, f)
    else:
        with open("path_cache.pkl", "rb") as f:
            path_cache = pickle.load(f)
    print("DONE")

    # Add in graphs with different markup for the nodes, based on style (shown to user), but same raw text (seen by system)
    # setup data & parsers
    answerParser = AnswerTemplateParser()
    logicParser = LogicTemplateParser()
    sysParser = SystemTemplateParser()
    valueBackend = ReimbursementRealValueBackend(a1_laender=data.a1_countries, data=data)
    # setup model and encoding
    # Mono-Lingual
    bi_encoder = SentenceTransformer("multi-qa-mpnet-base-dot-v1",
                                    device="cpu", 
                                    cache_folder="./models")
    node_list = get_node_candidate_list_by_type(data=data, node_types=RETRIEVAL_NODE_TYPES)

    print("Loading embedding...")
    if not os.path.isfile("embedding_cache.pt"):
        node_text_embeddings = calculate_node_text_embeddings(bi_encoder=bi_encoder, node_list=node_list)
        torch.save(node_text_embeddings, "embedding_cache.pt")
    else:
        node_text_embeddings = torch.load("embedding_cache.pt")



    # SSL options
    # ssl_ctx = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
    # ssl_ctx.load_cert_chain(certfile="./server/certificate.crt", keyfile="./server/private.key")

    settings = {
        "login_url": "/",
        "cookie_secret": "YOUR_SECRET_KEY",
        "debug": DEBUG,
        "static_path": "./server/templates",
        # "ssl_options": {
        #     "certfile": "./server/certificate.crt",
        #     "keyfile": "./server/private.key",
        # }
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
        (r"/agreed_to_data_collection", UserAgreed),
    ], 
        # ssl_options=ssl_ctx,
        **settings)
    print("created app")
    http_server = tornado.httpserver.HTTPServer(app) #,  ssl_options=ssl_ctx)
    http_server.listen(8081)
    # http_server.listen(443)
    print("set up server address")

    io_loop = tornado.ioloop.IOLoop.current()
    print("created io loop")
    io_loop.start()