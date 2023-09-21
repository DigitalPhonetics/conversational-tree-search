import asyncio
import logging
import hashlib
from multiprocessing import freeze_support
import json
from typing import Dict, Union

import tornado
import tornado.httpserver
from tornado.web import RequestHandler, Application
from tornado.websocket import WebSocketHandler

# from services.service import DialogSystem, Service, PublishSubscribe

import os
import base64

import multiprocessing

multiprocessing.set_start_method("spawn")

chat_logger = logging.getLogger("chat")
chat_logger.setLevel(logging.INFO)
chat_log_file_handler = logging.FileHandler("chat.txt")
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

DEBUG = True
NUM_CONDITIONS = 1
GROUP_ASSIGNMENTS = {"hdc": [], "faq": [], "cts": []}

# on start, check if we have an assignment file, if so, load it and pre-fill group assignments with content
with open("user_log.txt", "rt") as assignments:
    for line in assignments:
        if "GROUP" in line:
            user, group = line.split("||")
            user = user.split(":")[1].strip()
            group = group.split(":")[1].strip()
            GROUP_ASSIGNMENTS[group].append(user)

# class GUIInterface(Service):
#     def __init__(self, domain = None, logger =  None):
#         Service.__init__(self, domain=domain)
#         self.logger = logger
#         self.loopy_loop = asyncio.new_event_loop()

#     @PublishSubscribe(sub_topics=['sys_utterance'])
#     def send_sys_output(self, user_id: str = "default", sys_utterance: str = ""):
#         asyncio.set_event_loop(self.loopy_loop)
#         print("SYS MSG:", sys_utterance)
#         tts.read_aloud(sys_utterance)
#         if self.get_state(user_id, 'socket'):
#             logging.getLogger("chat").info(f"MSG SYSTEM ({user_id}): {sys_utterance}")
#             self.get_state(user_id, 'socket').write_message({"EVENT": "MSG", "VALUE": sys_utterance})
#         else:
#             print("NOT CONNECTED")

#     # TODO: update once we have speech and not text
#     @PublishSubscribe(pub_topics=["user_utterance"])
#     def forward_to_dialog_system(self, user_id, user_utterance):
#         print("USER", user_id)
#         print("PUBLISH TO DIALOG SYSTEM")
#         return {"user_utterance": user_utterance}
    

# gui_interface = GUIInterface(domain=domain)

# TODO initialise three policies as separate domains
# ds = DialogSystem([gui_interface, policy_hdc, policy_faq, policy_cts])

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
            self.redirect("/data_agreement")
        else:
            self.render("./templates/login.html")


class CheckLogin(RequestHandler):
    def post(self):
        username = self.get_body_argument("username").encode()
        h = hashlib.shake_256(username)
        self.set_secure_cookie("user", h.hexdigest(15))
        if not self.get_cookie("seen_conditions"):
            self.set_cookie("seen_conditions", "0")
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
        global NUM_CONDITIONS
        seen_conditions = int(self.get_cookie('seen_conditions')) + 1
        self.set_cookie("seen_conditions", str(seen_conditions))
        self.render("./templates/chat.html", total_conditions=NUM_CONDITIONS)

class DataAgreement(BaseHandler):
    @tornado.web.authenticated
    def get(self):
        self.render("./templates/data_agreement.html")

class PostSurvey(BaseHandler):
    @tornado.web.authenticated
    def get(self):
        self.render("./templates/post_survey.html")

class PreSurvey(BaseHandler):
    @tornado.web.authenticated
    def get(self):
        self.render("./templates/pre_survey.html")

class ThankYou(BaseHandler):
    @tornado.web.authenticated
    def get(self):
        self.render("./templates/thank_you.html", completion_key=self.current_user)


# class UserChatSocket(AuthenticatedWebSocketHandler):
#     def open(self):
#         global gui_interface
#         global ds
#         print(f"Opened socket for user: {self.current_user}")
#         gui_interface.set_state(user_id=self.current_user, attribute_name='socket', attribute_value=self)
#         ds._start_dialog(start_signals={"user_utterance/GasStations": ""}, user_id=self.current_user)
#         print(f"starting dialog system for user {self.current_user}")
#         logging.getLogger("chat").info(f"==== NEW DIALOG STARTED FOR USER {self.current_user} ====")

#     def on_message(self, message):
#         global gui_interface
#         data = tornado.escape.json_decode(message)
#         event = data["EVENT"]
#         value = data["VALUE"]
#         if event == "MSG":
#             # TODO: might need to update for audio files
#             print(f"MSG for user {self.current_user}: {message}")
#             logging.getLogger("chat").info(f"MSG USER ({self.current_user}): {value}")
#             gui_interface.forward_to_dialog_system(self.current_user, value)
#         if event == "END_DIALOG":
#             logging.getLogger("chat").info(f"USER ({self.current_user} FINISHED DIALOG)")
#         if event == "SPEECH":
#             value = base64.b64decode(value)
#             with open("tmp/speech.wav", "wb") as outfile:
#                 outfile.write(value)
#             transcription = model.transcribe(['tmp/speech.wav'])[0]['transcription']
#             os.remove("tmp/speech.wav")
#             logging.getLogger("chat").info(f"MSG USER ({self.current_user}): {transcription}")
#             gui_interface.forward_to_dialog_system(self.current_user, transcription)
#             self.write_message({"EVENT": "USER_MSG", "VALUE": transcription})
       
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
        # (r"/channel", UserChatSocket),
        (r"/log_pre_survey", LogPreSurvey),
        (r"/log_post_survey", LogPostSurvey),
        (r"/data_agreement", DataAgreement),
        (r"/thank_you", ThankYou),
        (r"/agreed_to_data_collection", UserAgreed)
    ], **settings)
    print("created app")
    http_server = tornado.httpserver.HTTPServer(app) #, ssl_options = ssl_ctx)
    http_server.listen(8000)
    print("set up server address")

    io_loop = tornado.ioloop.IOLoop.current()
    print("created io loop")
    io_loop.start()