import asyncio
import logging
from multiprocessing import freeze_support
import time
from typing import Dict, Union

import tornado
import tornado.httpserver
from tornado.web import RequestHandler, Application
from tornado.websocket import WebSocketHandler

from utils.common import Language
from services.policy import GasstationsPolicy
from services.adaptivity.adaptivity import Adaptivity
from services.nlu.nlu import HandcraftedNLU
from services.bst.bst import HandcraftedBST
from services.policy.policy_handcrafted import HandcraftedPolicy
from services.nlg.nlg import HandcraftedNLG
from services.service import DialogSystem, Service, PublishSubscribe
from utils.domain.jsonlookupdomain import JSONLookupDomain
from huggingsound import SpeechRecognitionModel
import os
import base64
from resources.tts.InferenceInterfaces.ToucanTTSInterface import ToucanTTSInterface

import multiprocessing

multiprocessing.set_start_method("spawn")

print("initializing asr model")
model = SpeechRecognitionModel("jonatasgrosman/wav2vec2-large-xlsr-53-german")
print("asr model loaded")

logger = logging.getLogger("chat")
logger.setLevel(logging.INFO)
log_file_handler = logging.FileHandler("chat_log.txt")
log_file_handler.setLevel(logging.INFO)
logger.addHandler(log_file_handler)

DEBUG = True
NUM_CONDITIONS = 2

CURRENT_USER_CONNECTION = None 

domain = JSONLookupDomain(name="GasStations")


class GUIInterface(Service):
    def __init__(self, domain = None, logger =  None):
        Service.__init__(self, domain=domain)
        self.logger = logger
        self.loopy_loop = asyncio.new_event_loop()

    @PublishSubscribe(sub_topics=['sys_utterance'])
    def send_sys_output(self, user_id: str = "default", sys_utterance: str = ""):
        asyncio.set_event_loop(self.loopy_loop)
        print("SYS MSG:", sys_utterance)
        tts.read_aloud(sys_utterance)
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
    

gui_interface = GUIInterface(domain=domain)
nlu = HandcraftedNLU(domain=domain, language=Language.GERMAN)
bst = HandcraftedBST(domain=domain)
policy = GasstationsPolicy(domain=domain)
nlg = HandcraftedNLG(domain=domain)
adaptivity = Adaptivity(domain=domain)

print("Initializing TTS Model")
tts = ToucanTTSInterface(
    device="cpu", 
    language="de")
print("TTS initialized")

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
        if event == "SPEECH":
            value = base64.b64decode(value)
            with open("tmp/speech.wav", "wb") as outfile:
                outfile.write(value)
            transcription = model.transcribe(['tmp/speech.wav'])[0]['transcription']
            os.remove("tmp/speech.wav")
            logging.getLogger("chat").info(f"MSG USER ({self.current_user}): {transcription}")
            gui_interface.forward_to_dialog_system(self.current_user, transcription)
            self.write_message({"EVENT": "USER_MSG", "VALUE": transcription})
       
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
        (r"/survey", Survey),
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

