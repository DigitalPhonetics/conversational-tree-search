import hashlib
import logging
import multiprocessing
from copy import deepcopy
import traceback

import tornado
import tornado.httpserver
from tornado.web import Application

from data.dataset import DataAugmentationLevel, ReimburseGraphDataset
from server.handlers import AuthenticatedWebSocketHandler, BaseHandler, LogPostSurvey, LogPreSurvey, LoginHandler, PilotDataAgreement, PilotLogPreferences, PilotStylePreference, PilotTextAnalysis, PreSurvey, ThankYou, KnownEntry
from tornado.web import RequestHandler
import numpy as np





survey_logger = logging.getLogger("survey")
survey_logger.setLevel(logging.INFO)
survey_log_file_handler = logging.FileHandler("pilot_log.txt")
survey_log_file_handler.setLevel(logging.INFO)
survey_logger.addHandler(survey_log_file_handler)



multiprocessing.set_start_method("spawn")


class CheckLogin(RequestHandler):
    def post(self):
        global NODE_ID
        username = self.get_body_argument("username").encode()
        h = hashlib.shake_256(username)
        user_id = h.hexdigest(15)
        # NOTE: disallow users who already participated in other versions of this task via Prolific UI
        # if user_id in known_users:
        #     self.redirect("/known_entry")
        # else:
        self.set_secure_cookie("user", user_id)
        self.redirect("/data_agreement")


class UserAgreed(BaseHandler):
    def post(self):
        global NODE_ID
        logging.getLogger("survey").info(f"USER: {self.current_user} || AGREED: True")
        self.redirect(f"/style_preference")


if __name__ == "__main__":
    settings = {
        "login_url": "/",
        "cookie_secret": "YOUR_SECRET_KEY",
        "debug": False,
        "static_path": "./server/templates"
    }
    print("settings created")
    app = Application([
        (r"/", LoginHandler),
        (r"/check_login", CheckLogin),
        (r"/style_preference", PilotStylePreference),
        (r"/chat", PilotTextAnalysis),
        (r"/log_pre_survey", LogPreSurvey),
        (r"/data_agreement", PilotDataAgreement),
        (r"/log_preferences", PilotLogPreferences),
        (r"/thank_you", ThankYou),
        (r"/known_entry", KnownEntry),
        (r"/agreed_to_data_collection", UserAgreed)
    ], **settings)
    print("created app")
    http_server = tornado.httpserver.HTTPServer(app) #, ssl_options = ssl_ctx)
    http_server.listen(8081)
    print("set up server address")

    io_loop = tornado.ioloop.IOLoop.current()
    print("created io loop")
    io_loop.start()