import hashlib
import logging
import tornado
from tornado.web import RequestHandler
from tornado.websocket import WebSocketHandler


class BaseHandler(RequestHandler):
    def get_current_user(self):
        return tornado.escape.to_unicode(self.get_secure_cookie("user"))    


class LoginHandler(BaseHandler):
    def get(self):
        if self.current_user:
            self.redirect("/data_agreement")
        else:
            self.render("./templates/login.html")

class LogPostSurvey(BaseHandler):
    def post(self):
        results = self.request.body_arguments
        results = {key : str(results[key][0])[2:-1] for key in results}
        logging.getLogger("survey").info(f"USER: {self.current_user} || POST-SURVEY: {results}")
        self.redirect(f"/thank_you")


class ChatIndex(BaseHandler):
    @tornado.web.authenticated
    def get(self):
        self.render("./templates/chat.html")


class LogPreSurvey(BaseHandler):
    @tornado.web.authenticated
    def post(self):
        results = self.request.body_arguments
        results = {key : str(results[key][0])[2:-1] for key in results}
        logging.getLogger("survey").info(f"USER: {self.current_user} || PRE-SURVEY: {results}")
        self.redirect(f"/chat")


class DataAgreement(BaseHandler):
    @tornado.web.authenticated
    def get(self):
        self.render("./templates/data_agreement.html")

class KnownEntry(BaseHandler):
    def get(self):
        self.render("./templates/known_entry.html")


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


class AuthenticatedWebSocketHandler(WebSocketHandler):
    def get_current_user(self):
        return tornado.escape.to_unicode(self.get_secure_cookie("user"))

