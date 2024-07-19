import hashlib
import logging
import numpy as np
import tornado
from tornado.web import RequestHandler
from tornado.websocket import WebSocketHandler

from data.dataset import DataAugmentationLevel, ReimburseGraphDataset

NODE_ID = 16363755463439219

NODE_IDS = [
    16365521324065600,
    16460328708250870,
    16378349334755637,
    16370483534787100,
    16363755463439219,
    16363834594338823,
    16384329210117153,
    16457053159041482,
    16365525829145685,
    16387868859695624]
COMPLETION_LINK = "www.to_change.com"

DISTRIBUTION = {id: [] for id in NODE_IDS}


class BaseHandler(RequestHandler):
    def get_current_user(self):
        return tornado.escape.to_unicode(self.get_secure_cookie("user"))    


class LoginHandler(BaseHandler):
    def get(self):
        if self.current_user:
            # if the user has done both dialogs already, they should not be able to chat again
            if self.get_cookie("goal_counter") and int(self.get_cookie("goal_counter")) >= 3:
                self.redirect('/thank_you')
            else:
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
        goal_counter = self.get_cookie("goal_counter")
        if not goal_counter:
            # first visit to chat index: initialize first goal
            goal_counter = 0
            self.set_cookie("goal_counter", str(goal_counter))
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
        print(self.current_user)
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
        global COMPLETION_LINK
        self.render("./templates/thank_you.html", completion_key=COMPLETION_LINK)


class AuthenticatedWebSocketHandler(WebSocketHandler):
    def get_current_user(self):
        return tornado.escape.to_unicode(self.get_secure_cookie("user"))
    
class PilotDataAgreement(BaseHandler):
    @tornado.web.authenticated
    def get(self):
        print(self.current_user)
        self.render("./templates/pilot_data_agreement.html")

class PilotTextAnalysis(BaseHandler):
    @tornado.web.authenticated
    def get(self):
        order = np.random.permutation(4)
        conditions = ["FRIENDLY", "PERSONAL", "BASE", "FORMAL"]
        user_conditions = [conditions[i-1] for i in order]
        logging.getLogger("survey").info(f"USER: {self.current_user} || ORDER: {user_conditions}")
        user_texts = self.get_node_texts(user_conditions=user_conditions)        
        self.render("./templates/text_analysis.html", template1=user_texts[0], template2=user_texts[1], template3=user_texts[2], template4=user_texts[3])

    def get_node_texts(self, user_conditions):
        node_texts = []
        for condition in user_conditions:
            if condition == "BASE":
                graph = ReimburseGraphDataset('en/reimburse/test_graph.json', 'en/reimburse/test_answers.json', False, augmentation=DataAugmentationLevel.NONE)
            elif condition == "FORMAL":
                graph = ReimburseGraphDataset('en/reimburse/linguistic_variations/formal_graph.json', 'en/reimburse/test_answers.json', False, augmentation=DataAugmentationLevel.NONE)
            elif condition == "PERSONAL":
                graph = ReimburseGraphDataset('en/reimburse/linguistic_variations/personal_graph.json', 'en/reimburse/test_answers.json', False, augmentation=DataAugmentationLevel.NONE)
            elif condition == "FRIENDLY":
                graph = ReimburseGraphDataset('en/reimburse/linguistic_variations/friendly_graph.json', 'en/reimburse/test_answers.json', False, augmentation=DataAugmentationLevel.NONE)
            node_id = self.get_node_assignment()
            logging.getLogger("survey").info(f"USER: {self.current_user} || NODE_ID: {node_id}")
            node = graph.nodes_by_key[node_id].text
            if node.startswith("In"):
                node = "In Japan" + node[16:]
            node_texts.append(node)
        return node_texts

    def get_node_assignment(self):
        global DISTRIBUTION
        global NODE_ID
        user = self.current_user
        next_node = sorted(DISTRIBUTION.items(), key=lambda item: item[1])[0][0]
        # return next_node
        return NODE_ID

class PilotStylePreference(BaseHandler):
    @tornado.web.authenticated
    def get(self):
        self.render("./templates/style_preference.html")

class PilotLogPreferences(BaseHandler):
    @tornado.web.authenticated
    def post(self):
        results = self.request.body_arguments
        results = {key : str(results[key][0])[2:-1] for key in results}
        logging.getLogger("survey").info(f"USER: {self.current_user} || Preferences: {results}")
        self.redirect(f"/thank_you")
        

