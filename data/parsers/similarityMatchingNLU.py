from data.parsers.answerTemplateParser import AnswerTemplateParser
from encoding.text.text import SentenceEmbeddings
from dashboard.models import Tagegeld
from chatbot.adviser.app.rl.dataset import DialogNode
from typing import List

import torch
from torch.nn.functional import cosine_similarity

from chatbot.adviser.services.service import Service, PublishSubscribe
from chatbot.adviser.app.useracts import UserActionType, UserAct
from chatbot.adviser.utils.logger import DiasysLogger

from chatbot.adviser.app.nlu import NLU

class SimilarityMatchingNLU(Service):
	SIMILARITY_THRESHOLD = 0.01 # TODO find acceptable threshold

	def __init__(self, embeddings: SentenceEmbeddings, domain='reisekosten', logger: DiasysLogger = None) -> None:
		super().__init__(domain=domain)
		self.embeddings = embeddings
		self.logger = logger
		self.templateParser = AnswerTemplateParser()
		self.nlu = NLU()

	def match_help(self, utterance: str) -> bool:
		if 'hilfe' in utterance or 'helfen':
			return True
		return False

	def dialog_start(self, user_id: str):
		self.set_state(user_id, "beliefstate", {})
		self.set_state(user_id, "turn", 0)

	def _extract_location(self, user_id: int, user_utterance: str, beliefstate: dict):
		acts = []
		locations = self.nlu.extract_places(user_utterance)
		# Check that user did not mention more than one location in this turn.
		# If slot is empty, remove it (makes checking slot validity easier later)
		locations = {slot: locations[slot] for slot in locations if len(locations[slot]) > 0}
		for slot in locations:
			if len(locations[slot]) > 1:
				acts.append(UserAct(act_type=UserActionType.TooManyValues, slot=slot, value=f'"{", ".join(locations[slot])}"'))
		# Check that mentioned locations are valid. If so, add to beliefstate
		for slot in locations: 
			location_value = locations[slot][0]
			if slot == "LAND":
				# check that country is known to our database
				if Tagegeld.objects.filter(land__iexact=location_value).exists():
					beliefstate[slot] = Tagegeld.objects.filter(land__iexact=location_value).first().land
					self.set_state(user_id, "beliefstate", beliefstate)
				else:
					acts.append(UserAct(act_type=UserActionType.UnrecognizedValue, slot=slot, value=location_value))
			elif slot == "STADT":
				# check that the city is (hopefully) not an emty expression
				if len(location_value) >= 2:
					if Tagegeld.objects.filter(stadt__iexact=location_value).exists():
						beliefstate[slot] = location_value
						self.set_state(user_id, "beliefstate", beliefstate)
					else:
						beliefstate[slot] = "$REST" # city unkown -> no special rules
						self.set_state(user_id, "beliefstate", beliefstate)
				else:
					acts.append(UserAct(act_type=UserActionType.UnrecognizedValue, slot=slot, value=location_value))
		return acts

	def _extract_time(self, user_id: int, user_utterance: str, beliefstate: dict):
		times = self.nlu.extract_time(user_utterance)
		return []
		# TODO handle times in BST


	def _fill_variable(self, user_id: int, answer_template: str, user_utterance: str, beliefstate: dict, node: DialogNode):
		acts = []
		expected_var = self.templateParser.find_variable(answer_template)
		# print(" - requested filling variable", expected_var.name, expected_var.type)
		if expected_var.name and expected_var.type:
			if len(user_utterance.strip()) == 0:
				acts.append(UserAct(act_type=UserActionType.UnrecognizedValue, slot=expected_var.name, value=""))
				return acts
		if expected_var.type == "TEXT":
			if len(user_utterance.split()) > 1:
				acts.append(UserAct(act_type=UserActionType.TooManyValues, slot=expected_var.name, value=f'"{", ".join(user_utterance.split())}"'))
			else:
				# set variable
				beliefstate[expected_var.name] = user_utterance
		elif expected_var.type == "NUMBER":
			if len(user_utterance.split()) > 1:
				acts.append(UserAct(act_type=UserActionType.TooManyValues, slot=expected_var.name, value=f'"{", ".join(user_utterance.split())}"'))
			else:
				try:
					# set variable
					beliefstate[expected_var.name] = float(user_utterance.strip())
				except:
					acts.append(UserAct(act_type=UserActionType.UnrecognizedValue, slot=expected_var.name, value=user_utterance.strip()))
		elif expected_var.type == "BOOLEAN":
			utterance_emb = self.embeddings.encode(user_utterance) # 1 x 512
			answers = ["ja", "nein"]
			answer_embs = torch.cat([self.embeddings.encode(answer) for answer in answers], axis=0) # 2 x 512
			similarities = cosine_similarity(utterance_emb, answer_embs, -1)
			most_similar_answer_idx = similarities.argmax(-1).item()
			max_similarity_score = similarities[most_similar_answer_idx] # top answer score
			if max_similarity_score >= self.SIMILARITY_THRESHOLD:
				beliefstate[expected_var.name] = True if most_similar_answer_idx == 0 else False
			else:
				acts.append(UserAct(act_type=UserActionType.UnrecognizedValue, slot=expected_var.name, value=user_utterance.strip()))
		elif expected_var.type == "TIMEPOINT":
			times = self.nlu.extract_time(user_utterance)
			if times:
				times = times['time_points']
			if len(times) == 0:
				acts.append(UserAct(act_type=UserActionType.UnrecognizedValue, slot=expected_var.name, value=user_utterance.strip()))
			else:
				beliefstate[expected_var.name] = times
				# TODO extract only one type of time from dict
		elif expected_var.type == "TIMESPAN":
			times = self.nlu.extract_time(user_utterance)
			if times:
				times = times['time_spans']
			if len(times) == 0:
				acts.append(UserAct(act_type=UserActionType.UnrecognizedValue, slot=expected_var.name, value=user_utterance.strip()))
			else:
				beliefstate[expected_var.name] = times[-1]
				# TODO extract only one type of time from dict
		elif expected_var.type == "LOCATION":
			locations = self.nlu.extract_places(user_utterance)
			# Check that user did not mention more than one location in this turn.
			# If slot is empty, remove it (makes checking slot validity easier later)
			locations = {slot: locations[slot] for slot in locations if len(locations[slot]) > 0}
			for slot in locations:
				if len(locations[slot]) > 1:
					acts.append(UserAct(act_type=UserActionType.TooManyValues, slot=slot, value=f'"{", ".join(locations[slot])}"'))
			# Check that mentioned locations are valid. If so, add to beliefstate
			if expected_var.name == "LAND" and not locations:
				# only check for invalid countries; for cities, we don't have an exhaustive list
				acts.append(UserAct(act_type=UserActionType.UnrecognizedValue, slot=expected_var.name, value=user_utterance))
			for slot in locations:
				location_value = locations[slot][0]
				if slot == "LAND":
					# check that country is known to our database
					if Tagegeld.objects.filter(land__iexact=location_value).exists():
						beliefstate[expected_var.name] = Tagegeld.objects.filter(land__iexact=location_value).first().land
						self.set_state(user_id, "beliefstate", beliefstate)
					else:
						acts.append(UserAct(act_type=UserActionType.UnrecognizedValue, slot=slot, value=location_value))
				elif slot == "STADT":
					# check that the city is (hopefully) not an emty expression
					if Tagegeld.objects.filter(stadt__iexact=location_value).exists():
						beliefstate[expected_var.name] = location_value
						self.set_state(user_id, "beliefstate", beliefstate)
					else:
						beliefstate[expected_var.name] = "$REST" # city unkown -> no special rules
						self.set_state(user_id, "beliefstate", beliefstate)
			if expected_var.name == "STADT" and "STADT" not in locations:
				# City that we don't know from table
				location_value = user_utterance.strip()
				if len(location_value) >= 2:
					beliefstate[expected_var.name] = "$REST" # city unkown -> no special rules
					self.set_state(user_id, "beliefstate", beliefstate)
				else:
					acts.append(UserAct(act_type=UserActionType.UnrecognizedValue, slot=slot, value=location_value))

		return acts
		


	@PublishSubscribe(sub_topics=["user_utterance", "node_id"], pub_topics=["user_acts", "beliefstate"])
	def extract_user_acts(self, user_id: str = "default", version: int = 0, user_utterance: str = None, node_id: int = -1) -> dict(user_acts=List[UserAct]):
		acts = []
		beliefstate = self.get_state(user_id, "beliefstate")
		current_node = DialogNode.objects.get(pk=node_id)
		turn = self.get_state(user_id, "turn")
		self.set_state(user_id, "turn", turn + 1)

		if turn == 0:
			# handle first turn: check if user mentioned city or country directly
			acts += self._extract_location(user_id, user_utterance, beliefstate)
			acts += self._extract_time(user_id, user_utterance, beliefstate)

		# check if this turn should fill a variable for BST
		user_answer_options = current_node.answers.all().order_by("answer_index")
		if current_node.node_type == "userInputNode":
			acts += self._fill_variable(user_id, user_answer_options[0].content.text, user_utterance, beliefstate, current_node)

		self.logger.dialog_turn(f"BST {user_id}: {beliefstate}")
		if len(acts) > 0:
			self.logger.dialog_turn(f"NLU {user_id}: {acts}")
			return {
				'user_acts': acts,
				'beliefstate': beliefstate
			}
		
		# match user utterance against possible answers
		utterance_emb = self.embeddings.encode(user_utterance) # 1 x 512
		answer_embs = self.embeddings.embed_node_answers(current_node).squeeze(0) # 1 x answers x 512 -> answers x 512
		similarities = cosine_similarity(utterance_emb, answer_embs, -1)
		most_similar_answer_idx = similarities.argmax(-1).item()
		max_similarity_score = similarities[most_similar_answer_idx] # top answer score
		if max_similarity_score >= self.SIMILARITY_THRESHOLD:
			# found acceptable answer, return top answer
			acts.append(UserAct(text=current_node.answers.all().order_by("answer_index")[most_similar_answer_idx].content.text, act_type=UserActionType.NormalUtterance))
			if self.logger:
				self.logger.dialog_turn(f'NLU {user_id}: Most similar answer chosen {current_node.answers.all().order_by("answer_index")[most_similar_answer_idx].content.text} with score {max_similarity_score}')
		
		self.logger.dialog_turn(f"NLU {user_id}: {acts}")
		return {
			'user_acts': acts,
			'beliefstate': beliefstate
		}
