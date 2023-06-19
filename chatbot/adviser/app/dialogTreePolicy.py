import json
from pathlib import Path

from django.conf import settings

from chatbot.adviser.app.answerTemplateParser import AnswerTemplateParser
from chatbot.adviser.app.logicParser import LogicTemplateParser
from chatbot.adviser.app.parserValueProvider import RealValueBackend
from chatbot.adviser.app.systemTemplateParser import SystemTemplateParser
from chatbot.adviser.app.useracts import UserActionType
from chatbot.adviser.utils.topics import Topic
from chatbot.adviser.utils.useract import UserAct
from typing import List
from chatbot.adviser.utils.logger import DiasysLogger
from chatbot.adviser.services.service import PublishSubscribe
from chatbot.adviser.services.service import Service

from chatbot.adviser.app.rl.dataset import DialogNode, DialogAnswer
from dashboard.models import Tagegeld


resource_dir = Path('.', 'resources', 'en')

class DialogTreePolicy(Service):
	# State variables
	TURN = 'turn'
	NODE_ID = 'node_id'

	def __init__(self, domain='reisekosten', logger: DiasysLogger = None) -> None:
		super().__init__(domain=domain)
		self.logger = logger
		self.templateParser = SystemTemplateParser()
		self.answerParser = AnswerTemplateParser()
		self.logicParser = LogicTemplateParser()
		with open(resource_dir / "a1_countries.json", "r") as f:
			self.a1_laenderliste = json.load(f)

	def get_first_node(self, version: int) -> DialogNode:
		# find start node and then return its successor
		start_candidates = DialogNode.objects.filter(node_type="startNode", version=version)
		assert len(start_candidates) == 1, f'expected 1 dialog node not to be target of any answer, got {len(start_candidates)}'
		if start_candidates[0].connected_node:
			return start_candidates[0].connected_node
		return None

	def dialog_start(self, user_id: str):
		self.set_state(user_id, DialogTreePolicy.TURN, 0)
		self.set_state(user_id, DialogTreePolicy.NODE_ID, -1)

	def fillTemplate(self, delexicalized_utterance: str, beliefstate: dict):
		return self.templateParser.parse_template(delexicalized_utterance, RealValueBackend(beliefstate, self.a1_laenderliste))

	def fillLogicTemplate(self, delexicalized_utterance: str, beliefstate: dict):
		return self.logicParser.parse_template(delexicalized_utterance, RealValueBackend(beliefstate, self.a1_laenderliste))

	def get_possible_answers(self, node: DialogNode, beliefstate: dict):
		candidates = []
		for answer in node.answers.all().order_by("answer_index"):
			if "{{" in answer.content.text:
				var = self.answerParser.find_variable(answer.content.text)
				if var.name:
					# if answer is template, fill in example values
					example_bst = {}
					if var.name == "LAND" and var.name not in beliefstate:
						for land in ["Deutschland", "USA", "England", "Frankreich", "Spanien", "Italien", "..."] :
							candidates.append(land)
					elif var.name == "STADT" and var.name not in beliefstate and "LAND" in beliefstate:
						# return all known cities for country in beliefstate
						for stadt in [tagegeld.stadt for tagegeld in Tagegeld.objects.filter(land=beliefstate["LAND"]).all() if tagegeld.stadt != "$REST"]:
							candidates.append(stadt)
						candidates.append("Andere Stadt")
					elif var.type == "BOOLEAN":
						# return yes / no as answer option for boolean variable nodes
						candidates += ['ja', 'nein']
			else:
				# no template, just output answer
				if ("[null]" not in answer.content.text) and ("Nutzereingabe festlegen..." not in answer.content.text): # don't show skip node answers
					candidates.append(answer.content.text)
		return candidates

	def _handle_logic_node(self, user_id: int, node: DialogNode, beliefstate: dict):
		# check if we are currently at a logic node
		if isinstance(node, DialogNode) and node.node_type == "logicNode":
			# logic template in node! 
			# Form: "{{ lhs"
			#  -> incomplete, add each answer of form "operator rhs }}" to complete statement
			lhs = node.content.text	
			default_answer = None
			for answer in node.answers.all():
				# check if full statement {{{lhs rhs}}} evaluates to True
				rhs = answer.content.text
				if not "DEFAULT" in rhs: # handle DEFAULT case last!
					if self.fillLogicTemplate(f"{lhs} {rhs}", beliefstate):
						# evaluates to True, follow this path!
						next_node_id = answer.connected_node.pk
						self.set_state(user_id, DialogTreePolicy.NODE_ID, next_node_id)
						return next_node_id, True
				else:
					default_answer = answer
			# default case
			self.set_state(user_id, DialogTreePolicy.NODE_ID, default_answer.connected_node.pk)
			return answer.connected_node.pk, True
		return node, False

	def _handle_info_node(self, user_id: int, node: DialogNode, beliefstate: dict):
		""" Skip user input for nodes with answer [null] """
		# skip [null] anwsers to get to next system output immediately without user input
		sys_utterances = []
		if isinstance(node, DialogNode) and node.node_type == "infoNode":
			sys_utterances.append((self.fillTemplate(node.content.markup, beliefstate), node.node_type))
			node = node.connected_node.pk
			self.set_state(user_id, DialogTreePolicy.NODE_ID, node)
		return node, sys_utterances

	def _handle_var_node(self, user_id: int, node: DialogNode, beliefstate: dict):
		if isinstance(node, DialogNode) and node.node_type == "userInputNode":
			# check if variable is already known
			answer = node.answers.all()[0]
			expected_var = self.answerParser.find_variable(answer.content.text)
			if expected_var.name in beliefstate:
				# variable is alredy knonwn, skip to next node
				next_node_id = answer.connected_node.pk
				self.set_state(user_id, DialogTreePolicy.NODE_ID, next_node_id)
				return next_node_id, [], True
			else:
				# variable is not known, ask
				self.set_state(user_id, DialogTreePolicy.NODE_ID, node.pk)
				return node, [(self.fillTemplate(node.content.text, beliefstate), node.node_type)], True 
		return node, [], False

	@PublishSubscribe(sub_topics=['user_acts', 'beliefstate'], pub_topics=["sys_utterances", "node_id", 'answer_candidates', 'user_acts', 'beliefstate', Topic.DIALOG_END])
	def choose_sys_act(self, user_id: str = "default", version: int = 0, user_acts: List[UserAct] = None, beliefstate: dict = {}) -> dict(sys_utterances=str, node_id=int, answer_candidates=list, user_acts=list,beliefstate=dict):
		turn_count = self.get_state(user_id, DialogTreePolicy.TURN)
		if turn_count == 0:
			# greeting message: find dialog entry node
			node = self.get_first_node(version=version)

			self.set_state(user_id, DialogTreePolicy.TURN, turn_count+1)
			self.set_state(user_id, DialogTreePolicy.NODE_ID, node.pk)

			# if 1st node is info node, return self-call instead of only node content
			node, isInfoNode = self._handle_info_node(user_id=user_id, node=node, beliefstate=beliefstate)
			if isInfoNode:
				self.logger.dialog_turn(f"POLICY {user_id}: TURN {turn_count}, NODE: {node}, TYPE: infoNode, TEXT: {isInfoNode}")
				return {
					"sys_utterances": isInfoNode,
					"node_id": node,
					"answer_candidates": [],
					'user_acts': [],
					'beliefstate': beliefstate
				}
			
			self.logger.dialog_turn(f"POLICY {user_id}: TURN {turn_count}, NODE: {node.pk}, TYPE: {node.node_type}, TEXT: {[(node.content.markup, node.node_type)]}")
			return {
				"sys_utterances": [(node.content.markup, node.node_type)],
				"node_id": node.pk,
				"answer_candidates": self.get_possible_answers(node, beliefstate),
			} 	
		sys_utterances = []
		node = DialogNode.objects.get(pk=self.get_state(user_id, DialogTreePolicy.NODE_ID))
		turn_count += 1
		self.set_state(user_id, DialogTreePolicy.TURN, turn_count)

		# check if we have unrecognized user inputs
		for act in user_acts:
			if act.type == UserActionType.UnrecognizedValue:
				sys_utterances.append((f"{act.value} ist mir leider nicht als {act.slot} bekannt. Wiederholen Sie bitte Fehlerfrei, falls dies nicht funktioniert kontaktieren Sie bitte unsere Mitarbeiter*innen!", "errorMsg"))
			if act.type == UserActionType.TooManyValues:
				sys_utterances.append((f"Für {act.slot} habe ich mehrere Werte erkannt: {act.value}. Bitte geben Sie nur einen Wert ein.", "errorMsg"))
			if len(sys_utterances) > 0:
				# tell users we don't recognize their inputs, stay at same node.
				self.logger.dialog_turn(f"POLICY {user_id}: TURN {turn_count}, NODE: {node.pk}, TYPE: {node.node_type}, TEXT: {sys_utterances}")
				return {
					"sys_utterances": sys_utterances,
					"node_id": node.pk,
					"answer_candidates": self.get_possible_answers(node, beliefstate),
				}

		# Handle User acts
		# dialog is already running and we don't have to handle a logic node, select next answer depending on user acts 
		# (TODO later: depending on beliefstate as well)
		for act in user_acts:
			selected_answer: DialogAnswer = node.answers.filter(content__text=act.text)[0]
			node = selected_answer.connected_node
			self.set_state(user_id, DialogTreePolicy.NODE_ID, node.pk)
			if node.node_type not in ['infoNode', 'logicNode', 'userInputNode']:
				sys_utterances.append((self.fillTemplate(node.content.markup, beliefstate), node.node_type))
			# else:
				# print('UNK USER ACT', act.type)
		
		# Handle Logic nodes
		node, isLogicTemplate = self._handle_logic_node(user_id=user_id, node=node, beliefstate=beliefstate)
		if isLogicTemplate:
			if sys_utterances:
				self.logger.dialog_turn(f"POLICY {user_id}: TURN {turn_count}, NODE: {node.pk}, TYPE: {node.node_type}, TEXT: {node.content.text}")
				return {
					"sys_utterances": sys_utterances,
					"node_id": node,
					"answer_candidates": [],
					'user_acts': [],
					'beliefstate': beliefstate
			}
			return {
				'user_acts': [],
				'beliefstate': beliefstate,
				"node_id": node	
			}

		node, isInfoNode = self._handle_info_node(user_id=user_id, node=node, beliefstate=beliefstate)
		if isInfoNode:
			sys_utterances += isInfoNode
			self.logger.dialog_turn(f"POLICY {user_id}: TURN {turn_count}, NODE: {node}, TYPE: infoNode, TEXT: {sys_utterances}")
			return {
				"sys_utterances": sys_utterances,
				"node_id": node,
				"answer_candidates": [],
				'user_acts': [],
				'beliefstate': beliefstate
			}

		node, varUtterances, isVarNode = self._handle_var_node(user_id=user_id, node=node, beliefstate=beliefstate)
		if isVarNode:
			sys_utterances += varUtterances
			if len(varUtterances) > 0:
				# value unkown, don't skip user input
				self.logger.dialog_turn(f"POLICY {user_id}: TURN {turn_count}, NODE: {node.pk}, TYPE: {node.node_type}, TEXT: {sys_utterances}")
				return {
					"sys_utterances": sys_utterances,
					"node_id": node.pk,
					"answer_candidates": self.get_possible_answers(node, beliefstate),
				}
			else:
				# value alredy known skip user input
				self.logger.dialog_turn(f"POLICY {user_id}: TURN {turn_count}, NODE: {node}, TYPE: varNode, TEXT: {sys_utterances}")
				return {
					"sys_utterances": sys_utterances,
					"node_id": node,
					"answer_candidates": [],
					'user_acts': [],
					'beliefstate': beliefstate
				}

		# normal template, fill with values from beliefstate
		# sys_utterances.append(self.fillTemplate(node.content.markup, beliefstate))
		# if self.logger:
		# 	self.logger.dialog_turn(f'Policy: user selected answer {selected_answer.content.text}')
		# 	self.logger.dialog_turn(f'Policy: transitioning to next node {node.pk}')

		# Handle self-calls
		if len(user_acts) == 0:
			if not isinstance(node, DialogNode):
				node = DialogNode.objects.get(pk=node)
			sys_utterances += [(self.fillTemplate(node.content.markup, beliefstate), node.node_type)]

		self.logger.dialog_turn(f"POLICY {user_id}: TURN {turn_count}, NODE: {node.pk}, TYPE: {node.node_type}, TEXT: {sys_utterances}")
		# for sysutt in sys_utterances:
		# 	if "vielen dank" in sysutt[0].lower():
		# 		return {
		# 			f'{Topic.DIALOG_END}': True
		# 		}
		return {
			"sys_utterances": sys_utterances,
			"node_id": node.pk,
			"answer_candidates": self.get_possible_answers(node, beliefstate),
		}
	
