from chatbot.adviser.services.service import Service, PublishSubscribe
from chatbot.adviser.utils.logger import DiasysLogger

class TopicTracker(Service):

	def __init__(self, domain='reisekosten', logger: DiasysLogger = None) -> None:
		super().__init__(domain=domain)
		self.logger = logger


	@PublishSubscribe(sub_topics=["gen_user_utterance"], pub_topics=["user_utterance"])
	def extract_user_acts(self, user_id: str = "default", version: int = 0, gen_user_utterance: str = None) -> dict(user_utterance=str):
		return {
			f'user_utterance': gen_user_utterance
		}
