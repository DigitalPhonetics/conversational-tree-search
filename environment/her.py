import random
from typing import Tuple
from environment.goal import DummyGoal

from utils.utils import EnvInfo

from data.dataset import GraphDataset

from data.parsers.answerTemplateParser import AnswerTemplateParser
from data.parsers.logicParser import LogicTemplateParser
from data.parsers.systemTemplateParser import SystemTemplateParser
from data.parsers.parserValueProvider import RealValueBackend
from utils.utils import AutoSkipMode
from environment.free import FreeEnvironment
from environment.guided import GuidedEnvironment
import config as cfg


class CTSHEREnvironment:
    """
    Extra environment for HER Replay (very similar to CTSEnvironment, but does not mess up the statistics there)
    """
    def __init__(self, 
                # env_id: int,
                dataset: GraphDataset,
                auto_skip: AutoSkipMode,
                normalize_rewards: bool,
                max_steps: int,
                user_patience: int,
                stop_when_reaching_goal: bool,
                stop_on_invalid_skip: bool,
                sys_token: str, usr_token: str, sep_token: str,
                **kwargs):
        # self.env_id = env_id
        self.data = dataset
        self.sys_token = sys_token
        self.usr_token = usr_token
        self.sep_token = sep_token

        self.max_reward = 4 * dataset.get_max_tree_depth() if normalize_rewards else 1.0
        self.max_distance = dataset.get_max_tree_depth() + 1
        cfg.INSTANCES[cfg.InstanceArgs.MAX_DISTANCE] = self.max_distance

        # text parsers
        answer_parser = AnswerTemplateParser()
        logic_parser = LogicTemplateParser()
        system_parser = SystemTemplateParser()
        value_backend = RealValueBackend(dataset.a1_countries, dataset)

        # initialize task-specific environments
        self.guided_env = GuidedEnvironment(dataset=dataset,
                sys_token=sys_token, usr_token=usr_token, sep_token=sep_token,
                max_steps=max_steps, max_reward=self.max_reward, user_patience=user_patience,
                stop_when_reaching_goal=stop_when_reaching_goal, stop_on_invalid_skip=stop_on_invalid_skip,
                answer_parser=answer_parser, system_parser=system_parser, logic_parser=logic_parser,
                value_backend=value_backend,
                auto_skip=auto_skip)
        self.free_env = FreeEnvironment(dataset=dataset,
                sys_token=sys_token, usr_token=usr_token, sep_token=sep_token,
                max_steps=max_steps, max_reward=self.max_reward, user_patience=user_patience,
                stop_when_reaching_goal=stop_when_reaching_goal, stop_on_invalid_skip=stop_on_invalid_skip,
                answer_parser=answer_parser, system_parser=system_parser, logic_parser=logic_parser, 
                value_backend=value_backend,
                auto_skip=auto_skip)

        print("HER ENV!!", "TOKENS:", sys_token, usr_token, sep_token)
    
    @property
    def current_episode(self):
        return self.guided_env.current_episode + self.free_env.current_episode

    def reset(self, mode: str, replayed_goal: DummyGoal): 
        """
        Args:
            mode: "free" or "guided"
        """
        assert mode in ["free", "guided"]
        if mode == "free":
            self.active_env = self.free_env
        elif mode == "guided":
            self.active_env = self.guided_env

        # choose uniformely at random between guided and free env according to ratio
        self.active_env.episode_log = []
        return self.active_env.reset(current_episode=self.current_episode, max_distance=self.max_distance, replayed_goal=replayed_goal)

    def step(self, action: int, replayed_user_utterance: Tuple[str, None] = None) -> Tuple[dict, float, bool, dict]:
        obs, reward, done = self.active_env.step(action, replayed_user_utterance)
        obs[EnvInfo.IS_FAQ] = self.active_env == self.free_env
        obs["is_success"] = obs[EnvInfo.ASKED_GOAL]
        return obs, reward, done, obs # info = obs before encoding
    