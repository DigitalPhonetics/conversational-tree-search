import random
from typing import Tuple

import gym
from chatbot.adviser.app.rl.utils import EnvInfo

from data.dataset import GraphDataset
from data.cache import Cache

from chatbot.adviser.app.answerTemplateParser import AnswerTemplateParser
from chatbot.adviser.app.logicParser import LogicTemplateParser
from chatbot.adviser.app.systemTemplateParser import SystemTemplateParser
from chatbot.adviser.app.parserValueProvider import RealValueBackend
from chatbot.adviser.app.rl.utils import AutoSkipMode
from encoding.state import StateEncoding
from environment.free import FreeEnvironment
from environment.guided import GuidedEnvironment


class CTSEnvironment(gym.Env):
    def __init__(self, env_id: int, mode: str,
                cache: Cache,
                dataset: GraphDataset,
                state_encoding: StateEncoding,
                guided_free_ratio: float,
                auto_skip: AutoSkipMode,
                normalize_rewards: bool,
                max_steps: int,
                user_patience: int,
                stop_when_reaching_goal: bool,
                num_train_envs: int,
                num_val_envs: int,
                num_test_envs: int,
                sys_token: str, usr_token: str, sep_token: str):
        self.env_id = env_id
        print("ENV!!", mode, "TOKENS:", sys_token, usr_token, sep_token)

        answer_parser = AnswerTemplateParser()
        logic_parser = LogicTemplateParser()
        system_parser = SystemTemplateParser()
        value_backend = RealValueBackend(dataset.a1_countries)

        self.guided_free_ratio = guided_free_ratio
        self.max_reward = 4 * dataset.get_max_tree_depth() if normalize_rewards else 1.0
        
        
        if guided_free_ratio > 0.0:
            self.guided_env = GuidedEnvironment(env_id=env_id, cache=cache, dataset=dataset, state_encoding=state_encoding,
                sys_token=sys_token, usr_token=usr_token, sep_token=sep_token,
                max_steps=max_steps, max_reward=self.max_reward, user_patience=user_patience,
                answer_parser=answer_parser, logic_parser=logic_parser,
                value_backend=value_backend,
                auto_skip=auto_skip)
        if guided_free_ratio < 1.0:
            self.free_env = FreeEnvironment(env_id=env_id, cache=cache, dataset=dataset, state_encoding=state_encoding,
                sys_token=sys_token, usr_token=usr_token, sep_token=sep_token,
                max_steps=max_steps, max_reward=self.max_reward, user_patience=user_patience,
                answer_parser=answer_parser, system_parser=system_parser, logic_parser=logic_parser, 
                value_backend=value_backend,
                auto_skip=auto_skip)

        # TODO add logger
        # TODO forward coverage stats
    
    @property
    def current_episode(self):
        return self.guided_env.current_episode + self.free_env.current_episode

    def reset(self):
        # choose uniformely at random between guided and free env according to ratio
        self.active_env = self.guided_env if random.random() < self.guided_free_ratio else self.free_env
        return self.active_env.reset(current_episode=self.current_episode)

    def step(self, action: int, replayed_user_utterance: Tuple[str, None] = None) -> Tuple[dict, float, bool, dict]:
        obs, reward, done, info = self.active_env.step(action, replayed_user_utterance)
        info[EnvInfo.IS_FAQ] = self.active_env == self.free_env
        return obs, reward, done, info

    def get_goal_node_coverage_free(self):
        return len(self.free_env.goal_node_coverage) / self.data.count_question_nodes()

    def get_goal_node_coverage_guided(self):
        return len(self.guided_env.goal_node_coverage) / self.data.num_guided_goal_nodes

