import random
from typing import Tuple

from utils.utils import EnvInfo

from data.dataset import GraphDataset

from data.parsers.answerTemplateParser import AnswerTemplateParser
from data.parsers.logicParser import LogicTemplateParser
from data.parsers.systemTemplateParser import SystemTemplateParser
from data.parsers.parserValueProvider import RealValueBackend
from utils.utils import AutoSkipMode
from environment.free import FreeEnvironment
from environment.guided import GuidedEnvironment
from utils.envutils import GoalDistanceMode
import config as cfg

import gymnasium

class CTSEnvironment(gymnasium.Env):
    def __init__(self, 
                # env_id: int,
                mode: str,
                dataset: GraphDataset,
                guided_free_ratio: float,
                auto_skip: AutoSkipMode,
                normalize_rewards: bool,
                max_steps: int,
                user_patience: int,
                stop_when_reaching_goal: bool,
                stop_on_invalid_skip: bool,
                sys_token: str, usr_token: str, sep_token: str,
                goal_distance_mode: GoalDistanceMode,
                goal_distance_increment: int,
                **kwargs):
        # self.env_id = env_id
        self.goal_distance_mode = goal_distance_mode
        self.goal_distance_increment = goal_distance_increment
        self.data = dataset
        self.mode = mode

        self.max_reward = 4 * dataset.get_max_tree_depth() if normalize_rewards else 1.0
        self.max_distance = dataset.get_max_tree_depth() + 1  if goal_distance_mode == GoalDistanceMode.FULL_DISTANCE else 1 # set max. or min. distance to start
        cfg.INSTANCES[cfg.InstanceArgs.MAX_DISTANCE] = self.max_distance

        # text parsers
        answer_parser = AnswerTemplateParser()
        logic_parser = LogicTemplateParser()
        system_parser = SystemTemplateParser()
        value_backend = RealValueBackend(dataset.a1_countries)

        # initialize task-specific environments
        self.guided_free_ratio = guided_free_ratio
        if guided_free_ratio > 0.0:
            self.guided_env = GuidedEnvironment(dataset=dataset,
                sys_token=sys_token, usr_token=usr_token, sep_token=sep_token,
                max_steps=max_steps, max_reward=self.max_reward, user_patience=user_patience,
                stop_when_reaching_goal=stop_when_reaching_goal, stop_on_invalid_skip=stop_on_invalid_skip,
                answer_parser=answer_parser, system_parser=system_parser, logic_parser=logic_parser,
                value_backend=value_backend,
                auto_skip=auto_skip)
        if guided_free_ratio < 1.0:
            self.free_env = FreeEnvironment(dataset=dataset,
                sys_token=sys_token, usr_token=usr_token, sep_token=sep_token,
                max_steps=max_steps, max_reward=self.max_reward, user_patience=user_patience,
                stop_when_reaching_goal=stop_when_reaching_goal, stop_on_invalid_skip=stop_on_invalid_skip,
                answer_parser=answer_parser, system_parser=system_parser, logic_parser=logic_parser, 
                value_backend=value_backend,
                auto_skip=auto_skip)

        # TODO add logger
        # TODO forward coverage stats

        print("ENV!!", mode, "TOKENS:", sys_token, usr_token, sep_token)
    
    @property
    def current_episode(self):
        episode = 0
        if hasattr(self, "guided_env"):
            episode += self.guided_env.current_episode
        if hasattr(self, "free_env"):
            episode += self.free_env.current_episode
        return episode

    def reset(self):
        # adapt max. goald distance
        if self.mode == "train" and self.goal_distance_mode == GoalDistanceMode.INCREMENT_EVERY_N_EPISODES:
            # don't adapt in evaluation / testing, because we have less episodes there 
            self.max_distance = max(self.current_episode // self.goal_distance_increment, 1)
            cfg.INSTANCES[cfg.InstanceArgs.MAX_DISTANCE] = self.max_distance
        elif not self.mode == "train":
            self.max_distance = cfg.INSTANCES[cfg.InstanceArgs.MAX_DISTANCE]

        # choose uniformely at random between guided and free env according to ratio
        self.active_env = self.guided_env if random.random() < self.guided_free_ratio else self.free_env
        return self.active_env.reset(current_episode=self.current_episode, max_distance=self.max_distance)
    
    @property
    def episode_log(self):
        return self.active_env.episode_log

    def step(self, action: int, replayed_user_utterance: Tuple[str, None] = None) -> Tuple[dict, float, bool, dict]:
        obs, reward, done = self.active_env.step(action, replayed_user_utterance)
        obs[EnvInfo.IS_FAQ] = hasattr(self, 'free_env') and (self.active_env == self.free_env)
        obs["is_success"] = obs[EnvInfo.ASKED_GOAL]
        return obs, reward, done, False, obs # truncated, info = obs before encoding
    

    def get_goal_node_coverage_free(self):
        return len(self.free_env.goal_node_coverage) / self.data.count_question_nodes()

    def get_goal_node_coverage_guided(self):
        return len(self.guided_env.goal_node_coverage) / self.data.num_guided_goal_nodes

    def seed(self, seed: int):
        pass
