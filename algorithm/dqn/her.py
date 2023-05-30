import copy
from dataclasses import dataclass
from collections import deque
from statistics import mean
from typing import Any, Dict, List, NamedTuple, Tuple, Union

import numpy as np
import torch as th
from algorithm.dqn.buffer import CustomReplayBuffer
from chatbot.adviser.app.answerTemplateParser import AnswerTemplateParser
from chatbot.adviser.app.rl.goal import DummyGoal, UserInput, UserResponse

from chatbot.adviser.app.rl.utils import AutoSkipMode, EnvInfo
from chatbot.adviser.app.systemTemplateParser import SystemTemplateParser
from config import ActionType, InstanceType, INSTANCES
from data.dataset import DialogNode, GraphDataset, NodeType
from encoding.state import StateDims, StateEncoding
from environment.her import CTSHEREnvironment
from chatbot.adviser.app.rl.utils import rand_remove_questionmark


class HERReplaySample(NamedTuple):
    observations: np.ndarray
    next_observations: np.ndarray
    action: np.ndarray
    reward: np.ndarray
    done: np.ndarray
    info: List[Dict[EnvInfo, Any]]



AVERAGE_WINDOW = 1000

class HindsightExperienceReplayWrapper(object):
    """
    Wrapper around a replay buffer in order to use HER.
    This implementation is inspired by to the one found in https://github.com/NervanaSystems/coach/.

    :param replay_buffer: (ReplayBuffer)
    :param n_sampled_goal: (int) The number of artificial transitions to generate for each actual transition
    :param goal_selection_strategy: (GoalSelectionStrategy) The method that will be used to generate
        the goals for the artificial transitions.
    :param wrapped_env: (HERGoalEnvWrapper) the GoalEnv wrapped using HERGoalEnvWrapper,
        that enables to convert observation to dict, and vice versa
    """

    def __init__(self, buffer_size: int,
                    observation_space,
                    action_space,
                    num_train_envs: int,
                    batch_size: int,
                    dataset: GraphDataset,
                    append_ask_action: bool, # if we should always end the episode with ASKED_GOAL=True by appending an artificial ASK transition
                    # state_encoding: StateEncoding,
                    auto_skip: AutoSkipMode,
                    normalize_rewards: bool,
                    max_steps: int,
                    user_patience: int,
                    sys_token: str, usr_token: str, sep_token: str,
                    stop_when_reaching_goal: bool,
                    stop_on_invalid_skip: bool,
                    device: Union[th.device, str] = "cpu",
                    **kwargs):
        
        self.append_ask_action = append_ask_action
        self.replay_buffer = CustomReplayBuffer(buffer_size=buffer_size, observation_space=observation_space, action_space=action_space, device=device, **kwargs)
        self.batch_size = batch_size
        self.data = dataset
        self.system_parser = SystemTemplateParser()
        self.env = CTSHEREnvironment(dataset=dataset, auto_skip=auto_skip,
                                    normalize_rewards=normalize_rewards,
                                    max_steps=max_steps, user_patience=user_patience,
                                    stop_when_reaching_goal=stop_when_reaching_goal, stop_on_invalid_skip=stop_on_invalid_skip,
                                    sys_token=sys_token, usr_token=usr_token, sep_token=sep_token)

        # Buffer for storing transitions of the current episode, for vectorized environment
        self.episode_transitions: List[List[HERReplaySample]] = [list() for _ in range(num_train_envs)]
        # Buffer for storing artificial transitions until we have enough to process a full batch
        self.artificial_transition_buffer: List[HERReplaySample] = []
        self.staging_transitions: List[HERReplaySample] = []

        # stats
        self.artifical_rewards_free = deque([], maxlen=AVERAGE_WINDOW) # reward over last n episodes
        self.artifical_rewards_guided = deque([], maxlen=AVERAGE_WINDOW) # reward over last n episodes
        self.replay_success_free = deque([], maxlen=AVERAGE_WINDOW) # successful replays over last n episodes
        self.replay_success_guided = deque([], maxlen=AVERAGE_WINDOW) # successful replays over last n episodes

    @property
    def artificial_episodes(self):
        return self.env.current_episode
    
    @property
    def artificial_mean_episode_reward_free(self):
        return mean(self.artifical_rewards_free) if len(self.artifical_rewards_free) > 0 else 0
    
    @property
    def artificial_mean_episode_reward_guided(self):
        return mean(self.artifical_rewards_guided) if len(self.artifical_rewards_guided) > 0 else 0
    
    @property
    def replay_success_mean_free(self):
        return mean(self.replay_success_free) if len(self.replay_success_free) > 0 else 0
    
    @property
    def replay_success_mean_guided(self):
        return mean(self.replay_success_guided) if len(self.replay_success_guided) > 0 else 0
    

    def add(
        self,
        obs: th.Tensor,
        next_obs: th.Tensor,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: List[Dict[str, Any]],
    ) -> None:
        # Update normal replay buffer
        self.replay_buffer.add(obs, next_obs, action, reward, done, infos)

        for env_idx, env_info in enumerate(infos):
             # add to episode buffer
            self.episode_transitions[env_idx].append(HERReplaySample(
                obs[env_idx].clone().detach(), next_obs[env_idx].clone().detach(), action.item(env_idx),
                reward.item(env_idx), done.item(env_idx), copy.deepcopy(env_info)
            ))
            if done.item(env_idx):
                # generate sub-goals
                self._generate_aritificial_transitions(env_idx)
                # reset episode buffer
                self.episode_transitions[env_idx].clear()
               

    def sample(self, *args, **kwargs):
        return self.replay_buffer.sample(*args, **kwargs)

    def __len__(self):
        return len(self.replay_buffer)

    def _staging_complete(self):
        self.artificial_transition_buffer.extend(self.staging_transitions)

        # trigger batch encoding when full (>= batch_size elements)
        while len(self.artificial_transition_buffer) >= self.batch_size:
            # encode states
            batch_obs = INSTANCES[InstanceType.STATE_ENCODING].batch_encode([transition.observations for transition in self.artificial_transition_buffer[:self.batch_size]], sys_token=self.env.sys_token, usr_token=self.env.usr_token, sep_token=self.env.sep_token)
            batch_next_obs = INSTANCES[InstanceType.STATE_ENCODING].batch_encode([transition.next_observations for transition in self.artificial_transition_buffer[:self.batch_size]], sys_token=self.env.sys_token, usr_token=self.env.usr_token, sep_token=self.env.sep_token)
            # encode actions, dones, infos, rewards
            batch_actions = np.array([transition.action for transition in self.artificial_transition_buffer[:self.batch_size]], dtype=np.int16)
            batch_dones = np.array([transition.done for transition in self.artificial_transition_buffer[:self.batch_size]], dtype=np.int8)
            batch_rewards = np.array([transition.reward for transition in self.artificial_transition_buffer[:self.batch_size]], dtype=np.float32)
            batch_infos = [transition.info for transition in self.artificial_transition_buffer[:self.batch_size]]

            # add to real buffer
            self.replay_buffer.add(batch_obs, batch_next_obs, batch_actions, batch_rewards, batch_dones, batch_infos, is_aritificial=True)

            # reset buffer
            self.artificial_transition_buffer = self.artificial_transition_buffer[self.batch_size:]
        
        # reset staging area
        self.staging_transitions.clear()

    
    def _store_aritificial_transition(self, obs, next_obs, action, reward, done, infos):
        self.staging_transitions.append(HERReplaySample(obs, next_obs, action, reward, done, infos))

        
    
    def _draw_artificial_goal(self, original_transitions: List[HERReplaySample], goal_candidate_indicator_method) -> Union[Tuple[DummyGoal, int], None]:
        # walk backwards until we find a suitable goal candidate
        if original_transitions[-1].info[EnvInfo.LAST_VALID_SKIP_TRANSITION_IDX] < 0:
            # No single valid SKIP in episode - don't replay, otherwise it will include invalid jump mappings
            return None 
        goal = None
        final_transition_idx = original_transitions[-1].info[EnvInfo.LAST_VALID_SKIP_TRANSITION_IDX]
        
        if final_transition_idx + 1 < len(original_transitions):
            # check if last action is ASK - if so, keep it 
            if original_transitions[final_transition_idx + 1].action == ActionType.ASK:
                final_transition_idx += 1
            
        for transition in reversed(original_transitions[:final_transition_idx+1]):
            node = self.data.nodes_by_key[transition.info[EnvInfo.DIALOG_NODE_KEY]]
            if not goal_candidate_indicator_method(node):
                continue
            # we found a goal node candidate
            goal = copy.deepcopy(transition.info[EnvInfo.GOAL])
            goal.goal_node_key = node.key

            # change visited_ids to reflect new correct path
            path_ids = set()
            for i in range(final_transition_idx):
                path_ids.add(original_transitions[i].info[EnvInfo.DIALOG_NODE_KEY])
            goal.visited_ids = path_ids
            goal.constraints = original_transitions[final_transition_idx].info[EnvInfo.BELIEFSTATE]

            return goal, final_transition_idx
        return None
    
    def _replay_episode(self, mode: str, original_transitions: List[HERReplaySample], artificial_goal: Tuple[DummyGoal, int], final_transition_idx: int) -> float:
        # replay episode with new goal
        episode_reward = 0.0
        obs = self.env.reset(mode=mode, replayed_goal=artificial_goal)
        done = False
        transition_idx = 0
        while not done and transition_idx <= final_transition_idx:
            # get original transition
            original_transition = original_transitions[transition_idx]
            # recover original action
            original_action = original_transition.action
            # replay action
            next_obs, reward, done, info = self.env.step(original_action, replayed_user_utterance=original_transition.info[EnvInfo.CURRENT_USER_UTTERANCE])
            # record new observations
            self._store_aritificial_transition(obs, next_obs, original_action, reward, done, info)
            episode_reward += reward
            transition_idx += 1
        
        if info[EnvInfo.REACHED_GOAL_ONCE] == True:
            if self.append_ask_action and not (original_action == ActionType.ASK):
                # append an artificial ASK action as last action, if replayed episode didn't end in one
                next_obs, reward, done, info = self.env.step(ActionType.ASK) 
                self._store_aritificial_transition(obs, next_obs, original_action, reward, done, info)
                episode_reward += reward
                transition_idx += 1
                assert info[EnvInfo.ASKED_GOAL] == True, "replay did not ask goal node"
            # integrate artificial experiences
            self._staging_complete()
            return episode_reward
        # not successful - reset staging transitions
        self.staging_transitions = []
        return None 
    
    def _replay_guided(self, original_transitions: List[HERReplaySample]):
        goal = self._draw_artificial_goal(original_transitions, self.env.guided_env.goal_gen._is_guided_goal_candidate)
        if isinstance(goal, type(None)): 
            # we didn't find a goal candidate - return 
            self.replay_success_guided.append(0.0)
            return
           
        # replay
        goal, final_transition_idx = goal
        total_reward = self._replay_episode(mode='guided', original_transitions=original_transitions, artificial_goal=goal, final_transition_idx=final_transition_idx)
        if isinstance(total_reward, type(None)):
            self.replay_success_guided.append(0)
        else:
            self.artifical_rewards_guided.append(total_reward)
            self.replay_success_guided.append(1.0)

    def _replay_free(self, original_transitions: List[HERReplaySample]):
        goal = self._draw_artificial_goal(original_transitions, self.env.free_env.goal_gen._is_free_goal_candidate)
        if isinstance(goal, type(None)): 
            # we didn't find a goal candidate - return 
            self.replay_success_free.append(0.0)
            return
        
        # modify goal, if necessary
        goal, final_transition_idx = goal
        goal_node = self.data.nodes_by_key[goal.goal_node_key]
        initial_user_utterance = original_transitions[0].info[EnvInfo.INITIAL_USER_UTTERANCE].replace("?", "")
        if not initial_user_utterance in [question.text.replace("?", "") for question in goal_node.questions]:
            # original user utterance is not part of questions associated with new goal node ->s change goal
            question = goal_node.random_question()
            # create dummy goal
            goal.delexicalised_initial_user_utterance = rand_remove_questionmark(question.text)
            try:
                goal.initial_user_utterance = self.system_parser.parse_template(goal.delexicalised_initial_user_utterance, self.env.free_env.value_backend, goal.constraints)
            except:
                print('missing bst value for HER first utterance')
                self.replay_success_guided.append(0)
                return

        # otherwise, we can keep the same goal

        # replay
        total_reward = self._replay_episode(mode='free', original_transitions=original_transitions, artificial_goal=goal, final_transition_idx=final_transition_idx)
        if isinstance(total_reward, type(None)):
            self.replay_success_guided.append(0)
        else:
            self.artifical_rewards_free.append(total_reward)
            self.replay_success_free.append(1.0)

    def _generate_aritificial_transitions(self, env_idx: int):
        """
        Sample artificial goals and store transition of the current
        episode in the replay buffer.
        This method is called only after each end of episode.
        """
        original_transitions = self.episode_transitions[env_idx]
        if original_transitions[-1].info[EnvInfo.ASKED_GOAL]:
            # episode was already successful - nothing to do
            return
        # episode was not successful: replay
        if original_transitions[0].info[EnvInfo.IS_FAQ]:
            self._replay_free(original_transitions)
        else:
            self._replay_guided(original_transitions)
        





