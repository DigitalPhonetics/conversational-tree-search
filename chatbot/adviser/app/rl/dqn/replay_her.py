from copy import deepcopy
from typing import Any, Dict, Union
import torch
from chatbot.adviser.app.answerTemplateParser import AnswerTemplateParser
from chatbot.adviser.app.logicParser import LogicTemplateParser
from chatbot.adviser.app.parserValueProvider import RealValueBackend
from chatbot.adviser.app.rl.dialogenv import DialogEnvironment, EnvironmentMode, ParallelDialogEnvironment
from chatbot.adviser.app.rl.dqn.replay_prioritized import PrioritizedLAPReplayBuffer
from chatbot.adviser.app.rl.goal import DummyGoal, UserResponse
from chatbot.adviser.app.rl.utils import AutoSkipMode, AverageMetric, ExperimentLogging, StateEntry, EnvInfo, rand_remove_questionmark
from chatbot.adviser.app.systemTemplateParser import SystemTemplateParser
from data.dataset import GraphDataset, DialogNode

from chatbot.adviser.app.rl.spaceAdapter import SpaceAdapter

import wandb




class HindsightExperienceReplay(PrioritizedLAPReplayBuffer):
    # Idea: change goal s.t. bad trajectory gets a positive reward
    # - pretend our goal was one of the states on bad trajectory
    # - adapt rewards, states and add positive trajectory to the replay buffer as well

    def __init__(
        self,
        envs: ParallelDialogEnvironment,
        buffer_size: int,
        adapter: SpaceAdapter,
        dialog_tree: GraphDataset,
        answerParser: AnswerTemplateParser,
        logicParser: LogicTemplateParser,
        dialog_faq_ratio: float,
        max_reward: float,
        train_noise: float,
        device: Union[torch.device, str] = "cpu",
        alpha: float = 0.6, beta: float = 0.4,
        experiment_logging: ExperimentLogging = ExperimentLogging.NONE,
        auto_skip: AutoSkipMode = AutoSkipMode.NONE,
        stop_when_reaching_goal: bool = False,
        similarity_model=None,
    ):
        super().__init__(buffer_size, adapter, device, alpha, beta)
        self.n_envs = len(envs.envs)
        self.dialog_faq_ratio = dialog_faq_ratio
        self.dialog_tree = dialog_tree
        self.answerParser = answerParser
        self.logicParser = logicParser
        self.system_parser = SystemTemplateParser()
        self.value_backend = RealValueBackend(dialog_tree)
        self.adapter = adapter
        self.max_reward = max_reward
        self.hindsight_episode_rewards = AverageMetric(name="train/HER_episode_rewards", running_avg=25)
        self.hindsight_episodes = AverageMetric(name="train/HER_episodes", running_avg=25)
        self.hindsight_goal_fails = AverageMetric(name="train/HER_goal_fail", running_avg=25)
        self.experiment_logging = experiment_logging
        self.call_counter = 0
        self.envs = envs
        self.her_env = DialogEnvironment(dialog_tree=dialog_tree, adapter=adapter, mode=EnvironmentMode.TRAIN,
                                            stop_action=adapter.configuration.stop_action, 
                                            train_noise=train_noise, eval_noise=0.0, test_noise=0.0,
                                            max_steps=self.envs.envs[0].max_steps, user_patience=self.envs.envs[0].user_patience,
                                            normalize_rewards=self.envs.envs[0].normalize_rewards, stop_when_reaching_goal=stop_when_reaching_goal,
                                            dialog_faq_ratio=dialog_faq_ratio, log_to_file=None, env_id=1, goal_gen=self.envs.envs[0].goal_gen,
                                            logic_parser=self.logicParser, answer_template_parser=self.envs.envs[0].answer_template_parser,
                                            return_obs=True, auto_skip=auto_skip, similarity_model=similarity_model)

        # HER episode buffers: record stuff from beginning of an episode until end
        self.s_episode = [[] for _ in range(self.n_envs)]
        self.s_next_episode = [[] for _ in range(self.n_envs)]
        self.a_episode = [[] for _ in range(self.n_envs)]
        self.i_episode = [[] for _ in range(self.n_envs)]

    def _draw_new_goal_free(self, env_id: int) -> DummyGoal:
        # ensure we have at least 1 transition in the trajectory and are not stopping directly 
        if len(self.i_episode[env_id]) <= 1:
            return

        goal_idx = None
        initial_user_utterance = self.envs.envs[env_id].initial_user_utterance
        faq_key = self.envs.envs[env_id].goal.faq_key
        
        for idx in reversed(range(len(self.i_episode[env_id]))):
            # walk backward until we find a suitable goal candidate
            goal_idx = len(self.i_episode[env_id]) - idx # revert index since we are traversing i_episode backwards
            node = self.dialog_tree.nodes_by_key[self.s_episode[env_id][goal_idx-1][StateEntry.DIALOG_NODE_KEY.value]]
            if not self._is_freeform_goal_node_candidate(node, self.a_episode[env_id][goal_idx-1]):
                continue

            # we found a candidate - save index
            if not initial_user_utterance.replace("?", "") in [faq.text.replace("?", "") for faq in node.questions]:
                # original user utterance is not part of FAQ questions associated with new goal node - change goal
                faq = node.random_question()
                faq_key = faq.key
                initial_user_utterance = rand_remove_questionmark(faq.text)
            bst = self.s_episode[env_id][goal_idx-1][StateEntry.BST.value] # keep bst from current turn
            # check if we have all the variables required to fill the system template
            missing_variables = self.system_parser.find_variables(node.text).union(self.system_parser.find_variables(initial_user_utterance)) - set(bst.keys())
            if not missing_variables:
                # we can reach the current node given our bst
                self.hindsight_goal_fails.log(0)
                self.hindsight_episodes.log(1)
                initial_user_utterance = self.system_parser.parse_template(initial_user_utterance, self.value_backend, self.s_episode[env_id][-1][StateEntry.BST.value])
                return DummyGoal(goal_idx, node, faq_key, initial_user_utterance, bst, 
                                    {node_key: UserResponse(relevant=True, answer_key=self.envs.envs[env_id].user_answer_keys[node_key].answer_key) if self.envs.envs[env_id].user_answer_keys[node_key] else None for node_key in self.envs.envs[env_id].user_answer_keys},
                                    self.answerParser)

        # we didn't find a reachable goal in trajectory
        self.hindsight_goal_fails.log(1)
        self.hindsight_episodes.log(0)
        return None

    def _is_freeform_goal_node_candidate(self, current_node: DialogNode, action: int) -> bool:
        # node should either be asked or stopped on to be an eligible candidate with positive reward
        # also, it has to contain at least 1 FAQ question
        action_offset = 0 if self.adapter.configuration.stop_action else 1
        return action + action_offset < 2 and len(current_node.questions) > 0

    def _replay_freeform(self, env_id: int):
        # look through trajectory and choose last state that ended in a node with associated FAQ (if in FAQ or mixed mode)
        # if in dialog mode, change goal s.t. last node is goal
        # assert self.a_episode[env_id].count(0) <= 1, f"More than 1 STOP action in trajectory!, { self.a_episode[env_id] }"

        # find new goal along the recorded trajectory
        goal = self._draw_new_goal_free(env_id)
        if not goal:
            return
       
        obs = self.her_env._her_faq_reset(goal) # create initial observation with new goal
        for idx in range(goal.goal_idx):
            action = self.a_episode[env_id][idx] # get action from trajectory
            user_utterance = self.s_next_episode[env_id][idx][StateEntry.CURRENT_USER_UTTERANCE.value] # get user utterance from trajectory
            assert action == self.a_episode[env_id][idx]
            next_obs, reward, done, info = self.her_env.step(action, _replayed_user_utterance=user_utterance)

            # add new imagined experience to buffer
            super().add(env_id=env_id, obs=obs, next_obs=next_obs, action=action, reward=reward, done=done, info=info, global_step=-1)
            obs = next_obs
            
        if self.her_env.reached_goal_once:
            self.hindsight_goal_fails.log(0)
            self.hindsight_episodes.log(1)
        else:
            self.hindsight_goal_fails.log(1)
            self.hindsight_episodes.log(1)
            print("HER replay error: did not reach goal")
        self.hindsight_episode_rewards.log(self.her_env.episode_reward)

    def _copy_obs(self, obs: Dict[str, Any]):
        return {
            key: obs[key].clone().detach().cpu() if torch.is_tensor(obs[key]) else deepcopy(obs[key]) for key in obs
        }

    def add(
        self,
        env_id: int,
        obs: Dict[StateEntry, Any],
        next_obs: Dict[StateEntry, Any],
        action: int,
        reward: float,
        done: bool,
        infos: Dict[EnvInfo, Any],
        global_step: int
    ):
        # add normal observation
        if next_obs != None:
            super().add(env_id, obs, next_obs, action, reward, done, infos, global_step)
            self.call_counter += 1

            # add to HER episode buffers
            self.s_episode[env_id].append(self._copy_obs(obs))
            self.s_next_episode[env_id].append(self._copy_obs(next_obs))
            self.a_episode[env_id].append(action)
            self.i_episode[env_id].append(deepcopy(infos))
        
        if done:
            if not infos[EnvInfo.REACHED_GOAL_ONCE] and infos[EnvInfo.IS_FAQ]:
                # episide is over, add new hindisght experiences if episode was not successful
                self._replay_freeform(env_id)
            else:
                self.hindsight_episodes.log(0)
            # reset episode trajectory buffers
            self.s_episode[env_id] = []
            self.s_next_episode[env_id] = []
            self.a_episode[env_id] = []
            self.i_episode[env_id] = [] 

        if self.experiment_logging != ExperimentLogging.NONE:
            wandb.log({
                    self.hindsight_episode_rewards.name: self.hindsight_episode_rewards.eval(),
                    self.hindsight_episodes.name: self.hindsight_episodes.eval(),
                    self.hindsight_goal_fails.name: self.hindsight_goal_fails.eval()
                },
                step=global_step,
                commit=(global_step % 2 == 0)
            )
