from copy import deepcopy
from typing import Any, Dict, Union
import torch
from chatbot.adviser.app.answerTemplateParser import AnswerTemplateParser
from chatbot.adviser.app.logicParser import LogicTemplateParser
from chatbot.adviser.app.parserValueProvider import RealValueBackend
from chatbot.adviser.app.rl.dialogenv import DialogEnvironment, EnvironmentMode, ParallelDialogEnvironment, _load_a1_laenderliste, _load_answer_synonyms
from chatbot.adviser.app.rl.dialogtree import DialogTree
from chatbot.adviser.app.rl.dqn.replay_prioritized import PrioritizedLAPReplayBuffer
from chatbot.adviser.app.rl.goal import DummyGoal, UserResponse
from chatbot.adviser.app.rl.utils import AutoSkipMode, AverageMetric, ExperimentLogging, StateEntry, EnvInfo, rand_remove_questionmark
from chatbot.adviser.app.systemTemplateParser import SystemTemplateParser
from chatbot.adviser.app.rl.dataset import DialogNode
import chatbot.adviser.app.rl.dataset as Data

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
        dialog_tree: DialogTree,
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
        self.a1_laenderliste = _load_a1_laenderliste()
        self.answer_synonyms = _load_answer_synonyms(EnvironmentMode.EVAL, adapter.configuration.use_answer_synonyms,use_joint_dataset=envs.envs[0].use_joint_dataset)
        self.value_backend = RealValueBackend(self.a1_laenderliste)
        self.adapter = adapter
        self.max_reward = max_reward
        self.hindsight_episode_rewards = AverageMetric(name="train/HER_episode_rewards", running_avg=25)
        self.hindsight_episodes = AverageMetric(name="train/HER_episodes", running_avg=25)
        self.hindsight_goal_fails = AverageMetric(name="train/HER_goal_fail", running_avg=25)
        self.experiment_logging = experiment_logging
        self.call_counter = 0
        self.envs = envs
        self.her_env = DialogEnvironment(dialog_tree=dialog_tree, adapter=adapter, mode=EnvironmentMode.TRAIN,
                                            stop_action=adapter.configuration.stop_action, use_answer_synonyms=adapter.configuration.use_answer_synonyms,
                                            train_noise=train_noise, eval_noise=0.0, test_noise=0.0,
                                            max_steps=self.envs.envs[0].max_steps, user_patience=self.envs.envs[0].user_patience,
                                            normalize_rewards=self.envs.envs[0].normalize_rewards, stop_when_reaching_goal=stop_when_reaching_goal,
                                            dialog_faq_ratio=dialog_faq_ratio, log_to_file=None, env_id=1, goal_gen=self.envs.envs[0].goal_gen, a1_laenderliste=self.a1_laenderliste,
                                            logic_parser=self.logicParser, answer_template_parser=self.envs.envs[0].answer_template_parser, answer_synonyms=self.answer_synonyms, 
                                            return_obs=True, auto_skip=auto_skip, similarity_model=similarity_model,
                                            use_joint_dataset=envs.envs[0].use_joint_dataset)

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
            # node: DialogNode = DialogNode.objects.get(version=self.dialog_tree.version, key=info[EnvInfo.NODE_KEY]) # get node from state info
            goal_idx = len(self.i_episode[env_id]) - idx # revert index since we are traversing i_episode backwards
            node = Data.objects[self.dialog_tree.version].node_by_key(self.s_episode[env_id][goal_idx-1][StateEntry.DIALOG_NODE_KEY.value])
            if not self._is_freeform_goal_node_candidate(node, self.a_episode[env_id][goal_idx-1]):
                continue

            # we found a candidate - save index
            # if not initial_user_utterance.replace("?", "") in [faq.text.replace("?", "") for faq in node.faq_questions.all()]:
            if not initial_user_utterance.replace("?", "") in [faq.text.replace("?", "") for faq in node.faq_questions]:
                # original user utterance is not part of FAQ questions associated with new goal node - change goal
                # faq_key = random.choice(node.faq_questions.values_list("key", flat=True))
                # initial_user_utterance = rand_remove_questionmark(FAQQuestion.objects.get(version=self.dialog_tree.version, key=faq_key).text)
                faq = node.random_faq()
                faq_key = faq.key
                initial_user_utterance = rand_remove_questionmark(faq.text)
            bst = self.s_episode[env_id][goal_idx-1][StateEntry.BST.value] # keep bst from current turn
            # check if we have all the variables required to fill the system template
            missing_variables = self.system_parser.find_variables(node.content.text).union(self.system_parser.find_variables(initial_user_utterance)) - set(bst.keys())
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
        # return (action == 1 or action == 0) and (current_node.faq_questions.count() > 0)
        action_offset = 0 if self.adapter.configuration.stop_action else 1
        return action + action_offset < 2 and (current_node.faq_questions_count() > 0)

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
            # assert self.her_env.current_node.key == self.s_next_episode[env_id][idx][StateEntry.DIALOG_NODE_KEY.value]
            # assert info[EnvInfo.NODE_KEY] == self.i_episode[env_id][idx][EnvInfo.NODE_KEY] # NOTE fails because last node (ASK -> auto-skip) is not in goal's answer_pk's

            # add new imagined experience to buffer
            super().add(env_id=env_id, obs=obs, next_obs=next_obs, action=action, reward=reward, done=done, info=info, global_step=-1)
            obs = next_obs
            
        # assert self.her_env.reached_goal_once, "HER did not reach goal!"
        if self.her_env.reached_goal_once:
            self.hindsight_goal_fails.log(0)
            self.hindsight_episodes.log(1)
        else:
            self.hindsight_goal_fails.log(1)
            self.hindsight_episodes.log(1)
            print("HER replay error: did not reach goal")
        self.hindsight_episode_rewards.log(self.her_env.episode_reward)

    # def _replay_guided(self, env_id: int):
    #     # start following the given trajectory, change goal for each step if necessary

    #     start_node = self.dialog_tree.get_start_node().connected_node
        
    #     # find first node transition along trajectory and generate first goal
    #     first_answer = None
    #     idx = 0
    #     while not first_answer:
    #         if self.a_episode[env_id][idx] > 1:
    #             first_answer = list(start_node.answers.order_by('answer_index'))[self.a_episode[env_id][idx]]
    #     if not first_answer:
    #         # no node transitions in episode found - failed to replay
    #         self.hindsight_goal_fails.log(1)
    #         self.hindsight_episodes.log(0)
    #         return
    #     initial_user_utterance = deepcopy(self.envs.envs[env_id].initial_user_utterance)
    #     if not initial_user_utterance.replace("?", "") in [answer_text.replace("?", "") for answer_text in self.answer_synonyms[first_answer.content.text.lower()]]:
    #         # original user utterance is not part of answers associated with new goal node - change goal
    #         initial_user_utterance = rand_remove_questionmark(random.choice(self.answer_synonyms[first_answer.content.text.lower()]))
    #     bst = deepcopy(self.i_episode[env_id][-1]['bst']) # keep bst from last turn
    #     self.hindsight_goal_fails.log(0)
    #     self.hindsight_episodes.log(1)

    #     # start replay
    #     obs = self.her_env._her_guided_reset(first_answer.connected_node, initial_user_utterance)
    #     for idx in range(len(self.a_episode[env_id])):
    #         action = self.a_episode[env_id][idx] # get action from trajectory
    #         assert action == self.a_episode[env_id][idx]
    #         next_obs, reward, done, info = self.her_env.step(action)
    #         assert info['node_key'] == self.i_episode[env_id][idx]['node_key']

    #         # add new imagined experience to buffer
    #         super().add(env_id=env_id, obs=obs, next_obs=next_obs, action=action, reward=reward, done=done, info=info, global_step=-1)
    #         obs = next_obs

    #         # adapt goal
    #         if action == 1:
    #             # find next node transition along trajectory
    #             lookahead_idx = idx
    #             next_answer = None
    #             while lookahead_idx < len(self.a_episode[env_id]) and not next_answer:
    #                 if self.her_env.current_node.node_type == NodeType.QUESTION.value:
    #                     # we found a suitable transition -> get answer and set env goal_node
    #                     answer = 
    #                     self.envs.envs[env_id].goal_node = answer.connected_node
        
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
