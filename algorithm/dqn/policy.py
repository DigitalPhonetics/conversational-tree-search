
from copy import deepcopy
from typing import Dict, Optional, Type, Any, Union, Tuple

import torch as th
import torch.nn as nn
import numpy as np

from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, FlattenExtractor
from stable_baselines3.dqn.policies import DQNPolicy, QNetwork
from stable_baselines3.common.type_aliases import Schedule #, PyTorchObs


def to_class(path:str):
    from pydoc import locate
    class_instance = locate(path)
    return class_instance


class CustomDQNPolicy(DQNPolicy):
    """
    Policy class with Q-Value Net and target net for DQN

    :param observation_space: Observation space
    :param action_space: Action space
    :param lr_schedule: Learning rate schedule (could be constant)
    :param net_arch: The specification of the policy and value networks.
    :param activation_fn: Activation function
    :param features_extractor_class: Features extractor to use.
    :param features_extractor_kwargs: Keyword arguments
        to pass to the features extractor.
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param optimizer_class: The optimizer to use,
        ``th.optim.Adam`` by default
    :param optimizer_kwargs: Additional keyword arguments,
        excluding the learning rate, to pass to the optimizer
    """
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Discrete,
        lr_schedule: Schedule,
        net_arch,
        normalization_layers: bool = False,
        activation_fn: Type[nn.Module] = nn.ReLU,
        features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = None,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        torch_compile: bool = True,
        num_prediction_heads: int = 1
    ) -> None:
        self.normalization_layers = normalization_layers
        self.torch_compile = torch_compile
        self.num_prediction_heads = num_prediction_heads

        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch, # net_arch -> replaced here by hidden_layer_sizes
            activation_fn,
            features_extractor_class,
            features_extractor_kwargs,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            normalize_images=normalize_images,
        )

    def make_q_net(self, num_prediction_heads: int = None) -> QNetwork:
        # Make sure we always have separate networks for features extractors etc
        net_args = deepcopy(self._update_features_extractor(self.net_args, features_extractor=None))
        del net_args['normalize_images']
        del net_args['features_extractor']
        del net_args['features_dim']
        del net_args['activation_fn']

        # if not isinstance(num_prediction_heads, type(None)):
        #     # target network: initialize with 1 head only, since it is the specific target for a single head from the online network
        #     del net_args['num_prediction_heads']
        #     net_args['num_prediction_heads'] = 1
        
        arch = net_args.pop('net_arch')
        net_cls = to_class(arch.pop('net_cls'))
        
        model = net_cls(**net_args, **arch).to(self.device)
        if self.torch_compile:
            print("COMPILE=TRUE")
            model = th.compile(model)
        else:
            print("COMPILE=FALSE")
        self.intent_prediction = model.intent_prediction
        print("ARCHITECUTRE", model)
        return model
    
    # def _build(self, lr_schedule: Schedule) -> None:
    #     """
    #     Create the network and the optimizer.

    #     Put the target network into evaluation mode.

    #     :param lr_schedule: Learning rate schedule
    #         lr_schedule(1) is the initial learning rate
    #     """

    #     self.q_net = self.make_q_net()
    #     if self.num_prediction_heads == 1:
    #         self.q_net_target = self.make_q_net()
    #         self.q_net_target.load_state_dict(self.q_net.state_dict())
    #         self.q_net_target.set_training_mode(False)
    #     else:
    #         q_net_layers = dict(self.q_net.named_children())

    #         q_net_target = []
    #         for head_idx in range(self.num_prediction_heads):
    #             target_net = self.make_q_net(num_prediction_heads=1)
    #             with th.no_grad():
    #                 target_layers = dict(target_net.named_children())
    #                 for layer_name in self.q_net.get_shared_module_names():
    #                     target_layers[layer_name].weight.copy_(q_net_layers[layer_name].weight)
    #                     target_layers[layer_name].bias.copy_(q_net_layers[layer_name].bias)
    #                     # TODO are there any .buffers we need to copy?
    #             q_net_target.append(target_net)
        
    #         self.q_net_target = nn.ModuleDict({head_idx: target_net for head_idx, target_net in enumerate(q_net_target)})

    #     # Setup optimizer with initial learning rate
    #     self.optimizer = self.optimizer_class(  # type: ignore[call-arg]
    #         self.parameters(),
    #         lr=lr_schedule(1),
    #         **self.optimizer_kwargs,
    #     )
    
    # obs: PyTorchObs
    def _predict(self, obs, deterministic: bool = True, prediction_head_mask: th.Tensor = None) -> th.Tensor:
        return self.q_net._predict(obs, deterministic=deterministic, prediction_head_mask=prediction_head_mask)
    
    def predict(
        self,
        observation: Union[np.ndarray, Dict[str, np.ndarray]],
        state: Optional[Tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
        prediction_head_mask: th.Tensor = None
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        """
        Get the policy action from an observation (and optional hidden state).
        Includes sugar-coating to handle different observations (e.g. normalizing images).

        :param observation: the input observation
        :param state: The last hidden states (can be None, used in recurrent policies)
        :param episode_start: The last masks (can be None, used in recurrent policies)
            this correspond to beginning of episodes,
            where the hidden states of the RNN must be reset.
        :param deterministic: Whether or not to return deterministic actions.
        :return: the model's action and the next hidden state
            (used in recurrent policies)
        """
        # Switch to eval mode (this affects batch norm / dropout)
        self.set_training_mode(False)

        observation, vectorized_env = self.obs_to_tensor(observation)

        with th.no_grad():
            actions = self._predict(observation, deterministic=deterministic, prediction_head_mask=prediction_head_mask)

        intents = None
        if self.intent_prediction:
            # unpack tuple
            actions, intents = actions

        # Convert to numpy, and reshape to the original action shape
        # TODO do we really need to reshape here?
        actions = actions.cpu().numpy() 
        # TODO I commented out the reshape: .reshape((-1, *self.action_space.shape))

        if isinstance(self.action_space, spaces.Box):
            if self.squash_output:
                # Rescale to proper domain when using squashing
                actions = self.unscale_action(actions)
            else:
                # Actions could be on arbitrary scale, so clip the actions to avoid
                # out of bound error (e.g. if sampling from a Gaussian distribution)
                actions = np.clip(actions, self.action_space.low, self.action_space.high)

        # Remove batch dimension if needed
        if not vectorized_env:
            actions = actions.squeeze(axis=0)
            if th.is_tensor(intents):
                intents = intents.squeeze(0)

        return actions, state, intents