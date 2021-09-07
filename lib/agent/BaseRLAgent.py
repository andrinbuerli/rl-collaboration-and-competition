import numpy as np
import abc
import torch
import torch.nn as nn
import os


class BaseRLAgent:

    def __init__(
            self,
            models: [nn.Module],
            model_names: [str],
            device: str,
            learning_rate: float):
        """
        Initialize the BaseRLAgent

        @param models: a list of the models used by the agent
        @param model_names: a list of the model names used by the agent (required for save / load)
        @param device: the device on which the calculations are to be executed
        @param learning_rate: the learning rate which is used for all optimizers
        """

        self.model_names = model_names
        self.learning_rate = learning_rate
        self.models = models
        self.device = device

    def _get_optimizer(self, model: nn.Module) -> torch.optim.Optimizer:
        return torch.optim.Adam(model.parameters(), lr=self.learning_rate)

    @abc.abstractmethod
    def act(self, states: np.ndarray) -> (np.ndarray, np.ndarray, np.array):
        """
        Determine next actions for each state

        @param states: array of states
        @return: (actions, action logits, log probability of action logits)
        """
        pass

    @abc.abstractmethod
    def learn(
            self,
            states: np.ndarray,
            actions: np.ndarray,
            action_logits: np.ndarray,
            action_log_probs: np.ndarray,
            rewards: np.ndarray,
            next_states: np.ndarray,
            dones: np.ndarray):
        """
        Learn from sampled trajectories.

        @param states: The state at time step t
        @param actions: The sampled value of the chosen action at time step t
        @param action_logits: The sampled logit value of the chosen action at time step t
        @param action_log_probs: The log probability of the chosen action at time step t
        @param rewards: The reward received from the environment at time step t+1
        @param next_states: The state at time step t+1
        @return:
        """
        pass

    @abc.abstractmethod
    def get_name(self) -> str:
        pass

    @abc.abstractmethod
    def get_log_dict(self) -> dict:
        pass

    def reset(self) -> dict:
        pass

    def load(self, directory_name: str, device="cpu"):
        if not os.path.exists(directory_name):
            raise FileNotFoundError(f"Directory {directory_name} not found")

        for name, model in zip(self.model_names, self.models):
            model.load_state_dict(torch.load(os.path.join(directory_name, name + ".pth"), map_location=torch.device(device)))

    def save(self, directory_name: str):
        if not os.path.exists(directory_name):
            os.mkdir(directory_name)

        for name, model in zip(self.model_names, self.models):
            torch.save(model.state_dict(), os.path.join(directory_name, name + ".pth"))
