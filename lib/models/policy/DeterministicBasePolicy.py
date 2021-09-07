import abc
import torch
import torch.nn as nn
from typing import Callable


class DeterministicBasePolicy(nn.Module):

    def __init__(
            self,
            state_size: int,
            action_size: int,
            seed: int):
        """
        Initialize the policy

        @param state_size: Dimension of each state
        @param action_size: Dimension of each action
        @param seed: Random seed
        @param output_transform: optional, generic transformation to be applied to policy output
        """
        super().__init__()
        self.action_size = action_size
        self.state_size = state_size
        self.seed = torch.manual_seed(seed)

    @abc.abstractmethod
    def forward(self, states: torch.Tensor) -> torch.Tensor:
        pass

