from typing import Callable
import torch
import torch.nn as nn

from lib.models.policy.DeterministicBasePolicy import DeterministicBasePolicy


class DeterministicDiscretePolicy(DeterministicBasePolicy):
    def __init__(
            self,
            state_size: int,
            action_size: int,
            seed: int):
        """
        Stochastic policy which learns to sample an action from a continuous multivariate gaussian distribution where
        each action dimension is considered to be independent.
        """
        super().__init__(state_size=state_size, action_size=action_size, seed=seed)

        self.policy_network = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, self.action_size),
            nn.Softmax()
        )

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return self.policy_network(states.to(torch.float32))
