from typing import Callable
import torch
import torch.nn as nn

from lib.models.policy.DeterministicBasePolicy import DeterministicBasePolicy


class DeterministicContinuousGaussianPolicy(DeterministicBasePolicy):
    def __init__(
            self,
            state_size: int,
            action_size: int,
            seed: int,
            output_transform: Callable[[torch.Tensor], torch.Tensor] = None):
        """
        Stochastic policy which learns to sample an action from a continuous multivariate gaussian distribution where
        each action dimension is considered to be independent.
        """
        super().__init__(state_size=state_size, action_size=action_size, seed=seed)

        self.output_transform = output_transform
        self.policy_network = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, self.action_size)
        )

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        x = self.policy_network(states.to(torch.float32))
        if self.output_transform is not None:
            return self.output_transform(x)
        else:
            return x
