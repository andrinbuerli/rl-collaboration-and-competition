import abc
import torch
import torch.nn as nn


class StateActionValueFunction(nn.Module):

    def __init__(
            self,
            state_size: int,
            action_size: int,
            seed: int):
        super().__init__()
        self.state_size = state_size
        self.seed = torch.manual_seed(seed)

        self.value_function_network = nn.Sequential(
            nn.Linear(state_size + action_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        x = torch.cat((states, actions), dim=1)
        return self.value_function_network(x)
