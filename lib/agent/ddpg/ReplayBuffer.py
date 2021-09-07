import numpy as np
import random
from collections import namedtuple, deque

import torch


class ReplayBuffer:

    def __init__(
            self,
            action_size: int,
            buffer_size: int,
            batch_size: int,
            seed: int,
            device: str = "cpu"):
        """
        Fixed-size buffer to store experience tuples.

        @param action_size: dimension of each action
        @param buffer_size: maximum size of buffer
        @param batch_size: size of each training batch
        @param seed: random seed
        @param device: device for calculations
        """
        self.device = device
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.seed = seed
        random.seed(seed)

    def add(self, state: np.array, action: np.array, reward: float, next_state: np.array, done: np.array):
        """
        Add a new experience to memory.

        @param state: previous state
        @param action: executed action
        @param reward: collected reward
        @param next_state: the next state
        """
        e = _Experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """
        Randomly sample a batch of experiences from memory.

        @return: (states, actions, rewards, next_states)
        """
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.stack([e.state for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.stack([e.action for e in experiences if e is not None])).long().to(self.device)
        rewards = torch.from_numpy(np.stack([e.reward for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.stack([e.next_state for e in experiences if e is not None])).float().to(
            self.device)
        dones = torch.from_numpy(np.stack([e.done for e in experiences if e is not None])).float().to(
            self.device)

        return {
            "states": states,
            "actions": actions,
            "rewards": rewards,
            "next_states": next_states,
            "dones": dones
        }

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)


class _Experience:
    def __init__(self, state: np.array, action: np.array, reward: float, next_state: np.array, done: np.array):
        self.done = done
        self.next_state = next_state
        self.next_state = next_state
        self.reward = reward
        self.action = action
        self.state = state
