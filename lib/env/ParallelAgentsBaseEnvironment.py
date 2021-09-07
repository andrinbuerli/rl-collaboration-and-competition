import numpy as np
import abc


class ParallelAgentsBaseEnvironment:

    def __init__(
            self,
            name: str,
            state_size: int,
            action_size: int,
            action_type: str,
            num_agents: int,
            target_reward: float):
        self.name = name
        self.target_reward = target_reward
        self.num_agents = num_agents
        self.action_type = action_type
        self.action_size = action_size
        self.state_size = state_size

    @abc.abstractmethod
    def reset(self) -> np.ndarray:
        """
        Reset the environment

        @return: the states of all agents in the current environment
        """
        pass

    @abc.abstractmethod
    def act(self, actions: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray):
        """
        Execute given actions in the environment

        @param actions: action to take for each agent in the current environment
        @return: next_states, rewards, dones (binary)
        """
        pass

    @abc.abstractmethod
    def dispose(self):
        """
        Dispose environment
        """
        pass
