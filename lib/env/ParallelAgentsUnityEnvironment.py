from unityagents import UnityEnvironment
import numpy as np

from lib.env.ParallelAgentsBaseEnvironment import ParallelAgentsBaseEnvironment


class ParallelAgentsUnityEnvironment(ParallelAgentsBaseEnvironment):

    def __init__(
            self,
            name: str,
            target_reward: float,
            env_binary_path: str = 'Reacher_Linux_NoVis/Reacher.x86_64'):
        self.name = name
        self.env = UnityEnvironment(file_name=env_binary_path)
        self.brain_name = self.env.brain_names[0]
        brain = self.env.brains[self.brain_name]

        # reset the environment
        env_info = self.env.reset(train_mode=True)[self.brain_name]

        # number of agents
        num_agents = len(env_info.agents)
        print('Number of agents:', num_agents)

        # size of each action
        action_size = brain.vector_action_space_size
        action_type = brain.vector_action_space_type
        print('Size of each action:', action_size)
        print('Type of each action:', action_type)

        # examine the state space
        states = env_info.vector_observations
        state_size = states.shape[1]
        print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))

        super().__init__(
            state_size=state_size, action_size=action_size, action_type=action_type,
            num_agents=num_agents, target_reward=target_reward, name=name)

    def reset(self) -> np.ndarray:
        env_info = self.env.reset(train_mode=True)[self.brain_name]  # reset the environment
        return env_info.vector_observations  # get the current state

    def act(self, actions: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray):
        env_info = self.env.step(actions)[self.brain_name]  # send the action to the environment
        next_states = env_info.vector_observations  # get next state (for each agent)
        rewards = env_info.rewards  # get the reward
        dones = env_info.local_done  # see if episode has finished

        return next_states, rewards, dones

    def dispose(self):
        self.env.close()
