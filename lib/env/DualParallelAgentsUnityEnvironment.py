from unityagents import UnityEnvironment
import numpy as np

from lib.env.ParallelAgentsBaseEnvironment import ParallelAgentsBaseEnvironment


class DualParallelAgentsUnityEnvironment(ParallelAgentsBaseEnvironment):

    def __init__(
            self,
            name: str,
            target_reward: float,
            env_binary_path: str = 'Reacher_Linux_NoVis/Reacher.x86_64',
            train_mode: bool = True):
        self.train_mode = train_mode
        self.name = name
        self.env = UnityEnvironment(file_name=env_binary_path)
        self.brain_name1 = self.env.brain_names[0]

        self.brain_name2 = self.env.brain_names[1]

        self.brain_names = [self.brain_name1, self.brain_name2]

        num_agents = 0
        action_size = {}
        action_type = {}
        state_size = {}

        env_info = self.env.reset(train_mode=self.train_mode)
        # reset the environment
        for brain_name in self.brain_names:
            # number of agents
            group_num_agents = len(env_info[brain_name].agents)
            print(f'{brain_name}: Number of agents:', group_num_agents)

            # size of each action
            brain = self.env.brains[brain_name]
            for i in range(group_num_agents):
                action_size[num_agents + i] = brain.vector_action_space_size
                action_type[num_agents + i] = brain.vector_action_space_type
            print(f'{brain_name}: Size of each action:', brain.vector_action_space_size)
            print(f'{brain_name}: Type of each action:', brain.vector_action_space_type)

            # examine the state space
            states = env_info[brain_name].vector_observations
            for i in range(group_num_agents):
                state_size[num_agents + i] = states.shape[1]

            num_agents += group_num_agents
            print(
                f'{brain_name}: There are {states.shape[0]} agents. Each observes a state with length: {states.shape[1]}')

        super().__init__(
            state_size=state_size, action_size=action_size, action_type=action_type,
            num_agents=num_agents, target_reward=target_reward, name=name)

    def reset(self) -> np.ndarray:
        env_info = self.env.reset(train_mode=self.train_mode)  # reset the environment

        observations = [env_info[brain_name].vector_observations for brain_name in self.brain_names]

        return np.stack(observations).reshape((self.num_agents, -1))  # get the current state

    def act(self, actions: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray):
        actions = dict(zip(self.brain_names, actions.reshape(len(self.brain_names), -1)))

        env_info = self.env.step(actions)  # send the action to the environment

        next_states = [env_info[brain_name].vector_observations for brain_name in self.brain_names]
        rewards = [env_info[brain_name].rewards for brain_name in self.brain_names]
        dones = [env_info[brain_name].local_done for brain_name in self.brain_names]

        return np.stack(next_states).reshape((self.num_agents, -1)), \
               np.stack(rewards).reshape(self.num_agents), \
               np.stack(dones).reshape(self.num_agents)

    def dispose(self):
        self.env.close()
