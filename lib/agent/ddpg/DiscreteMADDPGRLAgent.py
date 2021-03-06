import numpy as np
from typing import Callable
import torch
import torch.nn as nn

from lib.agent.BaseRLAgent import BaseRLAgent
from lib.models.policy import DeterministicDiscretePolicy
from lib.models.function import StateActionValueFunction
from lib.agent.ddpg.OrnsteinUhlenbeckProcess import OrnsteinUhlenbeckProcess
from lib.agent.ddpg.ReplayBuffer import ReplayBuffer


class __SingleDDPGAgent__:
    def __init__(
            self,
            state_size: int,
            action_size: int,
            get_actor: Callable[[int, int], type(DeterministicDiscretePolicy)],
            get_critic: Callable[[int, int], type(StateActionValueFunction)],
            get_optimizer: Callable[[nn.Module], type(torch.optim.Optimizer)],
            device: str = "cpu"):
        self.critic_local = get_critic(state_size, action_size).to(device)
        self.critic_target = get_critic(state_size, action_size).to(device)

        self.actor_local = get_actor(state_size, action_size).to(device)
        self.actor_target = get_actor(state_size, action_size).to(device)

        self.critic_optimizer = get_optimizer(self.critic_local)
        self.actor_optimizer = get_optimizer(self.actor_local)


class DiscreteMADDPGRLAgent(BaseRLAgent):

    def __init__(
            self,
            get_actor: Callable[[int, int], type(DeterministicDiscretePolicy)],
            get_critic: Callable[[int, int], type(StateActionValueFunction)],
            state_size: [int],  # per agent
            action_size: [int],  # per agent
            seed: int = 0,
            buffer_size: int = int(1e5),
            replay_min_size: int = int(1e4),
            batch_size: int = 64,
            gamma: float = 0.99,
            tau: float = 1e-3,
            lr: float = 5e-4,
            update_every: int = 4,
            update_for: int = 4,
            prioritized_exp_replay: bool = False,
            prio_a: float = 0.7,
            prio_b_init: float = 0.5,
            prio_b_growth: float = 1.1,
            epsilon: float = 1.0,
            epsilon_decay: float = .9,
            epsilon_min: float = .01,
            grad_clip_max: float = None,
            n_agents: int = 2,
            device="cpu"
    ):
        """
        Deep deterministic policy gradients (DDPG) agent.
        https://arxiv.org/pdf/1509.02971.pdf

        @param get_actor:
        @param get_critic:
        @param state_size: dimension of each state
        @param action_size: dimension of each action
        @param seed: random seed
        @param buffer_size: replay buffer size
        @param batch_size: minibatch size
        @param gamma: discount factor
        @param tau: for soft update of target parameters, ??_target = ??*??_local + (1 - ??)*??_target
        @param lr: learning rate
        @param update_every: how often to update the network, after every n step
        @param update_for: how many minibatches should be sampled at every update step
        @param prioritized_exp_replay: use prioritized experience replay
        @param prio_a: a = 0 uniform sampling, a = 1 fully prioritized sampling
        @param prio_b_init: importance sampling weight init
        @param prio_b_growth: importance sampling weight growth (will grow to max of 1)
        @param epsilon:
        @param epsilon_decay:
        @param epsilon_min:
        @param device: the device on which the calculations are to be executed
        """
        self.n_agents = n_agents
        self.replay_min_size = replay_min_size
        self.grad_clip_max = grad_clip_max
        self.epsilon_min = epsilon_min
        self.update_for = update_for
        self.eps = epsilon
        self.eps_decay = epsilon_decay
        self.state_size = state_size
        self.action_size = action_size
        self.prio_b_growth = prio_b_growth
        self.prio_b = prio_b_init
        self.prio_a = prio_a
        self.prioritized_exp_replay = prioritized_exp_replay
        self.update_every = update_every
        self.lr = lr
        self.tau = tau
        self.gamma = gamma
        self.batch_size = batch_size
        self.buffer_size = buffer_size

        super(DiscreteMADDPGRLAgent, self).__init__(models=[], device=device, learning_rate=lr, model_names=[])

        self.agents = [
            __SingleDDPGAgent__(state_size=s, action_size=a, get_actor=get_actor, get_critic=get_critic,
                                get_optimizer=self._get_optimizer, device=device)
            for s, a in zip(state_size, action_size)
        ]

        self.models = [[x.actor_local, x.actor_target, x.critic_local, x.critic_target] for x in self.agents]
        self.model_names = [[f"agent{i}-actor_local", f"agent{i}-actor_target", f"agent{i}-critic_local",
                             f"agent{i}-critic_target"] for i in range(self.n_agents)]
        self.models = [model for models in self.models for model in models]
        self.model_names = [name for names in self.model_names for name in names]

        # Replay memory
        self.memory = ReplayBuffer(action_size, self.buffer_size, self.batch_size, seed=seed, device=device)
        # Initialize time step (for updating every self.update_every steps)
        self.t_step = 0

        self.loss = None
        self.actor_loss = None
        self.critic_loss = None

    def act(self, states: np.ndarray, eps: float = None, training: int = 1, action_lower_bound=-1,
            action_upper_bound=1) -> (np.ndarray, np.ndarray):
        """
        Determine next action based on current states

        @param states: the current states
        @param training: binary integer indicating weather or not noise is incorporated into the action
        @param action_upper_bound: clip action upper bound
        @param action_lower_bound: clip action lower bound
        @return: the clipped actions, the action logits (zeros for this agent),
                 the log_probabilities of the actions (zeros for this agent)
        """

        states = torch.from_numpy(states).float().to(self.device)

        [agent.actor_local.eval() for agent in self.agents]
        with torch.no_grad():
            actions = [agent.actor_local(obs) for obs, agent in list(zip(states, self.agents))]
        [agent.actor_local.train() for agent in self.agents]

        if np.random.rand() < (eps if eps is not None else self.eps):
            actions = np.array([np.random.choice(a_size) for a_size in self.action_size])
        else:
            actions = np.array([a.argmax().cpu().detach().numpy() for a in actions])

        return {
            "actions": actions,
            "action_logits": np.zeros_like(actions),
            "log_probs": np.zeros_like(actions)
        }

    def learn(
            self,
            states: np.ndarray,
            actions: np.ndarray,
            action_logits: np.ndarray,
            action_log_probs: np.ndarray,
            rewards: np.ndarray,
            next_states: np.ndarray,
            dones: np.ndarray):
        self.t_step += 1
        # Save experience in replay memory
        for s_t, a_t, r_t1, s_t1, d_t1 in list(zip(
                np.transpose(states, axes=[1, 0, 2]), np.transpose(actions, axes=[1, 0]),
                np.transpose(rewards, axes=[1, 0]), np.transpose(next_states, axes=[1, 0, 2]),
                np.transpose(dones, axes=[1, 0]))):
            self.memory.add(s_t, a_t, r_t1, s_t1, d_t1)

        if len(self.memory) > self.replay_min_size and (self.t_step + 1) % self.update_every == 0:
            for _ in range(self.update_for):
                # If enough samples are available in memory, get random subset and learn
                experiences = self.memory.sample()

                self.__learn(experiences)

                for agent in self.agents:
                    self.__soft_update(agent.critic_local, agent.critic_target)
                    self.__soft_update(agent.actor_local, agent.actor_target)

            self.eps = max(self.eps * self.eps_decay, self.epsilon_min)

    def get_name(self) -> str:
        return f"DiscreteMADDPG"

    def reset(self):
        pass

    def get_log_dict(self) -> dict:
        return {
            "epsilon": self.eps,
            "replay_buffer_size": len(self.memory),
            "critic_loss": self.critic_loss if self.critic_loss is not None else 0.0,
            "actor_loss": self.actor_loss if self.actor_loss is not None else 0.0,
            "loss": self.loss if self.loss is not None else 0.0
        }

    def __learn(self, experiences):
        states = experiences["states"]
        actions = experiences["actions"]
        rewards = experiences["rewards"]
        next_states = experiences["next_states"]
        dones = experiences["dones"]

        obs_full = states.reshape((states.shape[0], -1))
        next_obs_full = next_states.reshape((next_states.shape[0], -1))

        actor_losses = []
        critic_losses = []

        batch_size = states.shape[0]
        for i_agent in range(self.n_agents):
            agent = self.agents[i_agent]

            target_actions = [
                torch.nn.functional.one_hot(agent.actor_target(obs).argmax(dim=1), num_classes=a)
                for obs, agent, a in list(zip(next_states.transpose(dim0=0, dim1=1), self.agents, self.action_size))
            ]
            # flatten actions
            target_actions = torch.stack([torch.cat([l[i] for l in target_actions]) for i in range(batch_size)])

            with torch.no_grad():
                q_next = agent.critic_target(next_obs_full, target_actions)

            q_target = rewards[:, i_agent].view(-1, 1) + self.gamma * q_next * (1 - dones[:, i_agent].view(-1, 1))
            one_hot_actions = \
                [torch.nn.functional.one_hot(a, num_classes=a_s) for a, a_s in list(zip(actions.T, self.action_size))]

            flat_one_hot_actions = torch.stack([torch.cat([l[i] for l in one_hot_actions]) for i in range(batch_size)])

            q_values = agent.critic_local(obs_full, flat_one_hot_actions)

            critic_loss = ((q_values - q_target) ** 2).mean()

            agent.critic_optimizer.zero_grad()
            critic_loss.backward()
            if self.grad_clip_max is not None:
                torch.nn.utils.clip_grad_norm_(agent.critic_local.parameters(), self.grad_clip_max)
            agent.critic_optimizer.step()

            action = [
                agent.actor_local(obs) if i == i_agent\
                    else agent.actor_local(obs)
                for i, (obs, agent, a_s) in enumerate(zip(states.transpose(dim0=0, dim1=1), self.agents, self.action_size))
            ]

            actions_one_hot = torch.stack([torch.cat([l[i] for l in action]) for i in range(batch_size)]).to(self.device)

            actor_loss = -agent.critic_local(obs_full, actions_one_hot).mean()

            agent.actor_optimizer.zero_grad()
            actor_loss.backward()
            if self.grad_clip_max is not None:
                torch.nn.utils.clip_grad_norm_(agent.actor_local.parameters(), self.grad_clip_max)
            agent.actor_optimizer.step()

            actor_losses.append(actor_loss.detach().cpu().numpy())
            critic_losses.append(critic_loss.detach().cpu().numpy())

        self.critic_loss = np.mean(critic_losses)
        self.actor_loss = np.mean(actor_losses)

        self.loss = self.critic_loss + self.actor_loss

    def __soft_update(self, local_model, target_model):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)
