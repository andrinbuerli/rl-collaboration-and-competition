import numpy as np
from typing import Callable
import torch

from lib.agent.BaseRLAgent import BaseRLAgent
from lib.models.policy import DeterministicBasePolicy
from lib.models.function import StateActionValueFunction
from lib.agent.ddpg.OrnsteinUhlenbeckProcess import OrnsteinUhlenbeckProcess
from lib.agent.ddpg.ReplayBuffer import ReplayBuffer


class DDPGRLAgent(BaseRLAgent):

    def __init__(
            self,
            get_actor: Callable[[], type(DeterministicBasePolicy)],
            get_critic: Callable[[], type(StateActionValueFunction)],
            state_size: int,
            action_size: int,
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
        @param tau: for soft update of target parameters, θ_target = τ*θ_local + (1 - τ)*θ_target
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

        self.qnetwork_local = get_critic().to(device)
        self.qnetwork_target = get_critic().to(device)

        self.argmaxpolicy_local = get_actor().to(device)
        self.argmaxpolicy_target = get_actor().to(device)

        self.random_process = OrnsteinUhlenbeckProcess(size=self.action_size)

        super(DDPGRLAgent, self).__init__(
            models=[self.qnetwork_local, self.qnetwork_target, self.argmaxpolicy_local, self.argmaxpolicy_target],
            device=device,
            learning_rate=lr,
            model_names=["qnetwork_local", "qnetwork_target", "policy_local", "policy_target"])

        self.qnetwork_optimizer = self._get_optimizer(self.qnetwork_local.parameters())
        self.argmaxpolicy_optimizer = self._get_optimizer(self.argmaxpolicy_local.parameters())

        # Replay memory
        if self.prioritized_exp_replay:
            self.memory = PrioritizedReplayBuffer(action_size, self.buffer_size, self.batch_size, seed=seed, peps=1e-4,
                                                  device=device)
        else:
            self.memory = ReplayBuffer(action_size, self.buffer_size, self.batch_size, seed=seed, device=device)
        # Initialize time step (for updating every self.update_every steps)
        self.t_step = 0

        self.loss = None
        self.policy_gradients = None
        self.critic_loss = None

    def act(self, states: np.ndarray, eps: float = None, training: int = 1, action_lower_bound=-1, action_upper_bound=1) -> (np.ndarray, np.ndarray):
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

        self.argmaxpolicy_local.eval()
        with torch.no_grad():
            actions = self.argmaxpolicy_local(states)
        self.argmaxpolicy_local.train()

        actions = actions.detach().cpu().numpy()

        if np.random.rand() < (eps if eps is not None else self.eps):
            actions += training * self.random_process.sample()

        actions = np.clip(actions, -1., 1.)

        return {
            "actions": actions,
            "action_logits": np.zeros_like(actions),
            "log_probs": np.zeros((actions.shape[0]))
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
        for state, action, reward, next_state, done in zip(states.reshape(states.shape[0] * states.shape[1], -1),
                                                     actions.reshape(states.shape[0] * states.shape[1], -1),
                                                     rewards.reshape(states.shape[0] * states.shape[1], -1),
                                                     next_states.reshape(states.shape[0] * states.shape[1], -1),
                                                     dones.reshape(states.shape[0] * states.shape[1], -1)):
            self.memory.add(state, action, reward, next_state, done)

        if len(self.memory) > self.replay_min_size:
            for _ in range(self.update_for):
                # If enough samples are available in memory, get random subset and learn
                experiences = self.memory.sample()

                self.__learn(experiences)

            if (self.t_step + 1) % self.update_every == 0:
                self.__soft_update(self.qnetwork_local, self.qnetwork_target)
                self.__soft_update(self.argmaxpolicy_local, self.argmaxpolicy_target)

            self.eps = max(self.eps * self.eps_decay, self.epsilon_min)

    def get_name(self) -> str:
        return "DDPG"

    def reset(self):
        return self.random_process.reset_states()

    def get_log_dict(self) -> dict:
        return {
            "epsilon": self.eps,
            "replay_buffer_size": len(self.memory),
            "critic_loss": self.critic_loss.detach().cpu().numpy() if self.critic_loss is not None else 0.0,
            "actor_loss": self.policy_gradients.detach().cpu().numpy() if self.policy_gradients is not None else 0.0,
            "loss": self.loss.detach().cpu().numpy() if self.loss is not None else 0.0
        }

    def __learn(self, experiences):
        states = experiences["states"]
        actions = experiences["actions"]
        rewards = experiences["rewards"]
        next_states = experiences["next_states"]
        dones = experiences["dones"]

        next_best_actions = self.argmaxpolicy_target.forward(next_states)
        q_values_next_state = self.qnetwork_target.forward(next_states, next_best_actions)

        q_targets = rewards + (1 - dones) * self.gamma * q_values_next_state
        q_values = self.qnetwork_local.forward(states, actions)
        td_error = q_values - q_targets

        self.critic_loss = (td_error ** 2).mean()

        self.qnetwork_optimizer.zero_grad()
        self.critic_loss.backward()
        if self.grad_clip_max is not None:
            torch.nn.utils.clip_grad_norm_(self.qnetwork_local.parameters(), self.grad_clip_max)

        self.qnetwork_optimizer.step()

        actions = self.argmaxpolicy_local(states)
        self.policy_gradients = -self.qnetwork_local(states, actions).mean()

        self.argmaxpolicy_optimizer.zero_grad()
        self.policy_gradients.backward()
        if self.grad_clip_max is not None:
            torch.nn.utils.clip_grad_norm_(self.argmaxpolicy_local.parameters(), self.grad_clip_max)

        self.argmaxpolicy_optimizer.step()

        self.loss = self.critic_loss + self.policy_gradients

    def __soft_update(self, local_model, target_model):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)
