from collections import deque
import numpy as np
from typing import Union, List
import shutil
from glob import glob
import os

from lib.agent.BaseRLAgent import BaseRLAgent
from lib.env.ParallelAgentsBaseEnvironment import ParallelAgentsBaseEnvironment
from lib.log.BaseLogger import BaseLogger


class RLAgentTrainer:

    def __init__(
            self,
            agent: BaseRLAgent,
            env: ParallelAgentsBaseEnvironment,
            seed: int,
            logger: BaseLogger = None,
            agent_save_dir="agents"):
        """
        Initialize the RLAgentTrainer

        @param agent: the agent to be trained
        @param env:  the (continuous) environment to sample the trajectories from
        @param logger: the logger to monitor the training progress
        """

        self.agent_save_dir = agent_save_dir
        self.seed = seed
        self.logger = logger
        self.env = env
        self.agent = agent
        self.scores = []  # list containing scores from each episode
        self.scores_window = deque(maxlen=100)  # last 100 scores

        self.states = None
        self.trajectory_scores = None
        self.t_sampled = None

        if not os.path.exists(self.agent_save_dir):
            os.mkdir(self.agent_save_dir)

    def train(
            self,
            n_iterations: int,
            max_t: Union[List[int], int],
            max_t_iteration:  Union[List[int], int] = None,
            intercept: bool = False,
            t_max_episode: int = 1e3):
        """
        RL agent training

        @param n_iterations: number of training iterations
        @param max_t: the number of time steps per sampled trajectory
        @param max_t_iteration: the iteration number at which the corresponding max_t should be used
        """

        max_t_original = max_t

        max_mean_score = None
        for i_iter in range(1, n_iterations + 1):
            if type(max_t_original) is list:
                max_t_index = min([i for i, tresh in enumerate(max_t_iteration) if tresh >= i_iter] + [len(max_t_original) - 1])
                max_t = max_t_original[max_t_index]

            trajectory = self.__collect_trajectories(max_t=max_t, intercept=intercept, t_max_episode=t_max_episode)

            self.agent.learn(
                states=trajectory["states"], action_logits=trajectory["action_logits"],
                action_log_probs=trajectory["log_probs"], rewards=trajectory["rewards"],
                next_states=trajectory["next_states"], actions=trajectory["actions"],
                dones=trajectory["dones"])

            score_window_mean = np.mean(self.scores_window)

            dir_name = f'{self.env.name}-{self.agent.get_name()}-{self.seed}-latest'
            self.agent.save(directory_name=os.path.join(self.agent_save_dir, dir_name))

            if max_mean_score is None or max_mean_score < score_window_mean:
                max_mean_score = score_window_mean if not np.isnan(score_window_mean) else max_mean_score
                directories = list(glob(f"{self.agent_save_dir}/*{self.seed}*-best-*"))
                if len(directories) > 0:
                    shutil.rmtree(directories[0])

                self.__save_agent(i_iter, score_window_mean)

            print('\rIteration {}\tAverage Score: {:.2f}'.format(i_iter, score_window_mean), end="")
            if i_iter % 100 == 0:
                print('\rIteration {}\tAverage Score: {:.2f}'.format(i_iter, score_window_mean))
            if score_window_mean >= self.env.target_reward:
                print('\nEnvironment solved in {:d} iterations!\tAverage Score: {:.2f}'.format(i_iter - 100,
                                                                                               score_window_mean))
                self.__save_agent(i_iter, score_window_mean)
                break

        score_window_mean = np.mean(self.scores_window)
        if score_window_mean < self.env.target_reward:
            print('\nEnvironment was not solved in {:d} iterations!\tAverage Score: {:.2f}'.format(n_iterations - 99,
                                                                                                   score_window_mean))
            self.__save_agent(i_iter, score_window_mean)

    def __save_agent(self, i_iter, score_window_mean):
        dir_name = f'{self.env.name}-{self.agent.get_name()}-{self.seed}-best-{round(score_window_mean, 2)}'
        self.agent.save(directory_name=os.path.join(self.agent_save_dir, dir_name))

    def __collect_trajectories(self, max_t: int, intercept: bool = False, t_max_episode: int = 1e3):
        """
        Sample trajectories from the environment

        @param max_t: max time steps
        @return: (
                states [trajectory, time_step, state_size], actions  [trajectory, time_step, action_size],
                action_logits  [trajectory, time_step, action_size], log_probs  [trajectory, time_step],
                rewards  [trajectory, time_step], next_states [trajectory, time_step, state_size]
            )
        """

        if not intercept or self.states is None:
            self.agent.reset()
            self.states = self.env.reset()
            self.trajectory_scores = np.zeros(self.env.num_agents)
            self.t_sampled = 0

        s_t0, a_t0, al_t0, pa_t0, r_t1, s_t1, d = ([] for _ in range(7))

        t = 0
        while True:
            pred = self.agent.act(self.states)
            next_states, rewards, dones = self.env.act(pred["actions"])

            any_nan_rewards = np.isnan(rewards).any()

            if any_nan_rewards:
                print("!!!!! WARNING NAN rewards detected, penalizing agent !!!!!")
                rewards = np.nan_to_num(rewards, nan=-5.0)  # NAN penalty
                if len(s_t0) > 0:
                    break
                else:
                    self.agent.reset()
                    self.states = self.env.reset()
                    self.trajectory_scores = np.zeros(self.env.num_agents)
                    self.t_sampled = 0


            s_t0.append(self.states), a_t0.append(pred["actions"]), al_t0.append(pred["action_logits"]),\
            pa_t0.append(pred["log_probs"]), r_t1.append(rewards), s_t1.append(next_states), d.append(dones)

            if any_nan_rewards:
                break

            self.states = next_states
            self.trajectory_scores = self.trajectory_scores * (1 - np.array(dones)) + rewards
            t += 1
            if t >= max_t or np.all(dones):
                if np.all(dones) or t >= t_max_episode\
                        or (intercept and self.t_sampled + t >=t_max_episode):
                    if intercept:
                        self.__log_and_metrics(self.t_sampled + t)

                        self.agent.reset()
                        self.states = self.env.reset()
                        self.trajectory_scores = np.zeros(self.env.num_agents)
                        self.t_sampled = 0

                    break

                if intercept:
                    break

        if not intercept:
            self.__log_and_metrics(t)
        else:
            self.t_sampled += t

        return {
            "states": np.transpose(np.array(s_t0), axes=[1, 0, 2]),
            "actions": np.transpose(np.array(a_t0), axes=[1, 0, 2]),
            "action_logits": np.transpose(np.array(al_t0), axes=[1, 0, 2]),
            "log_probs": np.transpose(np.array(pa_t0), axes=[1, 0]),
            "rewards": np.transpose(np.array(r_t1), axes=[1, 0]),
            "next_states": np.transpose(np.array(s_t1), axes=[1, 0, 2]),
            "dones": np.transpose(np.array(d), axes=[1, 0])
        }

    def __log_and_metrics(self, t_sampled):
        max_score = self.trajectory_scores.max()
        self.scores_window.append(max_score)  # save most recent score
        self.scores.append(max_score)  # save most recent score
        if self.logger is not None:
            self.logger.log({
                "reward": max_score,
                "trajectory_length": t_sampled,
                **self.agent.get_log_dict()
            })
