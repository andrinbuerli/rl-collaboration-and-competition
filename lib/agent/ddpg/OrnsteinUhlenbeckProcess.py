import numpy as np


class OrnsteinUhlenbeckProcess:
    def __init__(self, theta=0.15, mu=0., sigma=0.15, dt=1e-2, x0=None, size=1, sigma_min=0.05, n_steps_annealing=1000):
        """
        Discretized Ornstein Uhlenbeck process simulation.
        https://planetmath.org/ornsteinuhlenbeckprocess
        This gaussian process that has a bounded variance and admits a stationary probability distribution.
        https://en.wikipedia.org/wiki/Ornstein%E2%80%93Uhlenbeck_process

        @param theta: rate of mean reversion
        @param mu: long term mean of the process
        @param sigma: volatility or average magnitude
        @param dt: discrete temporal step size
        @param x0: initial value of process
        @param size: number of process dimensions
        @param sigma_min: minimal value for sigma
        @param n_steps_annealing: number of time steps until sigma_min is reached
        """

        self.mu = mu
        self.sigma = sigma
        self.n_steps = 0
        self.theta = theta
        self.mu = mu
        self.dt = dt
        self.x0 = x0
        self.x_prev = None
        self.size = size

        if sigma_min is not None:
            self.m = -float(sigma - sigma_min) / float(n_steps_annealing)
            self.c = sigma
            self.sigma_min = sigma_min
        else:
            self.m = 0.
            self.c = sigma
            self.sigma_min = sigma

        self.reset_states()

    def sample(self):
        """
        Sample next time step
        @return: X_t+1
        """
        x = self.x_prev \
            + self.theta * (self.mu - self.x_prev) * self.dt \
            + self.__get_current_sigma() * np.sqrt(self.dt) * np.random.normal(size=self.size)
        self.x_prev = x
        self.n_steps += 1
        return x

    def reset_states(self):
        """
        Reset the process
        """
        self.x_prev = self.x0 if self.x0 is not None else np.zeros(self.size)
        self.n_steps = 0

    def __get_current_sigma(self):
        sigma = max(self.sigma_min, self.m * float(self.n_steps) + self.c)
        return sigma