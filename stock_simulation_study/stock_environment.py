from typing import NoReturn

import numpy as np
import tensorflow as tf

"""
packages for dataload and simulation
"""
import ffn  # data access
from arch.univariate import SkewStudent, EGARCH, ARX, GARCH, StudentsT


class StockEnvironment:
    def __init__(self,
                 prices,
                 fee: float = 0,
                 n_time_points: int = 1000):
        self.fee = fee
        self.agent = None
        self.prices = prices
        self.n_time_points = n_time_points
        self.time = 0
        self.model_setup()
        self.state = []

    def step(self, action):
        # self.state.append(self.sim_prices[self.time])
        self.state.append(self.sim_prices[self.time] - self.sim_prices[self.time - 1])
        self.time += 1
        if self.time > 2:
            # reward = (self.state[-1] - self.state[-2]) * action
            reward = self.state[-1] * action
        else:
            reward = 0
        state = self.state
        return state, reward

    def reset(self):
        reward = 0
        self.simulate_prices()
        self.state = []
        self.time = 0
        for _ in range(100):
            # self.state.append(self.sim_prices[self.time])
            self.state.append(self.sim_prices[self.time] - self.sim_prices[self.time - 1])
            self.time += 1

        return self.state, reward

    def simulate_prices(self) -> NoReturn:
        simulation = self.am.simulate(params=self.am_params, nobs=(self.n_time_points + 100)) / 100
        self.time = 0
        self.state = []

        # self.sim_prices = simulation.data.to_price_index()
        self.sim_prices = [100 + 20 * np.sin(zz) for zz in np.linspace(0, 20, self.n_time_points + 100)]

    def model_setup(self) -> NoReturn:
        returns = self.prices.to_returns().dropna()
        rs = np.random.RandomState([self.time])
        dist = SkewStudent(random_state=rs)
        vol = EGARCH()
        lag = [1, 2, 3, 4, 5]

        # Model calibration
        self.am = ARX(returns * 100, lags=lag, volatility=vol, distribution=dist)
        self.am_params = self.am.fit().params


class StockEnvironmentTwo:
    def __init__(self,
                 prices,
                 price_type: str = "sim_prices",
                 fee: float = 0,
                 n_time_points: int = 1000,
                 kappa: float = 0.1,
                 trade_cost: float = 0.1):
        self.fee = fee
        self.agent = None
        self.prices = prices
        self.n_time_points = n_time_points
        self.time = 0
        self.model_setup()
        self.state = []
        self.price_type = price_type
        self.pos_size = 0
        self.mu = 0
        self.done = False
        self.kappa = kappa
        self.n_steps = 0
        self.trade_cost = trade_cost

    def step(self, action, state_size, mu_zero):
        # self.state.append(self.sim_prices[self.time])
        # self.state.append([self.sim_prices[self.time], self.sim_prices[self.time] - self.sim_prices[self.time-1]])
        if self.time >= len(self.sim_prices) - 1:
            self.done = True

        # self.state = np.array([self.sim_prices[self.time], self.sim_prices[self.time] - self.sim_prices[self.time-1]])
        prices = self.sim_prices[self.time - state_size:self.time]
        returns = np.array(self.sim_prices[self.time - state_size:self.time]) / np.array(
            self.sim_prices[self.time - state_size - 1:self.time - 1]) - 1
        self.state = []
        self.pos_size += (action - 2)
        self.state.append(self.pos_size)
        # for i in range(int((state_size-1)/2)):
        #     self.state.append(prices[i])
        for i in range(int((state_size - 1))):
            self.state.append(returns[i])
        self.state = np.array(self.state)
        self.time += 1
        self.n_steps += 1

        return_step = (prices[-1] / prices[-2] - 1) * self.pos_size

        if self.n_steps == 1:
            # self.mu = self.state[-1] * self.pos_size
            self.mu = return_step
        else:
            # self.mu = self.mu * ((self.n_steps - 1) / self.n_steps) + (self.state[-1] * self.pos_size) / self.n_steps
            self.mu = self.mu * ((self.n_steps - 1) / self.n_steps) + return_step / self.n_steps
        if mu_zero == True:
            mu = 0
        else:
            mu = self.mu

        if self.time > 2:
            trading_cost = self.trade_cost
            # reward = self.state[-1] * self.pos_size - self.kappa * (self.state[-1] * self.pos_size - mu) ** 2 - trading_cost * np.abs((action - 2))
            reward = return_step - self.kappa * (
                    return_step - mu) ** 2 - trading_cost * np.abs((action - 2))
        else:
            reward = 0

        state = self.state
        return state, reward, self.done, False

    def reset(self, state_size):
        self.done = False
        self.simulate_prices()
        self.time = 100
        self.pos_size = 0
        self.n_steps = 0
        # self.mu = 0

        for _ in range(100):

            prices = self.sim_prices[self.time - state_size:self.time]
            # returns = np.array(self.sim_prices[self.time-state_size:self.time]) - np.array(self.sim_prices[self.time-state_size-1:self.time-1])
            returns = np.array(self.sim_prices[self.time - state_size:self.time]) / np.array(
                self.sim_prices[self.time - state_size - 1:self.time - 1]) - 1
            self.state = []
            self.state.append(self.pos_size)
            for i in range(int((state_size - 1))):
                self.state.append(returns[i])

            self.state = np.array(self.state)

            self.time += 1

        # self.state = np.asarray(self.state)
        return self.state  # , reward

    def simulate_prices(self) -> NoReturn:
        self.time = 0
        self.state = []

        if self.price_type == "sim_prices":
            simulation = self.am.simulate(params=self.am_params, nobs=(self.n_time_points + 100)) / 100
            self.sim_prices = simulation.data.to_price_index().values.tolist()
        elif self.price_type == "test_prices":
            self.sim_prices = self.prices[-725:]
            self.sim_prices = self.sim_prices["spy"].to_list()
        else:
            self.sim_prices = [np.sin(zz) for zz in np.linspace(0, 100, self.n_time_points + 100)]

    def model_setup(self) -> NoReturn:
        returns = self.prices.to_returns().dropna()
        rs = np.random.RandomState([1])
        dist = SkewStudent(seed=0)
        vol = GARCH()  # EGARCH()
        lag = [1]  # [1, 2, 3, 4, 5]

        # Model calibration
        self.am = ARX(returns * 100, lags=lag, volatility=vol, distribution=dist)
        self.am_params = self.am.fit().params
