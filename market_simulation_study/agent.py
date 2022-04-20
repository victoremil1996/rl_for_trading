import os
import sys
from copy import deepcopy

from typing import NoReturn, Tuple
from numpy import ndarray
import abc
import numpy as np
import pandas as pd
import random
# Tensorflow
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN, LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import math

# Pytorch
import torch
from torch.optim import Optimizer
from torch.optim import Adam
from torch import nn
from torch.distributions import Uniform
from torch.distributions import Normal, Binomial
from torch.nn import functional as f


class Agent(abc.ABC):
    """
    Abstract class for agents
    """

    # @abc.abstractmethod
    # def __init__(self,
    #              agent_id: int = None,
    #              delta: float = None,
    #              position: int = 0,
    #              pnl: float = None,
    #              buy_price: float = None,
    #              sell_price: float = None,
    #              all_trades: np.ndarray = None):
    #     """
    #     Constructor
    #     :param latency: latency when matching agents in the market environment
    #     """
    #
    #     self.agent_id = agent_id
    #     self.delta = delta
    #     self.latency = delta
    #     self.position = position
    #     self.pnl = pnl
    #     self.buy_price = buy_price
    #     self.sell_price = sell_price
    #     self.buy_volume = buy_volume
    #     self.sell_volume = sell_volume
    #     self.all_trades = all_trades
    #     self.buy_order = pd.DataFrame(np.array([[self.buy_price, self.buy_volume, self.latency, self.agent_id]]),
    #                                   columns = ["buy_price", "buy_volume", "latency", "agent_id"])
    #     self.sell_order = pd.DataFrame(np.array([[self.sell_price, self.sell_volume, self.latency, self.agent_id]]),
    #                                   columns = ["sell_price", "sell_volume", "latency", "agent_id"])
    @abc.abstractmethod
    def calculate_buy_price(self, state: dict) -> float:
        """
        Calculates buy price

        :param state:
        :return:
        """
        raise NotImplementedError("Abstract Class")

    @abc.abstractmethod
    def calculate_sell_price(self, state: dict) -> float:
        """
        Calculates sell price

        :param state:
        :return:
        """
        raise NotImplementedError("Abstract Class")

    @abc.abstractmethod
    def calculate_buy_volume(self, state: dict) -> float:
        """
        Calculates buy price

        :param state:
        :return:
        """
        raise NotImplementedError("Abstract Class")

    @abc.abstractmethod
    def calculate_sell_volume(self, state: dict) -> float:
        """
        Calculates sell price

        :param state:
        :return:
        """
        raise NotImplementedError("Abstract Class")

    @abc.abstractmethod
    def calculate_profit_and_loss(self, state: dict) -> float:
        """
        Calculates profit and loss

        :param state:
        :return:
        """
        raise NotImplementedError("Abstract Class")

    @abc.abstractmethod
    def update(self, state: dict) -> NoReturn:
        """
        Updates agents ask and bid prices, and corresponding volumes, when new state is provided

        :param state:
        :return:
        """
        # Update trades and position

        # Update prices


class RandomAgent(Agent):
    """
    Agent which makes noisy buy and sell prices around market_prices
    """

    def __init__(self,
                 agent_id: int = None,
                 delta: float = None,
                 position: int = 0,
                 pnl: float = None,
                 buy_price: float = None,
                 sell_price: float = None,
                 all_trades: np.ndarray = None,
                 buy_volume: float = None,
                 sell_volume: float = None,
                 noise_range: Tuple = None,
                 mid_price_noise: float = 0.001):
        """
        Constructor
        :param latency: latency when matching agents in the market environment
        """
        self.agent_class = "Random"
        self.agent_id = agent_id
        self.delta = delta
        self.latency = delta
        self.position = position
        self.pnl = pnl
        self.buy_price = buy_price
        self.sell_price = sell_price
        self.all_trades = all_trades if all_trades else np.array([0, 0])
        self.buy_volume = buy_volume
        self.sell_volume = sell_volume
        self.mid_price_noise = mid_price_noise
        self.noise_range = noise_range if noise_range else [0.001, 0.005]
        self.random_agent_price = None
        self.buy_order = pd.DataFrame(np.array([[self.buy_price, self.buy_volume, self.latency, self.agent_id]]),
                                      columns=["buy_price", "buy_volume", "latency", "agent_id"])
        self.sell_order = pd.DataFrame(np.array([[self.sell_price, self.sell_volume, self.latency, self.agent_id]]),
                                       columns=["sell_price", "sell_volume", "latency", "agent_id"])

    def calculate_buy_price(self, state: dict) -> float:
        """
        Calculates buy price

        :param state: market state information
        :return: buy price
        """
        buy_price = self.random_agent_price * (
                    1 - np.random.uniform(low=self.noise_range[0], high=self.noise_range[1]))
        buy_price = np.maximum(buy_price, 0)
        return buy_price

    def calculate_sell_price(self, state: dict) -> float:
        """
        Calculates sell price

        :param state: market state information
        :return: sell price
        """
        sell_price = self.random_agent_price * (
                    1 + np.random.uniform(low=self.noise_range[0], high=self.noise_range[1]))
        sell_price = np.maximum(sell_price, 0)
        return sell_price

    def calculate_buy_volume(self, state: dict) -> float:
        """
        Calculates buy price

        :param state:
        :return:
        """
        volume = np.random.binomial(2, 0.51)

        return volume

    def calculate_sell_volume(self, state: dict) -> float:
        """
        Calculates sell price

        :param state:
        :return:
        """
        volume = np.random.binomial(2, 0.5)

        return volume

    def calculate_profit_and_loss(self, state: dict) -> NoReturn:
        """
        Calculates profit and loss

        :param state: market state information
        :return: total profit and loss
        """
        realized_value = np.sum(self.all_trades[:, 0] * self.all_trades[:, 1])
        unrealized_value = self.position * state["market_prices"][-1] * (1 - state["slippage"])

        self.pnl = realized_value + unrealized_value

    def update(self, state: dict) -> NoReturn:
        """
        Updates agents ask and bid prices, and corresponding volumes, when new state is provided

        :param state: market state information
        :return: NoReturn
        """
        # Update latency

        self.latency = self.delta + np.random.uniform(1e-6, 1)

        # Update prices and volume
        self.random_agent_price = state["market_prices"][-1] * (1 + np.random.normal(loc=0, scale=self.mid_price_noise))

        self.buy_price = self.calculate_buy_price(state)
        self.sell_price = self.calculate_sell_price(state)
        self.buy_volume = self.calculate_buy_volume(state)
        self.sell_volume = self.calculate_sell_volume(state)

        # Update orders

        self.buy_order = pd.DataFrame(np.array([[self.buy_price, self.buy_volume, self.latency, self.agent_id]]),
                                      columns=["buy_price", "buy_volume", "latency", "agent_id"],
                                      index=[self.agent_id])
        self.sell_order = pd.DataFrame(np.array([[self.sell_price, self.sell_volume, self.latency, self.agent_id]]),
                                       columns=["sell_price", "sell_volume", "latency", "agent_id"],
                                       index=[self.agent_id])


class InvestorAgent(Agent):
    """
    Agent which buys or sells large orders in chunks over small periods. Similiar to Institutional Investors.
    """

    def __init__(self,
                 agent_id: int = None,
                 delta: float = None,
                 position: int = 0,
                 pnl: float = None,
                 buy_price: float = None,
                 sell_price: float = None,
                 all_trades: np.ndarray = None,
                 buy_volume: float = 5,
                 sell_volume: float = 10,
                 intensity: float = None,
                 n_orders: int = 20,
                 orders_in_queue: int = 0,
                 buy_price_margin: float = 0.025,
                 sell_price_margin: float = 0.05,
                 is_buying: bool = False,
                 can_short: bool = False):
        """
        Constructor
        :param latency: latency when matching agents in the market environment
        """
        self.agent_class = "Investor"
        self.agent_id = agent_id
        self.delta = delta
        self.latency = delta
        self.position = position
        self.pnl = pnl
        self.buy_price = buy_price
        self.sell_price = sell_price
        self.all_trades = all_trades if all_trades else np.array([0, 0])
        self.buy_volume = buy_volume
        self.sell_volume = sell_volume
        self.intensity = intensity
        self.n_orders = n_orders
        self.orders_in_queue = orders_in_queue
        self.buy_price_margin = buy_price_margin
        self.sell_price_margin = sell_price_margin
        self.is_buying = is_buying
        self.can_short = can_short
        self.buy_order = pd.DataFrame(np.array([[self.buy_price, self.buy_volume, self.latency, self.agent_id]]),
                                      columns=["buy_price", "buy_volume", "latency", "agent_id"],
                                      index=[self.agent_id])
        self.sell_order = pd.DataFrame(np.array([[self.sell_price, self.sell_volume, self.latency, self.agent_id]]),
                                       columns=["sell_price", "sell_volume", "latency", "agent_id"],
                                       index=[self.agent_id])

    def calculate_buy_price(self, state: dict) -> float:
        """
        Calculates buy price

        :param state: market state information
        :return: buy price
        """
        buy_price = state["market_prices"][-1] * (1 + self.buy_price_margin)
        buy_price = np.maximum(buy_price, 0)
        return buy_price

    def calculate_sell_price(self, state: dict) -> float:
        """
        Calculates sell price

        :param state: market state information
        :return: sell price
        """
        sell_price = state["market_prices"][-1] * (1 - self.sell_price_margin)
        sell_price = np.maximum(sell_price, 0)
        return sell_price

    def calculate_buy_volume(self, state: dict) -> float:
        """
        Calculates buy price

        :param state:
        :return:
        """
        volume = self.buy_volume

        return volume

    def calculate_sell_volume(self, state: dict) -> float:
        """
        Calculates sell price

        :param state:
        :return:
        """
        volume = self.sell_volume

        return volume

    def calculate_profit_and_loss(self, state: dict) -> NoReturn:
        """
        Calculates profit and loss

        :param state: market state information
        :return: total profit and loss
        """
        realized_value = np.sum(self.all_trades[:, 0] * self.all_trades[:, 1])
        unrealized_value = self.position * state["market_prices"][-1] * (1 - state["slippage"])

        self.pnl = realized_value + unrealized_value

    def update(self, state: dict) -> NoReturn:
        """
        Updates agents ask and bid prices, and corresponding volumes, when new state is provided

        :param state: market state information
        :return: NoReturn
        """
        # Update latency
        self.latency = self.delta + np.random.uniform(1 + 1e-6, 2)

        # instantiate no prices
        self.buy_price = np.nan
        self.sell_price = np.nan

        # check if investor wants to buy or sell and calculate prices if so
        will_buy = np.random.uniform(0, 1) < self.intensity
        if self.can_short:
            sell_is_possible = True
        else:
            sell_is_possible = (self.position >= self.n_orders * self.sell_volume)
        will_sell = (np.random.uniform(0, 1) < self.intensity) and sell_is_possible

        if self.orders_in_queue == 0:
            self.is_buying = False

        if self.orders_in_queue > 0 and self.is_buying:  # buying
            self.orders_in_queue -= 1
            self.buy_price = self.calculate_buy_price(state)

        elif self.orders_in_queue > 0 and not self.is_buying:  # selling
            self.orders_in_queue -= 1
            self.sell_price = self.calculate_sell_price(state)

        elif will_buy and self.orders_in_queue == 0:  # starts to buy
            self.orders_in_queue = self.n_orders - 1
            self.is_buying = True
            self.buy_price = self.calculate_buy_price(state)

        elif will_sell and self.orders_in_queue == 0:  # starts to sell
            self.orders_in_queue = int(self.n_orders/2) - 1
            self.sell_price = self.calculate_sell_price(state)

        # Update volume

        self.buy_volume = self.calculate_buy_volume(state)
        self.sell_volume = self.calculate_sell_volume(state)

        # Update orders
        self.buy_order = pd.DataFrame(np.array([[self.buy_price, self.buy_volume, self.latency, self.agent_id]]),
                                      columns=["buy_price", "buy_volume", "latency", "agent_id"],
                                      index=[self.agent_id])
        self.sell_order = pd.DataFrame(np.array([[self.sell_price, self.sell_volume, self.latency, self.agent_id]]),
                                       columns=["sell_price", "sell_volume", "latency", "agent_id"],
                                       index=[self.agent_id])


class TrendAgent(Agent):
    """
    Agent which makes noisy buy and sell prices around market_prices
    """

    def __init__(self,
                 agent_id: int = None,
                 delta: float = None,
                 position: int = 0,
                 pnl: float = None,
                 buy_price: float = None,
                 sell_price: float = None,
                 all_trades: np.ndarray = None,
                 buy_volume: float = 0,
                 sell_volume: float = 0,
                 price_margin: float = 0.05,
                 const_position_size: int = 5,
                 moving_average_one: int = 25,
                 moving_average_two: int = 100):
        """
        Constructor
        :param latency: latency when matching agents in the market environment
        """
        self.agent_class = "Trend"
        self.agent_id = agent_id
        self.delta = delta
        self.latency = delta
        self.position = position
        self.pnl = pnl
        self.buy_price = buy_price
        self.sell_price = sell_price
        self.all_trades = all_trades if all_trades else np.array([0, 0])
        self.buy_volume = buy_volume
        self.sell_volume = sell_volume
        self.price_margin = price_margin
        self.const_position_size = const_position_size
        self.moving_average_one = moving_average_one
        self.moving_average_two = moving_average_two
        self.buy_order = pd.DataFrame(np.array([[self.buy_price, self.buy_volume, self.latency, self.agent_id]]),
                                      columns=["buy_price", "buy_volume", "latency", "agent_id"],
                                      index=[self.agent_id])
        self.sell_order = pd.DataFrame(np.array([[self.sell_price, self.sell_volume, self.latency, self.agent_id]]),
                                       columns=["sell_price", "sell_volume", "latency", "agent_id"],
                                       index=[self.agent_id])

    def calculate_buy_price(self, state: dict) -> float:
        """
        Calculates buy price

        :param state: market state information
        :return: buy price
        """
        noise = np.random.normal(loc=0, scale=0.01)
        buy_price = state["market_prices"][-1] * (1 + self.price_margin + noise)
        buy_price = np.maximum(buy_price, 0)
        return buy_price

    def calculate_sell_price(self, state: dict) -> float:
        """
        Calculates sell price

        :param state: market state information
        :return: sell price
        """
        noise = np.random.normal(loc=0, scale=0.01)
        sell_price = state["market_prices"][-1] * (1 + noise)
        sell_price = np.maximum(sell_price, 0)
        return sell_price

    def calculate_buy_volume(self, state: dict) -> float:
        """
        Calculates buy price

        :param state:
        :return:
        """
        volume = self.const_position_size  # np.random.randint(1, 2)

        return volume

    def calculate_sell_volume(self, state: dict) -> float:
        """
        Calculates sell price

        :param state:
        :return:
        """
        volume = self.const_position_size  # np.random.randint(1, 2)

        return volume

    def calculate_profit_and_loss(self, state: dict) -> NoReturn:
        """
        Calculates profit and loss

        :param state: market state information
        :return: total profit and loss
        """
        realized_value = np.sum(self.all_trades[:, 0] * self.all_trades[:, 1])
        unrealized_value = self.position * state["market_prices"][-1] * (1 - state["slippage"])

        self.pnl = realized_value + unrealized_value

    def update(self, state: dict) -> NoReturn:
        """
        Updates agents ask and bid prices, and corresponding volumes, when new state is provided

        :param state: market state information
        :return: NoReturn
        """
        # update latency
        self.latency = self.delta + np.random.uniform(1e-6, 1)

        # check trend direction and aim for strategic position

        ma1 = np.average(state["market_prices"][- self.moving_average_one:])
        ma2 = np.average(state["market_prices"][- self.moving_average_two:])

        trend = ma1 / ma2

        self.buy_price = np.nan
        self.sell_price = np.nan

        if trend >= 1 and self.position < self.const_position_size:
            self.buy_price = self.calculate_buy_price(state)
            self.buy_volume = self.calculate_buy_volume(state) - self.position
        elif trend >= 1 and self.position >= self.const_position_size:
            pass
        elif trend < 1 and self.position >= - self.const_position_size:
            self.sell_price = self.calculate_sell_price(state)
            self.sell_volume = self.calculate_sell_volume(state) + self.position
        elif trend < 1 and self.position < - self.const_position_size:
            pass

        # Update orders
        self.buy_order = pd.DataFrame(np.array([[self.buy_price, self.buy_volume, self.latency, self.agent_id]]),
                                      columns=["buy_price", "buy_volume", "latency", "agent_id"],
                                      index=[self.agent_id])
        self.sell_order = pd.DataFrame(np.array([[self.sell_price, self.sell_volume, self.latency, self.agent_id]]),
                                       columns=["sell_price", "sell_volume", "latency", "agent_id"],
                                       index=[self.agent_id])


class MarketMakerAgent(Agent):
    """
    Market making agent class
    """

    def __init__(self,
                 agent_id: int = None,
                 delta: float = None,
                 gamma: float = 0.01,
                 gamma2: float = 0.5,
                 position: int = 0,
                 pnl: float = None,
                 buy_price: float = None,
                 sell_price: float = None,
                 all_trades: np.ndarray = None):
        """
        Constructor
        :param latency: latency when matching agents in the market environment
        """
        self.agent_class = "MM"
        self.agent_id = agent_id
        self.delta = delta  # base latency
        self.gamma = gamma  # midprice sensitivity to position size
        self.gamma2 = gamma2  # spread sensitivity to local volatility
        self.latency = delta
        self.position = position
        self.pnl = pnl
        self.buy_price = buy_price
        self.sell_price = sell_price
        self.all_trades = all_trades if all_trades else np.array([0, 0])
        self.buy_volume = None
        self.sell_volume = None
        self.spread = None
        self.mid_price = None
        self.buy_order = pd.DataFrame(np.array([[self.buy_price, self.buy_volume, self.latency, self.agent_id]]),
                                      columns=["buy_price", "buy_volume", "latency", "agent_id"],
                                      index=[self.agent_id])
        self.sell_order = pd.DataFrame(np.array([[self.sell_price, self.sell_volume, self.latency, self.agent_id]]),
                                       columns=["sell_price", "sell_volume", "latency", "agent_id"],
                                       index=[self.agent_id])

    def calculate_volatility(self, state: dict, n_observations=10) -> float:
        """
        calculates market price volatility

        :param state:
        :param n_observations: dats to calculate volatility for
        :return:
        """
        vol = np.std(state["market_prices"][-n_observations:])
        return vol

    def calculate_spread(self, state: dict) -> float:
        """
        updates agents bid-ask spread

        :param state:
        :return:
        """
        vol = self.calculate_volatility(state)
        spread = vol * self.gamma2
        return spread

    def calculate_mid_price(self, state: dict) -> float:
        """
        updates agents own midprice to place ask and bids from

        :param state:
        :return:
        """

        market_mid_price = state["market_prices"][-1]
        market_leaning = self.gamma * self.position
        mid_price = market_mid_price * (1 - market_leaning)
        return mid_price

    def calculate_buy_price(self) -> float:
        """
        Calculates buy price

        :param state:
        :return:
        """
        buy_price = self.mid_price - self.spread / 2 + np.random.normal(loc=0, scale=0.001)
        return buy_price

    def calculate_sell_price(self) -> float:
        """
        Calculates sell price

        :param state:
        :return:
        """
        sell_price = self.mid_price + self.spread / 2 + np.random.normal(loc=0, scale=0.001)
        return sell_price

    def calculate_buy_volume(self) -> float:
        """
        Calculates buy price

        :param state:
        :return:
        """
        buy_volume = 3
        return buy_volume

    def calculate_sell_volume(self) -> float:
        """
        Calculates sell price

        :param state:
        :return:
        """
        sell_volume = 3
        return sell_volume

    def calculate_profit_and_loss(self, state: dict) -> NoReturn:
        """
        Calculates profit and loss

        :param state:
        :return:
        """
        realized_value = np.sum(self.all_trades[:, 0] * self.all_trades[:, 1])
        unrealized_value = self.position * state["market_prices"][-1] * (1 - state["slippage"])

        self.pnl = realized_value + unrealized_value

    def update(self, state: dict) -> NoReturn:
        """
        Updates agents ask and bid prices, and corresponding volumes, when new state is provided

        :param state:
        :return:
        """
        # Update latency
        self.latency = self.delta / (1 + np.random.uniform(1e-6, 1))

        # Update parameters to calculate prices
        self.mid_price = self.calculate_mid_price(state)
        self.spread = self.calculate_spread(state)

        # Update volumes
        self.buy_volume = self.calculate_buy_volume()
        self.sell_volume = self.calculate_sell_volume()

        # Update prices
        self.buy_price = self.calculate_buy_price()
        self.sell_price = self.calculate_sell_price()

        self.buy_order = pd.DataFrame(np.array([[self.buy_price, self.buy_volume, self.latency, self.agent_id]]),
                                      columns=["buy_price", "buy_volume", "latency", "agent_id"],
                                      index=[self.agent_id])
        self.sell_order = pd.DataFrame(np.array([[self.sell_price, self.sell_volume, self.latency, self.agent_id]]),
                                       columns=["sell_price", "sell_volume", "latency", "agent_id"],
                                       index=[self.agent_id])


##################################################
# Reinforcement learning Agents and Helper-classes
##################################################

class RLAgent(Agent):
    """
    Agent which makes noisy buy and sell prices around market_prices
    """

    def __init__(self,
                 agent_id: int = None,
                 delta: float = None,
                 position: int = 0,
                 pnl: float = None,
                 buy_price: float = None,
                 sell_price: float = None,
                 all_trades: np.ndarray = None,
                 buy_volume: float = None,
                 sell_volume: float = None,
                 epsilon: float = 0.1,
                 state_size: int = 2,
                 lr: float = 0.1,
                 alpha: float = 0.6,
                 noise_range: Tuple = [0.01, 0.1],
                 gamma: float = 0.99,
                 nn_parameters: dict = {"n_layers": 10, "n_units": 20, "n_features": 11, "batch_size": 1,
                                        "dense_units": 1, "n_timepoints": 16, "n_epochs": 20},
                 function_approximator: str = "rnn"):

        """
        Constructor
        :param latency: latency when matching agents in the market environment
        """
        self.agent_class = "RL"
        self.agent_id = agent_id
        self.delta = delta
        self.latency = delta
        self.position = position
        self.pnl = pnl
        self.buy_price = buy_price
        self.sell_price = sell_price
        self.all_trades = all_trades if all_trades else np.array([0, 0])
        self.buy_volume = buy_volume
        self.sell_volume = sell_volume
        self.noise_range = noise_range
        self.random_agent_price = None
        self.epsilon = epsilon
        self.lr = lr
        self.alpha = alpha
        self.state_size = state_size
        self.gamma = gamma
        self.nn_parameters = nn_parameters
        self.data = np.ndarray(shape = (0, self.nn_parameters["n_features"]))
        if function_approximator == "rnn":
            self.model = self.rnn_model()

    def calculate_buy_price(self, state: dict) -> float:
        """
        Calculates buy price

        :param state: market state information
        :return: buy price
        """
        buy_price = self.random_agent_price * (
                    1 - np.random.uniform(low=self.noise_range[0], high=self.noise_range[1] - 0.02))
        buy_price = np.maximum(buy_price, 0)
        return buy_price

    def calculate_sell_price(self, state: dict) -> float:
        """
        Calculates sell price

        :param state: market state information
        :return: sell price
        """
        sell_price = self.random_agent_price * (
                    1 + np.random.uniform(low=self.noise_range[0], high=self.noise_range[1]))
        sell_price = np.maximum(sell_price, 0)
        return sell_price

    def calculate_buy_volume(self, state: dict) -> float:
        """
        Calculates buy price

        :param state:
        :return:
        """
        volume = np.random.randint(0, 3)

        return volume

    def calculate_sell_volume(self, state: dict) -> float:
        """
        Calculates sell price

        :param state:
        :return:
        """
        volume = np.random.randint(0, 3)

        return volume

    def calculate_profit_and_loss(self, state: dict) -> NoReturn:
        """
        Calculates profit and loss

        :param state: market state information
        :return: total profit and loss
        """
        realized_value = np.sum(self.all_trades[:, 0] * self.all_trades[:, 1])
        unrealized_value = self.position * state["market_prices"][-1] * (1 - state["slippage"])

        self.pnl = realized_value + unrealized_value

    def create_features(self, state):
        ma1 = np.average(state["market_prices"][-50:])
        ma2 = np.average(state["market_prices"][-200:])
        trend_feature = ma1 / ma2
        spread_feature = np.round(state["mean_buy_price"] - state["mean_sell_price"], 1)
        feature_list = [ma1, ma2, trend_feature, spread_feature]
        return feature_list

    def store_data(self, state) -> NoReturn:
        last_pnl = self.pnl
        self.calculate_profit_and_loss(state)
        self.reward = self.pnl - last_pnl
        #ma1 = np.average(state["market_prices"][-50:])
        #ma2 = np.average(state["market_prices"][-200:])
        #trend_feature = ma1 / ma2
        #spread_feature = np.round(state["mean_buy_price"] - state["mean_sell_price"], 1)
        features = self.create_features(self, state)
        # data = np.array([ma1, ma2, trend_feature, spread_feature, state["market_prices"][-1],
        #                      self.position, self.buy_volume, self.sell_volume, self.buy_price,
        #                      self.sell_price, self.reward])
        data = np.array([features[0], features[1], features[2], features[3], state["market_prices"][-1],
                             self.position, self.buy_volume, self.sell_volume, self.buy_price,
                             self.sell_price, self.reward])
        self.data = np.vstack((self.data, data))
        
    def data_to_feather(self) -> NoReturn:
        pd.DataFrame(self.data, columns = ['ma1', 'ma2', 'trend_feature',
                                               'spread_feature', 'market_prices', 'position',
                                               'buy_volume', 'sell_volume', 'buy_price',
                                               'sell_price', 'reward']).to_feather('data/data.feather')

    def scale_and_reshape(self, data):
        feature_vector = data.iloc[:, :-1]
        col_names = feature_vector.columns
        scaler = MinMaxScaler(feature_range=(0, 1))
        feature_vector = pd.DataFrame(scaler.fit_transform(feature_vector), columns = col_names)

        # RESHAPING
        rows_x = len(feature_vector[:,0])
        x = feature_vector[range(self.nn_parameters["n_timepoints"] * rows_x), :]
        feature_vector_reshaped = np.reshape(x, (rows_x, self.nn_parameters["n_timepoints"], self.nn_parameters["n_features"]))
        return feature_vector_reshaped

    def train_model(self):

        data = pd.DataFrame(self.data, columns = ['ma1', 'ma2', 'trend_feature',
                                        'spread_feature', 'market_prices', 'position',
                                        'buy_volume', 'sell_volume', 'buy_price',
                                        'sell_price', 'reward'])
        reward_vector = data.iloc[:, -1].values

        disc_reward = np.zeros_like(reward_vector)
        cum_reward = 0
        for i in reversed(range(len(reward_vector))):
            cum_reward = cum_reward * self.gamma + reward_vector[i]
            disc_reward[i] = cum_reward

        feature_vector_reshaped = self.scale_and_reshape(data)
        # TRAIN
        self.model.fit(feature_vector_reshaped, disc_reward, epochs=self.nn_parameters["n_epochs"], batch_size=1, verbose=2)


    def rnn_model(self):
        model = Sequential()
        #     model.add(SimpleRNN(hidden_units, input_shape=input_shape,
        #                         activation=activation[0]))
        model.add(LSTM(self.nn_parameters["n_units"], batch_input_shape=(self.nn_parameters["batch_size"], self.time_steps, self.nn_parameters["n_features"]), stateful=True, return_sequences=True))
        model.add(LSTM(self.nn_parameters["n_units"], batch_input_shape=(self.nn_parameters["batch_size"], self.time_steps, self.nn_parameters["n_features"]), stateful=True))
        model.add(Dense(units=self.nn_parameters["dense_units"], activation="linear"))
        model.compile(loss='mean_squared_error', optimizer='adam')
        return model

    def predict_state(self, state):
        state_feature_vector = self.create_features(state)
        data = pd.DataFrame(state_feature_vector)
        state_feature_vector_reshaped = self.scale_and_reshape(data)
        prediction = self.model.predict(state_feature_vector_reshaped, batch_size = 1)
        return prediction

    def take_action(self):
        grid = []
        predictions = []
        for i in grid:
            predictions.append(self.predict_state())

    def nn(self, hidden_units, dense_units, input_shape, activation):
        model = Sequential()
        model.add(SimpleRNN(hidden_units, input_shape=input_shape, 
                            activation=activation[0]))
        model.add(Dense(units=dense_units, activation=activation[1]))
        model.compile(loss='mean_squared_error', optimizer='adam')
        return model


    def take_action(self, state) -> NoReturn:
        # Volume and price range 
        vol_range_lower, vol_range_upper = 0, 10
        price_range_lower, price_range_upper = state["market_prices"][-1] * 0.99, state["market_prices"][-1] * 1.01
        
        # Update volumes
        self.buy_volume = random.randint(vol_range_lower, vol_range_upper)
        self.sell_volume = random.randint(vol_range_lower, vol_range_upper)

        # Update prices
        self.buy_price = random.uniform(price_range_lower, price_range_upper)
        self.sell_price = random.uniform(price_range_lower, price_range_upper)

        self.buy_order = pd.DataFrame(np.array([[self.buy_price, self.buy_volume, self.latency, self.agent_id]]),
                              columns=["buy_price", "buy_volume", "latency", "agent_id"],
                              index=[self.agent_id])
        self.sell_order = pd.DataFrame(np.array([[self.sell_price, self.sell_volume, self.latency, self.agent_id]]),
                                       columns=["sell_price", "sell_volume", "latency", "agent_id"],
                                       index=[self.agent_id])

    def update(self, state: dict) -> NoReturn:
        """
        Updates agents ask and bid prices, and corresponding volumes, when new state is provided

        :param state: market state information
        :return: NoReturn
        """
        # Update trades (pays cash buys and receive for sells) and position

        temp_state = state
        self.last_price = state["market_prices"][-1]
        self.store_data(temp_state)
        self.take_action(state)


class Memory:
    """
    Memory class to perform experience replay
    """
    def __init__(self, max_size: int):

        self.max_size = max_size
        # Clear
        self.states = []
        self.counter = 0
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.terminals = []
        self.discounted_rewards = []

    def clear(self):
        """ Clear all memory"""
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.terminals = []
        self.discounted_rewards = []

    def clear_earliest_entry(self):
        """Clear first remembered experiance"""
        self.states = self.states[1:]
        self.actions = self.actions[1:]
        self.rewards = self.rewards[1:]
        self.next_states = self.next_states[1:]
        self.terminals = self.terminals[1:]
        self.discounted_rewards = self.discounted_rewards[1:]

    def add_transition(self, state, action, reward, next_state, terminal):
        """ Add new state, action and reward to memory """

        if len(self.states) == self.max_size:
            self.clear_earliest_entry()

        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.terminals.append(terminal)
        self.discounted_rewards.append(reward)

    def draw_batch(self, batch_size: int = 50):
        """draws a random sample batch from memory"""
        combined = list(zip(self.states,
                            self.actions,
                            self.rewards,
                            self.discounted_rewards,
                            self.next_states,
                            self.terminals))

        random.shuffle(combined)

        if batch_size is not None:
            combined = combined[:batch_size]

        states, actions, rewards, discounted_rewards, next_states, terminals = zip(*combined)

        batch = {'states': torch.stack(states).float(),
                 'actions': torch.stack(actions).float(),
                 'rewards': torch.stack(rewards).float(),
                 'discounted_rewards': torch.stack(discounted_rewards).float(),
                 'next_states': torch.stack(next_states).float(),
                 'terminals': torch.stack(terminals)}

        return batch

    def full_memory(self):
        """ returns full memory"""
        batch = {'states': self.states,
                 'actions': self.actions,
                 'rewards': self.rewards,
                 'discounted_rewards': self.discounted_rewards,
                 'next_states': self.next_states,
                 'terminals': self.terminals}
        return batch


class ActionValueNetwork(nn.Module):
    """Critic Neural Network approximating the action-value function"""

    def __init__(self, input_dims, fc1_dims: int = 256, fc2_dims: int = 256,
                 chkpt_dir = 'nn_models', name = 'gaus_bin'):
        super(ActionValueNetwork, self).__init__()
        
        self.checkpoint_file = os.path.join(chkpt_dir,name+'_nn')
        self.input_dims = input_dims
        self.fc1 = nn.Linear(self.input_dims, fc1_dims)  # Fully connected layer 1 / Hidden layer 1
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)  # Hidden layer 2
        self.fc3 = nn.Linear(fc2_dims, 1)  # Action value

    def forward(self, activation):
        """feed through the network"""
        activation = f.relu(self.fc1(activation))
        activation = f.relu(self.fc2(activation))
        activation = self.fc3(activation)

        return activation
    
    def save_checkpoint(self):
        print('...saving checkpoint...')
        torch.save(self.state_dict(), self.checkpoint_file)
        
    def load_checkpoint(self, file):
        print('... loading checkpoint ...')
        self.load_state_dict(torch.load(file))

class GaussianPolicyNetwork(nn.Module):

    def __init__(self, action_dims: int, input_dims: int, max_action_value, min_action_value,
                 max_action_value_two: int = 0, min_action_value_two: int = 15,
                 fc1_dims: int = 256, fc2_dims: int = 256, soft_clamp_function=None):
        super(GaussianPolicyNetwork, self).__init__()

        self.input_dims = input_dims
        self.action_dims = action_dims
        self.soft_clamp_function = soft_clamp_function
        self.max_action_value = max_action_value
        self.min_action_value = min_action_value
        self.max_action_value_two = max_action_value_two
        self.min_action_value_two = min_action_value_two
        self.max_log_sigma = -3  # to cutoff variance estimates
        self.min_log_sigma = -10  # to cutoff variance estimates
        self.max_log_sigma_two = 1
        self.min_log_sigma_two = -10

        self.fc1 = nn.Linear(self.input_dims, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.mu_layer = nn.Linear(fc2_dims, self.action_dims)
        self.log_sigma_layer = nn.Linear(fc2_dims, self.action_dims)

    def forward(self, activation):
        """feed through the network and output mu and sigma vectors"""
        activation = activation.float()
        activation = f.relu(self.fc1(activation))
        activation = f.relu(self.fc2(activation))
        mu = self.mu_layer(activation)
        log_sigma = self.log_sigma_layer(activation)

        log_sigma = log_sigma.clamp(min=self.min_log_sigma, max=self.max_log_sigma)
        sigma = torch.exp(log_sigma)

        return mu, sigma

    def get_action(self, state, eval_deterministic=False, save_mu = False):

        mu, sigma = self.forward(state)
        print("sigma: ", sigma)
        if eval_deterministic:
            action = mu.detach()
        else:
            gauss_dist = Normal(loc=mu, scale=sigma)
            action = gauss_dist.sample()
            action.detach()

        # action = self.max_action_value * torch.tanh(action / self.max_action_value)
        action[:2] = action[:2].clamp(min=self.min_action_value, max=self.max_action_value)  # CLAMP PRICES
        action[-2:] = action[-2:].clamp(min=self.min_action_value_two, max=self.max_action_value_two)  # CLAMP VOLUMES
        action[-2:] = action[-2:].int()
            
        if save_mu:
            return action, mu, sigma

        return action

    def get_action_and_log_prob(self, state):

        mu, sigma = self.forward(state)  # Initialize activation with state
        gauss_dist = Normal(loc=mu, scale=sigma)
        action = gauss_dist.sample()
        action.detach()
        action[:2] = action[:2].clamp(min=self.min_action_value, max=self.max_action_value)  # CLAMP PRICES
        action[-2:] = action[-2:].clamp(min=self.min_action_value_two, max=self.max_action_value_two)  # CLAMP VOLUMES
        action[-2:] = action[-2:].int()
        log_prob = gauss_dist.log_prob(action)

        return action, log_prob

    def random_sample(self, state):

        mu, sigma = self.forward(state)
        loc = torch.zeros(size=[state.shape[0], 1], dtype=torch.float32)
        scale = loc + 1.0
        unit_gauss = Normal(loc=loc, scale=scale)
        gauss = Normal(loc=mu, scale=sigma)
        epsilon = unit_gauss.sample()
        action = mu + sigma * epsilon
        action = action.requires_grad_()
        action = self.max_action_value * torch.tanh(action / self.max_action_value)
        log_prob = gauss.log_prob(action.data)

        return action, log_prob

class GaussianBinomialPolicyNetwork(nn.Module):

    def __init__(self, action_dims: int, input_dims: int, max_action_value, min_action_value,
                 max_action_value_two: int = 0, min_action_value_two: int = 15,
                 fc1_dims: int = 256, fc2_dims: int = 256, soft_clamp_function=None, sigma = 0.05,
                 chkpt_dir = 'nn_models', name = 'gaus_bin'):
        super(GaussianBinomialPolicyNetwork, self).__init__()
        
        self.checkpoint_file = os.path.join(chkpt_dir,name+'_nn')
        self.input_dims = input_dims
        self.action_dims = action_dims
        self.soft_clamp_function = soft_clamp_function
        self.max_action_value = max_action_value
        self.min_action_value = min_action_value
        self.max_action_value_two = max_action_value_two
        self.min_action_value_two = min_action_value_two
        self.max_log_sigma = -3  # to cutoff variance estimates
        self.min_log_sigma = -10  # to cutoff variance estimates
        self.max_log_sigma_two = 1
        self.min_log_sigma_two = -10
        self.sigma = sigma

        self.fc1 = nn.Linear(self.input_dims, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.mu_layer = nn.Linear(fc2_dims, int(self.action_dims/2))
        # self.log_sigma_layer = nn.Linear(fc2_dims, self.action_dims)
        self.p_layer = nn.Linear(fc2_dims, int(self.action_dims/2))

    def forward(self, activation):
        """feed through the network and output mu and sigma vectors"""
        activation = activation.float()
        activation = f.relu(self.fc1(activation))
        activation = f.relu(self.fc2(activation))
        mu = torch.tanh(self.mu_layer(activation))
        p = torch.sigmoid(self.p_layer(activation))
        # log_sigma = self.log_sigma_layer(activation)
        # log_sigma = log_sigma.clamp(min=self.min_log_sigma, max=self.max_log_sigma)
        # sigma = torch.exp(log_sigma)

        return mu, p#, sigma

    def get_action(self, state, eval_deterministic=False, save_mu = False):

        mu, p = self.forward(state)
        
        # if eval_deterministic:
        #     action = mu.detach()
        # else:
        gauss_dist = Normal(loc=mu, scale=self.sigma)
        action_mu = gauss_dist.sample()
        
        binomial_dist = Binomial(total_count = self.max_action_value_two, probs = p)
        action_p = binomial_dist.sample()
        action = torch.cat((action_mu, action_p))
        action.detach()
        
        # action = self.max_action_value * torch.tanh(action / self.max_action_value)
        action[:2] = action[:2].clamp(min=self.min_action_value, max=self.max_action_value)  # CLAMP PRICES
        action[-2:] = action[-2:].clamp(min=self.min_action_value_two, max=self.max_action_value_two)  # CLAMP VOLUMES
        action[-2:] = action[-2:].int()
            
        if save_mu:
            return action, mu, p

        return action

    def get_action_and_log_prob(self, state):

        mu, p = self.forward(state)  # Initialize activation with state
        gauss_dist = Normal(loc=mu, scale=self.sigma)
        action_mu = gauss_dist.sample()
        
        binomial_dist = Binomial(total_count = self.max_action_value_two, probs = p)
        action_p = binomial_dist.sample()
        action = torch.cat((action_mu, action_p), 1)
        action.detach()
        
        action[:2] = action[:2].clamp(min=self.min_action_value, max=self.max_action_value)  # CLAMP PRICES
        action[-2:] = action[-2:].clamp(min=self.min_action_value_two, max=self.max_action_value_two)  # CLAMP VOLUMES
        action[-2:] = action[-2:].int()
        
        log_prob_mu = gauss_dist.log_prob(action_mu)
        log_prob_p = binomial_dist.log_prob(action_p)
        
        log_prob = torch.cat((log_prob_mu, log_prob_p), 1)
        
        return action, log_prob

    def random_sample(self, state):

        mu, sigma = self.forward(state)
        loc = torch.zeros(size=[state.shape[0], 1], dtype=torch.float32)
        scale = loc + 1.0
        unit_gauss = Normal(loc=loc, scale=scale)
        gauss = Normal(loc=mu, scale=sigma)
        epsilon = unit_gauss.sample()
        action = mu + sigma * epsilon
        action = action.requires_grad_()
        action = self.max_action_value * torch.tanh(action / self.max_action_value)
        log_prob = gauss.log_prob(action.data)

        return action, log_prob
    
    def save_checkpoint(self):
        print('...saving checkpoint...')
        torch.save(self.state_dict(), self.checkpoint_file)
        
    def load_checkpoint(self, file):
        print('... loading checkpoint ...')
        self.load_state_dict(torch.load(file))


class ActorCriticAgent:
    def __init__(self,
                 policy: nn.Module,
                 qf: ActionValueNetwork,
                 qf_optimiser: Optimizer,
                 policy_optimiser: Optimizer,
                 discount_factor: float = None,
                 env=None,
                 max_evaluation_episode_length: int = 200,
                 batch_size=50,
                 max_memory_size=10000,
                 num_evaluation_episodes=5,
                 num_training_episode_steps=1000,
                 eval_deterministic=True,
                 training_on_policy=False,
                 vf: nn.Module=None,
                 vf_optimiser: Optimizer=None,
                 agent_id=0,
                 delta=1,
                 init_state=None,
                 position_penalty = 0.001):
        
        self.agent_class = "ActorCritic"
        self.agent_id = agent_id
        self.delta = delta  # base latency
        self.latency = delta
        self.position = 0
        self.pnl = 0
        self.buy_price = None
        self.sell_price = None
        self.all_trades = np.array([[0, 0]])
        self.buy_volume = None
        self.sell_volume = None
        self.spread = None
        self.mid_price = None
        self.buy_order = pd.DataFrame(np.array([[self.buy_price, self.buy_volume, self.latency, self.agent_id]]),
                                      columns=["buy_price", "buy_volume", "latency", "agent_id"],
                                      index=[self.agent_id])
        self.sell_order = pd.DataFrame(np.array([[self.sell_price, self.sell_volume, self.latency, self.agent_id]]),
                                       columns=["sell_price", "sell_volume", "latency", "agent_id"],
                                       index=[self.agent_id])

        self.policy = policy
        self.qf = qf
        self.vf = vf
        self.target_vf = deepcopy(vf)
        self.tau = 1e-2
        self.vf_optimiser = vf_optimiser
        self.qf_optimiser = qf_optimiser
        self.policy_optimiser = policy_optimiser
        self.env = env
        self.discount_factor = discount_factor
        self.batch_size = batch_size
        self.max_evaluation_episode_length = max_evaluation_episode_length
        self.num_evaluation_episodes = num_evaluation_episodes
        self.num_training_episode_steps = num_training_episode_steps
        self.training_on_policy = training_on_policy
        self.memory = Memory(max_size=max_memory_size)
        self.state_features = self.get_state_features(init_state)
        self.position_penalty = position_penalty 

        #self.pretraining_policy = Uniform(high=torch.Tensor([policy.max_action_value]), low=torch.Tensor([policy.min_action_value]))
        self.eval_deterministic = eval_deterministic


        self.mu1, self.mu2, self.mu3, self.mu4 = 0, 0, 0, 0


        #self.loss = nn.MSELoss()
        #self.R_av = None
        #self.R_tot = 0
    def reset(self):
        """
        Resets all attributes related to the market, pnl ect.
        """
        self.position = 0
        self.pnl = 0
        self.buy_price = None
        self.sell_price = None
        self.all_trades = np.array([[0, 0]]).copy()
        self.buy_volume = None
        self.sell_volume = None
        self.memory.clear()

    def score_gradient_descent(self) -> NoReturn:
        """
        Takes a gradient descent step and updates parameters in function approximators
        """

        """
        Calculate discounted rewards
        """
        reward_vector = [x for reward in self.memory.full_memory()["rewards"] for x in reward.numpy()]
        discounted_rewards = [0] * len(reward_vector)
        cum_reward = 0
        for i in reversed(range(len(reward_vector))):
            cum_reward = cum_reward * self.discount_factor + reward_vector[i]
            discounted_rewards[i] = torch.tensor([cum_reward])
        self.memory.discounted_rewards = discounted_rewards

        """
        Train
        """

        if self.training_on_policy:
            batch = self.memory.draw_batch(self.batch_size)
            self.memory.clear()
        else:
            batch = self.memory.draw_batch(self.batch_size)
        states = batch['states']
        actions = batch['actions']
        rewards = batch['rewards']
        next_states = batch['next_states']
        terminals = batch['terminals']

        """
        Calculate 
        """
        state_values = self.vf(states)
        state_actions = torch.cat((states, actions), 1)  # qf needs both state and actions as input
        q_values = self.qf(state_actions)
        next_state_values = self.target_vf(next_states)

        new_actions, log_pis = self.policy.get_action_and_log_prob(states)  # get new_actions from current parameters
        new_state_actions = torch.cat((states, new_actions), 1)  # old state from buffer plus new_actions
        new_q_values = self.qf(new_state_actions)  # get value of chosen action

        """
        State Value Losses - Critic
        """
        state_value_target = new_q_values  # The state should represent the value taking the best action
        vf_loss = (state_value_target.detach() - state_values).pow(2).mean()

        """
        Action Value Losses - Critic
        use temporal difference and approx TD error, see Silver slide set 7 (slide 326).
        """
        q_targets = rewards + self.discount_factor * (1 - terminals) * next_state_values
        qf_loss = (q_targets.detach() - q_values).pow(2).mean()

        """
        Policy Losses - Actor  
        NOT COMPLETELY SURE ABOUT THE LOSS FUNCTION (MAYBE IT SHOULD BE MULTIPLIED BY MINUS 1)
        """
        advantage = new_q_values - state_values
        #policy_loss = (log_pis * (log_pis - advantage.detach())).mean()
        policy_loss = (log_pis * (advantage.detach() - log_pis)).mean()
        
        """
        Parameter updates using gradient descent
        """
        self.qf_optimiser.zero_grad()
        qf_loss.backward()
        self.qf_optimiser.step()

        self.vf_optimiser.zero_grad()
        vf_loss.backward()
        self.vf_optimiser.step()

        self.policy_optimiser.zero_grad()
        policy_loss.backward()
        self.policy_optimiser.step()

        self.soft_update()

    def soft_update(self):
        """value function parameters"""
        for target_param, param in zip(self.target_vf.parameters(), self.vf.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)

    def memory_to_feather(self) -> NoReturn:
        """Save memory to feather"""
        pd.DataFrame(self.memory.full_memory()).to_feather('data/data.feather')

    def calculate_profit_and_loss(self, state: dict) -> NoReturn:
        """
        Calculates profit and loss

        :param state: market state information
        :return: total profit and loss
        """
        realized_value = np.sum(self.all_trades[:, 0] * self.all_trades[:, 1])
        unrealized_value = self.position * state["market_prices"][-1] * (1 - state["slippage"])

        self.pnl = realized_value + unrealized_value

    def get_state_features(self, state: dict, n_returns = 5):
        """
        Extract features from market state information

        :param state: market state
        :return: state features
        """
        features = []
        returns_relative = np.array(state["market_prices"][-n_returns:]) / np.array(state["market_prices"][-n_returns-1:-1]) - 1
        #returns = np.array(state["market_prices"][-n_returns:]) - np.array(state["market_prices"][-n_returns - 1:-1])
        average_return = np.average(returns_relative[-100:])
        local_volatility = np.average((returns_relative[-10:]-average_return)**2)
        for i in range(n_returns):
            #features.append(state["market_prices"][-i])
            features.append(returns_relative.tolist()[i])
        features.append(state["volume"])
        features.append(state["total_buy_volume"])
        features.append(state["total_sell_volume"])
        features.append(state["mean_buy_price"])
        features.append(state["mean_sell_price"])
        features.append(local_volatility)
        features.append(self.position)


        return torch.tensor(features)

    def update(self, state: dict, exploration_mode=False):
        """
        Updates RL model and its prices and volumes
        """
        state_features = self.state_features if self.state_features is not None else self.get_state_features(state)
        if exploration_mode:
            action = torch.tensor([-np.abs(np.random.normal(scale = 0.01)),  # buy_price
                                    np.abs(np.random.normal(scale = 0.01)),  # sell_price
                                    np.random.randint(0, 5),  # buy_volume
                                    np.random.randint(0, 5)   # sell_volume
                                    ])
            #================ UNCOMMENT BELOW AND COMMENT ABOVE TO ONLY RANDOM SAMPLE PRICES ================#
            
            # action_p = self.policy.get_action(state_features)
            # action = torch.tensor([-np.abs(np.random.normal(scale = 0.01)),  # buy_price
            #                        np.abs(np.random.normal(scale = 0.01)),
            #                        action_p[2],
            #                        action_p[3]
            #                        ])
            
            # action = torch.cat((action_mu, action_p[2], action_p[3]))
        else:
            action = self.policy.get_action(state_features)

        pnl = self.pnl
        self.calculate_profit_and_loss(state=state)
        new_pnl = self.pnl
        position = self.position
        reward = torch.tensor([new_pnl - pnl - self.position_penalty * position**2])
        next_state_features = self.get_state_features(state)
        terminal = torch.tensor(0)

        self.memory.add_transition(state=state_features,
                                   action=action,
                                   reward=reward,
                                   next_state=next_state_features,
                                   terminal=terminal)
        self.state_features = next_state_features
        if terminal:
            raise NotImplementedError("No terminal state definition")

        if exploration_mode:
            new_action = torch.tensor([-np.abs(np.random.normal(scale = 0.01)),  # buy_price
                                    np.abs(np.random.normal(scale = 0.01)),  # sell_price
                                    np.random.randint(0, 5),  # buy_volume
                                    np.random.randint(0, 5)   # sell_volume
                                    ])

            #================ UNCOMMENT BELOW AND COMMENT ABOVE TO ONLY RANDOM SAMPLE PRICES ================#
            # action_p = self.policy.get_action(state_features)
            # new_action = torch.tensor([-np.abs(np.random.normal(scale = 0.01)),  # buy_price
            #                        np.abs(np.random.normal(scale = 0.01)),
            #                        action_p[2],
            #                        action_p[3]
            #                        ])
            
        else:
            new_action, mus, ps = self.policy.get_action(self.state_features, save_mu = True)
            self.mu1 = mus[0]
            self.mu2 = mus[1]
            self.mu3 = ps[0]
            self.mu4 = ps[1]

        self.buy_price = (1 + new_action[0].numpy()) * state['market_prices'][-1]
        self.sell_price = (1 + new_action[1].numpy()) * state['market_prices'][-1]
        self.buy_volume = int(new_action[2].numpy())
        self.sell_volume = int(new_action[3].numpy())

        self.buy_order = pd.DataFrame(np.array([[self.buy_price, self.buy_volume, self.latency, self.agent_id]]),
                                      columns=["buy_price", "buy_volume", "latency", "agent_id"],
                                      index=[self.agent_id])
        self.sell_order = pd.DataFrame(np.array([[self.sell_price, self.sell_volume, self.latency, self.agent_id]]),
                                       columns=["sell_price", "sell_volume", "latency", "agent_id"],
                                       index=[self.agent_id])

        self.latency = self.delta / (1 + np.random.uniform(1e-6, 1))

    
    def save_models(self):
        self.policy.save_checkpoint()
        self.qf.save_checkpoint()
        self.vf.save_checkpoint()
        #self.target_vf.save_checkpoint()
        
        
    def load_models(self, file_policy = "", file_qf = "", file_vf = ""):
        if file_policy == "":
            file_policy = self.policy.checkpoint_file
        if file_qf == "":
            file_qf = self.qf.checkpoint_file
        if file_vf == "":
            file_vf = self.vf.checkpoint_file        
        self.policy.load_checkpoint(file_policy)
        self.qf.load_checkpoint(file_qf)
        self.vf.load_checkpoint(file_vf)
        #self.target_vf.load_checkpoint()
        
        
        
        
        
        
        
        
        
        
        
        

