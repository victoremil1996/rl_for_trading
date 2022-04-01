from typing import NoReturn, Tuple
import abc
import numpy as np


class Agent(abc.ABC):
    """
    Abstract class for agents
    """

    # @abc.abstractmethod
    # def __init__(self,
    #              agent_id: int = None,
    #              latency: float = None,
    #              position: int = 0,
    #              pnl: float = None,
    #              buy_price: float = None,
    #              sell_price: float = None,
    #              all_trades: list = []):
    #     """
    #     Constructor
    #     :param latency: latency when matching agents in the market environment
    #     """
    #
    #     self.agent_id = agent_id
    #     self.latency = latency
    #     self.position = position,
    #     self.pnl = pnl,
    #     self.buy_price = buy_price,
    #     self.sell_price = sell_price,
    #     self.all_trades = all_trades

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
        Updates agent-attributes when new state is provided

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
                 latency: float = None,
                 position: int = 0,
                 pnl: float = None,
                 buy_price: float = None,
                 sell_price: float = None,
                 all_trades: list = None,
                 noise_range: Tuple = [0.01, 0.1]):
        """
        Constructor
        :param latency: latency when matching agents in the market environment
        """

        self.agent_id = agent_id
        self.latency = latency
        self.position = position
        self.pnl = pnl
        self.buy_price = buy_price
        self.sell_price = sell_price
        self.all_trades = all_trades if all_trades else []
        self.noise_range = noise_range
        self.random_agent_price = None

    def calculate_buy_price(self, state: dict) -> float:
        """
        Calculates buy price

        :param state: market state information
        :return: buy price
        """
        buy_price = self.random_agent_price * (1 - np.random.uniform(low=self.noise_range[0], high=self.noise_range[1] - 0.02))
        buy_price = np.maximum(buy_price, 0)
        return buy_price

    def calculate_sell_price(self, state: dict) -> float:
        """
        Calculates sell price

        :param state: market state information
        :return: sell price
        """
        sell_price = self.random_agent_price * (1 + np.random.uniform(low=self.noise_range[0], high=self.noise_range[1]))
        sell_price = np.maximum(sell_price, 0)
        return sell_price

    def calculate_profit_and_loss(self, state: dict) -> NoReturn:
        """
        Calculates profit and loss

        :param state: market state information
        :return: total profit and loss
        """
        realized_value = np.sum(self.all_trades)
        unrealized_value = self.position * state["market_prices"][-1]

        self.pnl = realized_value + unrealized_value

    def update(self, state: dict) -> NoReturn:
        """
        Updates agent-attributes when new state is provided

        :param state: market state information
        :return: NoReturn
        """
        # Update trades (pays cash buys and receive for sells) and position
        execution_status = state["execution_status"][:, self.agent_id]
        if execution_status[0] == 1:
            buy_trade = - execution_status[0] * execution_status[2] - state["fee"] * execution_status[2]
            self.all_trades.append(buy_trade)
            self.position += 1

        if execution_status[1] == 1:
            sell_trade = execution_status[1] * execution_status[3] - state["fee"] * execution_status[3]
            self.all_trades.append(sell_trade)
            self.position -= 1


        # Update prices
        self.random_agent_price = state["market_prices"][-1] + np.random.normal(loc = 0, scale = 1)

        self.buy_price = self.calculate_buy_price(state)
        self.sell_price = self.calculate_sell_price(state)