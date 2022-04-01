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
                 buy_volume: float = None,
                 sell_volume: float = None,
                 noise_range: Tuple = [0.01, 0.1]):
        """
        Constructor
        :param latency: latency when matching agents in the market environment
        """
        self.agent_class = "Random"
        self.agent_id = agent_id
        self.latency = latency
        self.position = position
        self.pnl = pnl
        self.buy_price = buy_price
        self.sell_price = sell_price
        self.all_trades = all_trades if all_trades else []
        self.buy_volume = buy_volume
        self.sell_volume = sell_volume
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

        if execution_status[0] >= 1:
            buy_trade = - execution_status[0] * execution_status[2] - state["fee"] * execution_status[2]
            self.all_trades.append(buy_trade)
            self.position += execution_status[2]

        if execution_status[1] >= 1:
            sell_trade = execution_status[1] * execution_status[3] - state["fee"] * execution_status[3]
            self.all_trades.append(sell_trade)
            self.position -= execution_status[3]


        # Update prices and volume
        self.random_agent_price = state["market_prices"][-1] + np.random.normal(loc = 0, scale = 2)

        self.buy_price = self.calculate_buy_price(state)
        self.sell_price = self.calculate_sell_price(state)
        self.buy_volume = self.calculate_buy_volume(state)
        self.sell_volume = self.calculate_sell_volume(state)


class InvestorAgent(Agent):
    """
    Agent which buys or sells large orders in chunks over small periods. Similiar to Institutional Investors.
    """

    def __init__(self,
                 agent_id: int = None,
                 latency: float = None,
                 position: int = 0,
                 pnl: float = None,
                 buy_price: float = None,
                 sell_price: float = None,
                 all_trades: list = None,
                 buy_volume: float = 20,
                 sell_volume: float = 20,
                 intensity: float = None,
                 n_orders: int = 5,
                 orders_in_queue: int = 0,
                 price_margin: float = 0.1,
                 is_buying: bool = False):
        """
        Constructor
        :param latency: latency when matching agents in the market environment
        """
        self.agent_class = "Investor"
        self.agent_id = agent_id
        self.latency = latency
        self.position = position
        self.pnl = pnl
        self.buy_price = buy_price
        self.sell_price = sell_price
        self.all_trades = all_trades if all_trades else []
        self.buy_volume = buy_volume
        self.sell_volume = sell_volume
        self.intensity = intensity
        self.n_orders = n_orders
        self.orders_in_queue = orders_in_queue
        self.price_margin = price_margin
        self.is_buying = is_buying

    def calculate_buy_price(self, state: dict) -> float:
        """
        Calculates buy price

        :param state: market state information
        :return: buy price
        """
        buy_price = state["market_prices"][-1] * (1 + self.price_margin)
        buy_price = np.maximum(buy_price, 0)
        return buy_price

    def calculate_sell_price(self, state: dict) -> float:
        """
        Calculates sell price

        :param state: market state information
        :return: sell price
        """
        sell_price = state["market_prices"][-1]
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

        if execution_status[0] >= 1:
            buy_trade = - execution_status[0] * execution_status[2] - state["fee"] * execution_status[2]
            self.all_trades.append(buy_trade)
            self.position += execution_status[2]

        if execution_status[1] >= 1:
            sell_trade = execution_status[1] * execution_status[3] - state["fee"] * execution_status[3]
            self.all_trades.append(sell_trade)
            self.position -= execution_status[3]

        # instantiate no prices
        self.buy_price = np.nan
        self.sell_price = np.nan

        # check if investor wants to buy or sell and calculate prices if so
        will_buy = np.random.uniform(0, 1) < self.intensity
        will_sell = (np.random.uniform(0, 1) < self.intensity) and (self.position >= self.n_orders * self.sell_volume)

        if self.orders_in_queue == 0:
            self.is_buying = False

        if self.orders_in_queue > 0 and self.is_buying:  # buying
            self.orders_in_queue -= 1
            self.buy_price = self.calculate_buy_price(state)

        elif self.orders_in_queue > 0 and not self.is_buying:  # selling
            self.orders_in_queue -= 1
            self.sell_price = self.calculate_sell_price()

        elif will_buy and self.orders_in_queue == 0:  # starts to buy
            self.orders_in_queue = self.n_orders - 1
            self.is_buying = True
            self.buy_price = self.calculate_buy_price(state)

        elif will_sell and self.orders_in_queue == 0:  # starts to sell
            self.orders_in_queue = self.n_orders - 1
            self.sell_price = self.calculate_sell_price()

        # Update volume

        self.buy_volume = self.calculate_buy_volume(state)
        self.sell_volume = self.calculate_sell_volume(state)


class TrendAgent(Agent):
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
                 buy_volume: float = 0,
                 sell_volume: float = 0,
                 price_margin: float = 0.05,
                 const_position_size: int = 5,
                 moving_average_one: int = 50,
                 moving_average_two: int = 200):
        """
        Constructor
        :param latency: latency when matching agents in the market environment
        """
        self.agent_class = "Trend"
        self.agent_id = agent_id
        self.latency = latency
        self.position = position
        self.pnl = pnl
        self.buy_price = buy_price
        self.sell_price = sell_price
        self.all_trades = all_trades if all_trades else []
        self.buy_volume = buy_volume
        self.sell_volume = sell_volume
        self.price_margin = price_margin
        self.const_position_size = const_position_size
        self.moving_average_one = moving_average_one
        self.moving_average_two = moving_average_two

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
        volume = self.const_position_size # np.random.randint(1, 2)

        return volume

    def calculate_sell_volume(self, state: dict) -> float:
        """
        Calculates sell price

        :param state:
        :return:
        """
        volume = self.const_position_size # np.random.randint(1, 2)

        return volume

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

        if execution_status[0] >= 1:
            buy_trade = - execution_status[0] * execution_status[2] - state["fee"] * execution_status[2]
            self.all_trades.append(buy_trade)
            self.position += execution_status[2]

        if execution_status[1] >= 1:
            sell_trade = execution_status[1] * execution_status[3] - state["fee"] * execution_status[3]
            self.all_trades.append(sell_trade)
            self.position -= execution_status[3]

        # check trend direction and aim for strategic position

        ma1 = np.average(state["market_prices"][-self.moving_average_one:])
        ma2 = np.average(state["market_prices"][-self.moving_average_two:])

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
