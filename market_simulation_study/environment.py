from typing import NoReturn
import numpy as np
import pandas as pd
import random
from codelib.stats import weighted_percentile

#### TIME USAGE ######
import cProfile
import io
import pstats
import functools
#### END TIME USAGE ######

class MarketEnvironment:
    def __init__(self,
                 state: dict):
        self.state = state.copy()
        self.market_prices = state["market_prices"].copy()
        self.matched_volumes = state["volume"]
        self.fee = state["fee"]
        self.slippage = state["slippage"]
        self.agents = None

    def get_latencies(self):
        latencies = [0] * len(self.agents) 
        for i in range(len(self.agents)):
            latencies[i] = self.agents[i].latency
        #que_indices = range(len(self.agents))
        
        return latencies

    def get_agent_ids(self):
        agent_ids = [0] * len(self.agents)
        for i in range(len(self.agents)):
            agent_ids[i] = self.agents[i].agent_id
        return  agent_ids

    def calc_mean_order_prices(self) -> NoReturn:
        sell_prices, buy_prices = [0] * len(self.agents), [0] * len(self.agents)
        sell_volumes, buy_volumes = [0] * len(self.agents), [0] * len(self.agents)
        for i, agent in enumerate(self.agents):
            sell_prices[i] = agent.sell_price
            buy_prices[i] = agent.buy_price
            sell_volumes[i] = agent.sell_volume
            buy_volumes[i] = agent.buy_volume
            
        sell_price_list = np.array(sell_prices)
        weights = np.array(sell_volumes)
        mean_sell = np.ma.MaskedArray(sell_price_list, mask=np.isnan(sell_price_list))
        mean_sell = np.ma.average(mean_sell, weights=weights)
        self.mean_sell_price = mean_sell
        
        buy_price_list = np.array(buy_prices)
        weights = np.array(buy_volumes)
        mean_buy = np.ma.MaskedArray(buy_price_list, mask=np.isnan(buy_price_list))
        mean_buy = np.ma.average(mean_buy, weights=weights)
        self.mean_buy_price = mean_buy 
    
    def match(self):
        latencies = self.get_latencies()
        self.calc_mean_order_prices()
        matched_volume = []
        matched_price = []
        # Sort agents corresponding to latencies
        self.agents = [x for _, x in sorted(zip(latencies, self.agents))]  # Sorting buyers according to latency

        sell_order_book = self.agents[0].sell_order
        buy_order_book = self.agents[0].buy_order
        sell_order_book = pd.DataFrame(sell_order_book, index = sell_order_book.iloc[:, -1])
        buy_order_book = pd.DataFrame(buy_order_book, index = buy_order_book.iloc[:, -1])

        total_buy_volume = self.agents[0].buy_volume
        total_sell_volume = self.agents[0].sell_volume
        
        for i in range(len(self.agents)-1):
            total_buy_volume += self.agents[i+1].buy_volume
            total_sell_volume += self.agents[i+1].sell_volume
            #========================================#
            # CHECK IF AGENT i CAN MAKE A BUY TRADE #
            #========================================#
            if any(self.agents[i+1].buy_order["buy_price"].values >= sell_order_book.iloc[:, 0].values):                
                matched_order_book = sell_order_book[sell_order_book["sell_price"].values <= self.agents[i+1].buy_order["buy_price"].values]
                matched_order_book = matched_order_book.sort_values(["sell_price", "latency"], ascending = [True, True])

                for index, order in matched_order_book.iterrows():
                    if self.agents[i+1].buy_order["buy_volume"].values > order["sell_volume"]:
                        trade_volume = order["sell_volume"].copy()
                    else:
                        trade_volume = self.agents[i+1].buy_order["buy_volume"].values[0].copy()
                    
                    self.agents[i+1].buy_order["buy_volume"] -= trade_volume
                    sell_order_book.at[index, 'sell_volume'] -= trade_volume
                    trade_price = order["sell_price"]
                    buy_trade = np.array([trade_price, -trade_volume])
                    sell_trade = np.array([trade_price, trade_volume])

                    self.agents[i+1].position += trade_volume
                    self.agents[i + 1].all_trades = np.vstack((self.agents[i + 1].all_trades, buy_trade))

                    # Update agent who traded from order book position and trade history
                    for j in range(len(self.agents)):
                        if self.agents[j].agent_id == index:
                            self.agents[j].all_trades = np.vstack((self.agents[j].all_trades, sell_trade))
                            self.agents[j].position -= trade_volume

                    # UPDATE ALL MATCHED PRICES AND VOLUMES
                    matched_price.append(trade_price)
                    matched_volume.append(trade_volume)
            # SELL ORDER INTO SELL ORDER
            sell_order_book = sell_order_book[sell_order_book["sell_volume"] > 0]
            if self.agents[i+1].buy_order["buy_volume"].values > 0:
                buy_order_book = buy_order_book.append(self.agents[i+1].buy_order)
            
            sell_order_book = pd.DataFrame(sell_order_book, index = sell_order_book.iloc[:, -1])
            buy_order_book = pd.DataFrame(buy_order_book, index = buy_order_book.iloc[:, -1])
            
            #========================================#
            # CHECK IF AGENT i CAN MAKE A SELL TRADE #
            #========================================#
            if any(self.agents[i+1].sell_order["sell_price"].values <= buy_order_book.iloc[:, 0].values):
                matched_order_book = buy_order_book[buy_order_book["buy_price"].values >= self.agents[i+1].sell_order["sell_price"].values]
                matched_order_book = matched_order_book.sort_values(["buy_price", "latency"], ascending = [False, True])

                for index, order in matched_order_book.iterrows():
                    if self.agents[i+1].sell_order["sell_volume"].values > order["buy_volume"]:
                        trade_volume = order["buy_volume"].copy()
                    else:
                        trade_volume = self.agents[i+1].sell_order["sell_volume"].values[0].copy()
                    
                    self.agents[i+1].sell_order["sell_volume"] -= trade_volume
                    #self.agents[int(order["agent_id"])].buy_order["buy_volume"] -= trade_volume
                    buy_order_book.at[index, 'buy_volume'] -= trade_volume
                    trade_price = order["buy_price"]

                    buy_trade = np.array([trade_price, -trade_volume])
                    sell_trade = np.array([trade_price, trade_volume])

                    # Update agent i's position and trade history
                    self.agents[i + 1].all_trades = np.vstack((self.agents[i + 1].all_trades, sell_trade))
                    self.agents[i+1].position -= trade_volume

                    # Update agent who traded from order book position and trade history
                    for j in range(len(self.agents)):
                        if self.agents[j].agent_id == index:
                            self.agents[j].all_trades = np.vstack((self.agents[j].all_trades, buy_trade))
                            self.agents[j].position += trade_volume


                    # UPDATE ALL MATCHED PRICES AND VOLUMES
                    matched_price.append(trade_price)
                    matched_volume.append(trade_volume)
        
            # buy ORDER INTO buy ORDER
            buy_order_book = buy_order_book[buy_order_book["buy_volume"] > 0]
            if self.agents[i+1].sell_order["sell_volume"].values > 0:
                sell_order_book = sell_order_book.append(self.agents[i+1].sell_order)    
            
            sell_order_book = pd.DataFrame(sell_order_book, index = sell_order_book.iloc[:, -1])
            buy_order_book = pd.DataFrame(buy_order_book, index = buy_order_book.iloc[:, -1])        
        
            
        if np.sum(matched_volume) > 0:
            median_price = weighted_percentile(np.array(matched_price), p=0.5, probs=np.array(matched_volume))
        else:
            median_price = self.market_prices[-1]

        # Update prices and trade info
        self.market_prices.append(median_price)
        self.matched_volumes = np.sum(matched_volume)

        # Rearrange agents corresponding to agent ids
        agent_ids = self.get_agent_ids()
        self.agents = [x for _, x in sorted(zip(agent_ids, self.agents))]  # Sorting buyers according to latency
        self.total_buy_volume = total_buy_volume        
        self.total_sell_volume = total_sell_volume

    def update_market(self) -> NoReturn:

        self.state = {'volume': self.matched_volumes, # Total volume
                      'market_prices': self.market_prices,
                      'fee': self.fee,
                      'mean_buy_price': self.mean_buy_price,
                      'mean_sell_price': self.mean_sell_price,
                      'slippage': self.slippage,
                      'total_buy_volume': self.total_buy_volume,
                      'total_sell_volume': self.total_sell_volume}

    def step(self, agents: list) -> dict:
        self.agents = agents
        self.match()
        self.update_market()

        return self.agents, self.state
