from typing import NoReturn
import numpy as np
import pandas as pd
import random

class MarketEnvironment:
    def __init__(self,
                 state: dict):
        self.state = state
        self.market_prices = state["market_prices"]
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
        mean_price = 0
        agents_first, agents_second = np.array(self.agents), np.array(self.agents)
        agents_first = [x for _, x in sorted(zip(latencies, agents_first))]  # Sorting buyers according to latency
        agents_second = [x for _, x in sorted(zip(latencies, agents_second))]  # Sorting sellers according to latency
        total_market_volume = 0
        
        sell_order_book = self.agents[0].sell_order
        buy_order_book = self.agents[0].buy_order
        sell_order_book = pd.DataFrame(sell_order_book, index = sell_order_book.iloc[:, -1])
        buy_order_book = pd.DataFrame(buy_order_book, index = buy_order_book.iloc[:, -1])

        for i in range(len(self.agents)-1):
            #========================================#
            # CHECK IF AGENT i CAN MAKE A BUY TRADE #
            #========================================#
            if any(self.agents[i+1].buy_order["buy_price"].values >= sell_order_book.iloc[:, 0].values):
                #print("price_match: ", self.agents[i+1].buy_order["buy_price"].values, sell_order_book.iloc[:, 0].values)
                
                matched_order_book = sell_order_book[sell_order_book["sell_price"].values <= self.agents[i+1].buy_order["buy_price"].values]
                matched_order_book = matched_order_book.sort_values(["sell_price", "latency"], ascending = [True, True])

                for index, order in matched_order_book.iterrows():
                    if self.agents[i+1].buy_order["buy_volume"].values > order["sell_volume"]:
                        trade_volume = order["sell_volume"].copy()

                    else:
                        trade_volume = self.agents[i+1].buy_order["buy_volume"].values[0].copy()
                    
                    self.agents[i+1].buy_order["buy_volume"] -= trade_volume
                    #self.agents[int(order["agent_id"])].sell_order["sell_volume"] -= trade_volume
                    sell_order_book.at[index, 'sell_volume'] -= trade_volume
                    trade_price = order["sell_price"]
                    buy_trade = np.array([trade_price, trade_volume])
                    sell_trade = np.array([trade_price, -trade_volume])

                    self.agents[i+1].position += trade_volume
                    self.agents[i + 1].all_trades = np.vstack((self.agents[i + 1].all_trades, buy_trade))

                    # Update agent who traded from order book position and trade history
                    for j in range(len(self.agents)):
                        if self.agents[j].agent_id == index:
                            #self.agents[int(order["agent_id"])].position += trade_volume
                            #self.agents[int(order["agent_id"])].all_trades = np.vstack((self.agents[int(order["agent_id"])].all_trades, buy_trade))
                            self.agents[j].all_trades = np.vstack((self.agents[j].all_trades, sell_trade))
                            self.agents[j].position -= trade_volume

                    #self.agents[int(order["agent_id"])].position -= trade_volume
                    #self.agents[int(order["agent_id"])].all_trades = np.vstack((self.agents[int(order["agent_id"])].all_trades, buy_trade))

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
                #print("price_match: ", self.agents[i+1].sell_order["sell_price"].values, buy_order_book.iloc[:, 0].values)
                
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

                    buy_trade = np.array([trade_price, trade_volume])
                    sell_trade = np.array([trade_price, -trade_volume])

                    # Update agent i's position and trade history
                    self.agents[i + 1].all_trades = np.vstack((self.agents[i + 1].all_trades, sell_trade))
                    self.agents[i+1].position -= trade_volume

                    # Update agent who traded from order book position and trade history
                    for j in range(len(self.agents)):
                        if self.agents[j].agent_id == index:
                            #self.agents[int(order["agent_id"])].position += trade_volume
                            #self.agents[int(order["agent_id"])].all_trades = np.vstack((self.agents[int(order["agent_id"])].all_trades, buy_trade))
                            self.agents[j].all_trades = np.vstack((self.agents[j].all_trades, buy_trade))
                            self.agents[j].position += trade_volume


                    # UPDATE ALL MATCHED PRICES AND VOLUMES
                    matched_price.append(trade_price)
                    matched_volume.append(trade_volume)
                    # agents[sob["ID"]].all_trades.append([order["BP"], - trade_volume])
        
            # buy ORDER INTO buy ORDER
            buy_order_book = buy_order_book[buy_order_book["buy_volume"] > 0]
            if self.agents[i+1].sell_order["sell_volume"].values > 0:
                sell_order_book = sell_order_book.append(self.agents[i+1].sell_order)    
            
            sell_order_book = pd.DataFrame(sell_order_book, index = sell_order_book.iloc[:, -1])
            buy_order_book = pd.DataFrame(buy_order_book, index = buy_order_book.iloc[:, -1])        
        # for agent_one in agents_first:  # buy index
        #     agents_second_to_remove = []
        #     for j, agent_two in enumerate(agents_second):  # sell index
        #         """ 
        #         agent_one IS THE FIRST AGENT TO RUN THROUGH ALL OTHER AGENTS, FIRST CHECK IF THERE IS A BUY TRADE
        #         THEN IF THERE IS A SELL TRADE ON
        #         """
        #         if agent_one.agent_id != agent_two.agent_id:
        #             if agent_one.buy_price >= agent_two.sell_price:
        #                 if agent_one.buy_volume >= agent_two.sell_volume:
        #                     trade_volume = agent_two.sell_volume
        #                 #elif agent_one.buy_volume <= agent_two.sell_volume:
        #                 else:
        #                     trade_volume = agent_one.buy_volume
                        
        #                 agent_one.buy_volume -= trade_volume
        #                 agent_two.sell_volume -= trade_volume

        #                 trade_price = agent_one.buy_price #(buyer.buy_price + seller.sell_price) / 2
                        
        #                 # Assigning trade/price to calculate means
        #                 total_market_volume += trade_volume
        #                 matched_price.append(trade_price) #buyer.buy_price
        #                 matched_volume.append(trade_volume)
                        
        #                 # Assigning trades to the agents
        #                 buy_trade = np.array([trade_price, trade_volume])
        #                 sell_trade = np.array([trade_price, -trade_volume])

        #                 agent_one.all_trades = np.vstack((agent_one.all_trades, buy_trade))
        #                 agent_two.all_trades = np.vstack((agent_two.all_trades, sell_trade))

        #             if agent_one.sell_price <= agent_two.buy_price:
        #                 if agent_one.sell_volume >= agent_two.buy_volume:
        #                     trade_volume = agent_two.buy_volume
        #                 #elif agent_one.sell_volume <= agent_two.buy_volume:
        #                 else:
        #                     trade_volume = agent_one.sell_volume

        #                 agent_one.sell_volume -= trade_volume
        #                 agent_two.buy_volume -= trade_volume

        #                 trade_price = agent_one.sell_price #(buyer.buy_price + seller.sell_price) / 2
                        
        #                 # Assigning trade/price to calculate means
        #                 total_market_volume += trade_volume                        
        #                 matched_price.append(trade_price)
        #                 matched_volume.append(trade_volume)

        #                 # Assigning trades to the agents                        
        #                 buy_trade = np.array([trade_price, trade_volume])
        #                 sell_trade = np.array([trade_price, -trade_volume])

        #                 agent_one.all_trades = np.vstack((agent_one.all_trades, sell_trade))
        #                 agent_two.all_trades = np.vstack((agent_two.all_trades, buy_trade))
                    
        #             # ERROR CHECKING FOR VOLUME
        #             if agent_one.sell_volume < 0 or agent_one.buy_volume < 0:
        #                 print("ERROR IN VOLUME FOR AGENT_ONE: sell_vol", agent_one.sell_volume, 
        #                       " buy vol: ", agent_one.buy_volume)
        #             if agent_two.sell_volume < 0 or agent_two.buy_volume < 0:
        #                 print("ERROR IN VOLUME FOR AGENT_ONE: sell_vol", agent_two.sell_volume, 
        #                       " buy vol: ", agent_two.buy_volume)
                    
        #             if agent_two.sell_volume == 0 and agent_two.buy_volume == 0:
        #                 agents_second_to_remove.append(j)
        #             if agent_one.buy_volume == 0 and agent_one.sell_volume == 0:
        #                 break

        #     agents_second = np.delete(agents_second, agents_second_to_remove)
            
        if np.sum(matched_volume) > 0:
            #print("VOLUME", matched_volume)
            #print("PRICES", matched_price)
            #mean_price = np.sum(np.array(matched_price) * np.array(matched_volume)) / np.sum(matched_volume)
            mean_price = np.average(matched_price, weights = matched_volume)
            #print("MEAN PRICE", mean_price)
        else:
            mean_price = self.market_prices[-1]

        # Update prices and trade info
        self.market_prices.append(mean_price)
        #matched_trades = np.array([matched_buy_price, matched_sell_price, matched_buy_volume, matched_sell_volume])
        self.matched_volumes = np.sum(matched_volume)

    def update_market(self) -> NoReturn:

        self.state = {'volume': self.matched_volumes, # Total volume
                      'market_prices': self.market_prices,
                      'fee': self.fee,
                      'mean_buy_price': self.mean_buy_price,
                      'mean_sell_price': self.mean_sell_price,
                      'slippage': self.slippage}

    def step(self, agents: list) -> dict:
        self.agents = agents
        self.match()
        self.update_market()

        return self.agents, self.state
