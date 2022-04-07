from typing import NoReturn
import numpy as np
import pandas as pd
import random

class MarketEnvironment:
    def __init__(self,
                 state: dict):
        self.state = state
        self.market_prices = state["market_prices"]
        self.matched_trades = state["execution_status"]
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
        # latencies = self.get_latencies()
        # self.calc_mean_order_prices()
        
        # matched_buy_volume = [0] * len(self.agents)
        # matched_sell_volume = [0] * len(self.agents)
        # matched_buy_price = [0] * len(self.agents)
        # matched_sell_price = [0] * len(self.agents)
        # # ############## OLD STYLE ################  
        # mean_price, mean_buy, mean_sell = 0, 0, 0
        # n_trades = 0
        # buyers, sellers = np.array(self.agents), np.array(self.agents)
        # buyers = [x for _, x in sorted(zip(latencies, buyers))]  # Sorting buyers according to latency
        # sellers = [x for _, x in sorted(zip(latencies, sellers))]  # Sorting sellers according to latency

      
#         for i, buyer in enumerate(buyers):  # buy index
#             sellers_to_remove = []
#             for j, seller in enumerate(sellers):  # sell index
#                 # BUYER IS THE FIRST AGENT TO RUN THROUGH ALL OTHER AGENTS, FIRST CHECK IF THERE IS A BUY TRADE
#                 if buyer.agent_id != seller.agent_id:
#                     if buyer.buy_price >= seller.sell_price:
#                         if buyer.buy_volume >= seller.sell_volume:
#                             trade_volume = seller.sell_volume
#                             buyer.buy_volume -= trade_volume
#                             seller.sell_volume -= trade_volume
#                         elif buyer.buy_volume <= seller.sell_volume:
#                             trade_volume = buyer.buy_volume
#                             seller.sell_volume -= trade_volume
#                             buyer.buy_volume -= trade_volume
                        
#                         matched_buy_volume[buyer.agent_id] += trade_volume
#                         matched_sell_volume[seller.agent_id] += trade_volume

#                         trade_price = (buyer.buy_price + seller.sell_price) / 2#seller.sell_price
#                         matched_buy_price[buyer.agent_id] = trade_price #buyer.buy_price
#                         matched_sell_price[seller.agent_id] = trade_price #seller.sell_price
#                         # trade_price = seller.sell_price
#                         # matched_buy_price[buyer.agent_id] = buyer.buy_price
#                         # matched_sell_price[seller.agent_id] = seller.sell_price
#                         #buyer.all_trades = np.stack((buyer.all_trades, ))
                        
                        
#                         n_trades += 1
#                         if seller.sell_volume == 0:
#                             sellers_to_remove.append(j)
#                         if buyer.buy_volume == 0:
#                             break
# ########## HVER AGENT HAR EN self.all_trades, som skal appendes med køb og salg til givne priser
#             sellers = np.delete(sellers, sellers_to_remove)

############### NEW MATCH STYLE ######################
        latencies = self.get_latencies()
        self.calc_mean_order_prices()


        matched_volume = []
        matched_price = []
        # ############## OLD STYLE ################  
        mean_price = 0
        agents_first, agents_second = np.array(self.agents), np.array(self.agents)
        agents_first = [x for _, x in sorted(zip(latencies, agents_first))]  # Sorting buyers according to latency
        agents_second = [x for _, x in sorted(zip(latencies, agents_second))]  # Sorting sellers according to latency
        total_market_volume = 0
        for agent_one in agents_first:  # buy index
            agents_second_to_remove = []
            for j, agent_two in enumerate(agents_second):  # sell index
                """ 
                agent_one IS THE FIRST AGENT TO RUN THROUGH ALL OTHER AGENTS, FIRST CHECK IF THERE IS A BUY TRADE
                THEN IF THERE IS A SELL TRADE ON
                """
                if agent_one.agent_id != agent_two.agent_id:
                    if agent_one.buy_price >= agent_two.sell_price:
                        if agent_one.buy_volume >= agent_two.sell_volume:
                            trade_volume = agent_two.sell_volume
                        #elif agent_one.buy_volume <= agent_two.sell_volume:
                        else:
                            trade_volume = agent_one.buy_volume
                        
                        agent_one.buy_volume -= trade_volume
                        agent_two.sell_volume -= trade_volume

                        trade_price = agent_one.buy_price #(buyer.buy_price + seller.sell_price) / 2
                        
                        # Assigning trade/price to calculate means
                        total_market_volume += trade_volume
                        matched_price.append(trade_price) #buyer.buy_price
                        matched_volume.append(trade_volume)
                        
                        # Assigning trades to the agents
                        buy_trade = np.array([trade_price, trade_volume])
                        sell_trade = np.array([trade_price, -trade_volume])

                        agent_one.all_trades = np.vstack((agent_one.all_trades, buy_trade))
                        agent_two.all_trades = np.vstack((agent_two.all_trades, sell_trade))

                    if agent_one.sell_price <= agent_two.buy_price:
                        if agent_one.sell_volume >= agent_two.buy_volume:
                            trade_volume = agent_two.buy_volume
                        #elif agent_one.sell_volume <= agent_two.buy_volume:
                        else:
                            trade_volume = agent_one.sell_volume

                        agent_one.sell_volume -= trade_volume
                        agent_two.buy_volume -= trade_volume

                        trade_price = agent_one.sell_price #(buyer.buy_price + seller.sell_price) / 2
                        
                        # Assigning trade/price to calculate means
                        total_market_volume += trade_volume                        
                        matched_price.append(trade_price)
                        matched_volume.append(trade_volume)

                        # Assigning trades to the agents                        
                        buy_trade = np.array([trade_price, trade_volume])
                        sell_trade = np.array([trade_price, -trade_volume])

                        agent_one.all_trades = np.vstack((agent_one.all_trades, sell_trade))
                        agent_two.all_trades = np.vstack((agent_two.all_trades, buy_trade))
                    
                    # ERROR CHECKING FOR VOLUME
                    if agent_one.sell_volume < 0 or agent_one.buy_volume < 0:
                        print("ERROR IN VOLUME FOR AGENT_ONE: sell_vol", agent_one.sell_volume, 
                              " buy vol: ", agent_one.buy_volume)
                    if agent_two.sell_volume < 0 or agent_two.buy_volume < 0:
                        print("ERROR IN VOLUME FOR AGENT_ONE: sell_vol", agent_two.sell_volume, 
                              " buy vol: ", agent_two.buy_volume)
                    
                    if agent_two.sell_volume == 0 and agent_two.buy_volume == 0:
                        agents_second_to_remove.append(j)
                    if agent_one.buy_volume == 0 and agent_one.sell_volume == 0:
                        break

            agents_second = np.delete(agents_second, agents_second_to_remove)
            
        if np.sum(matched_volume) > 0:
            mean_price = np.sum(np.array(matched_price) * (np.array(matched_volume) / np.sum(matched_volume)))
        else:
            mean_price = self.market_prices[-1]

        # Update prices and trade info
        self.market_prices.append(mean_price)
        #matched_trades = np.array([matched_buy_price, matched_sell_price, matched_buy_volume, matched_sell_volume])
        self.matched_trades = np.array([total_market_volume])

    def update_market(self) -> NoReturn:

        self.state = {'execution_status': self.matched_trades, # Total volume
                      'market_prices': self.market_prices,
                      'fee': self.fee,
                      'mean_buy_price': self.mean_buy_price,
                      'mean_sell_price': self.mean_sell_price,
                      'slippage': self.slippage}

    def step(self, agents: list) -> dict:
        self.agents = agents
        self.match()
        self.update_market()

        return self.state
