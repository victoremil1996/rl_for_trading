import numpy as np
import pandas as pd
import random

class market_env():
    def __init__(self,
                 state: dict):
        self.state = state
        self.market_prices = state["market_prices"]
        self.fee = state["fee"]

    def get_latencies(self):
        latencies = [0] * len(self.agents) 
        for i in range(len(self.agents)):
            latencies[i] = self.agents[i].latency
        #que_indices = range(len(self.agents))
        
        return latencies
    
    def match(self):
        latencies = self.get_latencies()
        
        matched_buy = [0] * len(self.agents)
        matched_sell = [0] * len(self.agents)
        matched_buy_price = [0] * len(self.agents)
        matched_sell_price = [0] * len(self.agents)
        mean_price = 0
        n_trades = 0
        buyers, sellers = np.array(self.agents).copy(), np.array(self.agents).copy()
        buyers = [x for _, x in sorted(zip(latencies, buyers))] # Sorting buyers according to latency
        sellers = [x for _, x in sorted(zip(latencies, sellers))] # Sorting sellers according to latency
        
        for i, buyer in enumerate(buyers): # buy index
            for j, seller in enumerate(sellers): # sell index
                if buyer.agent_id != seller.agent_id:
                    if buyer.buy_price >= seller.sell_price:
                        matched_buy[buyer.agent_id] = 1
                        matched_sell[seller.agent_id] = 1

                        #print("ORDER FILLED")
                        trade_price = (buyer.buy_price + seller.sell_price) / 2
                        matched_buy_price[buyer.agent_id] = trade_price
                        matched_sell_price[seller.agent_id] = trade_price

                        mean_price += trade_price
                        n_trades += 1
                        sellers = np.delete(sellers, j)
                        break
                    #else:
                        #print("ORDER NOT FILLED")
        
        if np.sum(matched_sell) > 0:
            mean_price /= n_trades
        else:
            mean_price = self.market_prices[-1]

        self.market_prices.append(mean_price)
        return matched_buy, matched_sell, matched_buy_price, matched_sell_price
    
    def update_market(self):
        
        return
    
    def step(self, agents):
        self.agents = agents
        matched_buy, matched_sell, matched_buy_price, matched_sell_price = self.match()
        self.update_market()
        state = {'execution_status': np.array([matched_buy, matched_sell, matched_buy_price, matched_sell_price]),
                 'market_prices': self.market_prices,
                 'fee': self.fee}
        return state
