import numpy as np
import pandas as pd
import random

class market_env():
    def __init__(self, market_prices, fee):
        self.market_prices = market_prices
        self.fee = fee
        
    def get_latencies(self):
        latencies = [0] * len(self.agent) 
        for i in range(len(self.agent)):
            latencies[i] = self.agent.latency
        #que_indices = range(len(self.agent))
        
        return latencies
    
    def match(self):
        latencies = self.get_latencies()
        
        matched_buy = [0] * len(self.agent)
        matched_sell = [0] * len(self.agent)
        mean_price = 0
        buyers, sellers = np.array(self.agent).copy(), np.array(self.agent).copy()
        buyers = [x for _, x in sorted(zip(latencies, buyers))] # Sorting buyers according to latency
        sellers = [x for _, x in sorted(zip(latencies, sellers))] # Sorting sellers according to latency
        
        for i, buyer in enumerate(buyers): # buy index
            for j, seller in enumerate(sellers): # sell index
                if buyer.agent_id != seller.agent_id:
                    if buyer.buy_price >= seller.sell_price:
                        matched_buy[buyer.agent_id] = 1
                        matched_sell[seller.agent_id] = 1
                        
                        print("ORDER FILLED")
                        mean_price += (buyer.buy_price + seller.sell_price) / 2
                        
                        sellers = np.delete(sellers, 1)
                        break
                    else:
                        print("ORDER NOT FILLED")
        
        if len(matched_sell) > 0:
            mean_price /= (len(matched_sell))
        else:
            mean_price = self.market_prices[-1]
            
        print(mean_price)
        self.market_prices.append(mean_price)
        return matched_buy, matched_sell
    
    def update_market(self):
        
        return
    
    def step(self, agent):
        self.agent = agent
        matched_buy, matched_sell = self.match()
        self.update_market()
        state = [{'execution_status': np.array([matched_buy, matched_sell]),
                  'mid_prices': self.market_prices}]
        return state
    
    
    
    
    
    