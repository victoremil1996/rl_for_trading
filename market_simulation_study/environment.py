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
        
        matched_buy_volume = [0] * len(self.agents)
        matched_sell_volume = [0] * len(self.agents)
        matched_buy_price = [0] * len(self.agents)
        matched_sell_price = [0] * len(self.agents)
        mean_price, mean_buy, mean_sell = 0, 0, 0
        n_trades = 0
        buyers, sellers = np.array(self.agents).copy(), np.array(self.agents).copy()
        buyers = [x for _, x in sorted(zip(latencies, buyers))]  # Sorting buyers according to latency
        sellers = [x for _, x in sorted(zip(latencies, sellers))]  # Sorting sellers according to latency
        
        for i, buyer in enumerate(buyers):  # buy index
            sellers_to_remove = []
            for j, seller in enumerate(sellers):  # sell index
                if buyer.agent_id != seller.agent_id:
                    if buyer.buy_price >= seller.sell_price:
                        if buyer.buy_volume >= seller.sell_volume:
                            trade_volume = seller.sell_volume
                            buyer.buy_volume -= trade_volume
                            seller.sell_volume -= trade_volume
                        elif buyer.buy_volume <= seller.sell_volume:
                            trade_volume = buyer.buy_volume
                            seller.sell_volume -= trade_volume
                            buyer.buy_volume -= trade_volume
                        
                        matched_buy_volume[buyer.agent_id] += trade_volume
                        matched_sell_volume[seller.agent_id] += trade_volume

                        trade_price = (buyer.buy_price + seller.sell_price) / 2#seller.sell_price
                        matched_buy_price[buyer.agent_id] = buyer.buy_price#buyer.buy_price
                        matched_sell_price[seller.agent_id] = seller.sell_price#seller.sell_price
                        # trade_price = seller.sell_price
                        # matched_buy_price[buyer.agent_id] = buyer.buy_price
                        # matched_sell_price[seller.agent_id] = seller.sell_price
                        
                        n_trades += 1
                        if seller.sell_volume == 0:
                            sellers_to_remove.append(j)
                        if buyer.buy_volume == 0:
                            break

            sellers = np.delete(sellers, sellers_to_remove)
        print("TOTAL BUY VOLUME", np.sum(matched_buy_volume))
        print("TOTAL SELL VOLUME", np.sum(matched_sell_volume))
        if np.sum(matched_sell_volume) > 0:
            mean_price = np.sum(np.array(matched_sell_price) * (np.array(matched_sell_volume) / np.sum(matched_sell_volume)))
        else:
            mean_price = self.market_prices[-1]

        # Update prices and trade info
        self.market_prices.append(mean_price)
        matched_trades = np.array([matched_buy_volume, matched_sell_volume, matched_buy_price, matched_sell_price])
        self.matched_trades = matched_trades

    def update_market(self) -> NoReturn:

        self.state = {'execution_status': self.matched_trades,
                      'market_prices': self.market_prices,
                      'fee': self.fee,
                      'mean_buy_price': self.mean_buy_price,
                      'mean_sell_price': self.mean_sell_price}

    def step(self, agents: list) -> dict:
        self.agents = agents
        self.match()
        self.update_market()

        return self.state
