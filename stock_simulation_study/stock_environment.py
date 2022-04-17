from typing import NoReturn

import numpy as np

"""
packages for dataload and simulation
"""
import ffn # data access
from arch.univariate import SkewStudent, EGARCH, ARX

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
        #self.state.append(self.sim_prices[self.time])
        self.state.append(self.sim_prices[self.time] - self.sim_prices[self.time-1])
        self.time += 1
        if self.time > 2:
            #reward = (self.state[-1] - self.state[-2]) * action
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
            #self.state.append(self.sim_prices[self.time])
            self.state.append(self.sim_prices[self.time] - self.sim_prices[self.time-1])
            self.time += 1
        
        return self.state, reward
    
    def simulate_prices(self) -> NoReturn:
        simulation = self.am.simulate(params = self.am_params, nobs = (self.n_time_points + 100)) / 100
        self.time = 0
        self.state = []
        
        #self.sim_prices = simulation.data.to_price_index()
        self.sim_prices = [100 + 20 * np.sin(zz) for zz in np.linspace(0,20,self.n_time_points + 100)]
        
    def model_setup(self) -> NoReturn:
        returns = self.prices.to_returns().dropna()
        rs = np.random.RandomState([self.time])
        dist = SkewStudent(random_state = rs)
        vol = EGARCH()
        lag = [1, 2, 3, 4, 5]
        
        # Model calibration
        self.am = ARX(returns * 100, lags = lag, volatility = vol, distribution = dist)
        self.am_params = self.am.fit().params
        
        

class StockEnvironmentTwo:
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
        #self.state.append(self.sim_prices[self.time])
        #self.state.append([self.sim_prices[self.time], self.sim_prices[self.time] - self.sim_prices[self.time-1]])
        if self.time >= len(self.sim_prices) - 1:
            self.done = True
            
        self.state = np.array([self.sim_prices[self.time], self.sim_prices[self.time] - self.sim_prices[self.time-1]])
        self.time += 1
        if self.time > 2:
            #reward = (self.state[-1]) * (action-1)#self.state[-1] * (action - 1)
            reward = (self.state[-1]) * (action-2)#self.state[-1] * (action - 1)
        else:
            reward = 0
#        self.state = np.asarray(self.state)
        state = self.state
        return state, reward, self.done, False
    
    def reset(self):
        self.done = False
        self.simulate_prices()
        for _ in range(100):
            #self.state.append(self.sim_prices[self.time])
            self.state = np.array([self.sim_prices[self.time], self.sim_prices[self.time] - self.sim_prices[self.time-1]])
            self.time += 1
        
        #self.state = np.asarray(self.state)
        return self.state#, reward
    
    def simulate_prices(self) -> NoReturn:
        #simulation = self.am.simulate(params = self.am_params, nobs = (self.n_time_points + 100)) / 100
        self.time = 0
        self.state = []
        
        #self.sim_prices = simulation.data.to_price_index()
        self.sim_prices = [np.sin(zz) for zz in np.linspace(0,100,self.n_time_points + 100)]
        
    def model_setup(self) -> NoReturn:
        returns = self.prices.to_returns().dropna()
        rs = np.random.RandomState([self.time])
        dist = SkewStudent(random_state = rs)
        vol = EGARCH()
        lag = [1, 2, 3, 4, 5]
        
        # Model calibration
        self.am = ARX(returns * 100, lags = lag, volatility = vol, distribution = dist)
        self.am_params = self.am.fit().params
        
