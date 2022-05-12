# Thesis in reinforcement learning by LJBCPH and victoremil1996
Master thesis in reinforcement learning studying its applications in market making using agent based modelling and signal trading using market data.


## Multi Agent-Based Model Simulation of a Pseudo financial market with Reinforcement Learning
We model the underlying dynamics of the orderbook, through a MarketEnvironment class and agents who acts as market participants. More agents can easily be implemented by inhereting the core attributes from the Agent parent class. In general an agent needs to submit buy price, sell price, buy volume and sell volume at each timer interval. Implemented is an Advantage-Actor-Critic RL agent with neural network function approximators.  

Simulation studies and reinforcement learning training can be found in the notebooks, in the folder notebooks.

## RL for stock trading

We train an RL agent to trade simulated stock prices using a AR + GARCH model estimated from S&P 500 data. The implemented agent utilizing the REINFORCE Monte Carlo method. 

Simulation studies and backtest of algorithm can be found in the notebooks folder.
