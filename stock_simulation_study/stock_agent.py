from typing import NoReturn, Tuple
from numpy import ndarray
import abc
import numpy as np
import pandas as pd
import random

from keras.models import Sequential
from keras.layers import Dense, SimpleRNN, LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
#from keras import optimizers
from tensorflow import keras
import math

class RLStockAgent():
    """
    RL Agent which trades according to a RNN 
    """

    def __init__(self,
#                 agent_id: int = None,
                 delta: float = None,
                 position: int = 0,
                 pnl: float = None,
                 buy_price: float = None,
                 sell_price: float = None,
                 all_trades: np.ndarray = None,
                 buy_volume: float = None,
                 sell_volume: float = None,
                 epsilon: float = 0.2,
                 state_size: int = 2,
                 lr: float = 0.01,
                 alpha: float = 0.6,
                 noise_range: Tuple = [0.01, 0.1],
                 gamma: float = 0.99,
                 train_batch_size: int = 60,
                 nn_parameters: dict = {"n_layers": 10, "n_units": 20, "n_features": 2, "batch_size": 1,
                                        "dense_units": 1, "n_timepoints": 16, "n_epochs": 20},
                 function_approximator: str = "rnn"):

        """
        Constructor
        :param latency: latency when matching agents in the market environment
        """
        self.agent_class = "RL"
#        self.agent_id = agent_id
        self.delta = delta
        self.latency = delta
        self.position = position
        self.step_pnl = pnl
        self.total_pnl = 0
        self.all_trades = all_trades if all_trades else np.array([0, 0])
        self.random_agent_price = None
        self.epsilon = epsilon
        self.lr = lr
        self.alpha = alpha
        self.state_size = state_size
        self.gamma = gamma
        self.nn_parameters = nn_parameters
        self.data = np.ndarray(shape = (0, 6))
        if function_approximator == "rnn":
            self.model = self.rnn_model()
        self.position = 0
        self.train_batch_size = train_batch_size
        self.predicted_reward = 0

    def calculate_profit_and_loss(self, state) -> NoReturn:
        """
        Calculates profit and loss

        :param state: market state information
        :return: total profit and loss
        """
        #realized_value = np.sum(self.all_trades[:, 0] * self.all_trades[:, 1])
        #step_return = self.position * (state[-1] - state[-2])
        #self.step_pnl = step_return
        self.total_pnl += self.step_pnl

    def create_features(self, state):
        ma1 = np.average(state[-5:])
        ma2 = np.average(state[-20:])
        trend_feature = ma1 / ma2
        feature_list = [ma1, ma2, trend_feature]
        return feature_list
    
    def store_data(self, state) -> NoReturn:
        self.calculate_profit_and_loss(state)
        features = self.create_features(state)
        #reward = state[-1] - state[-2] - self.predicted_reward
        reward = self.step_pnl - self.predicted_reward
        data = np.array([features[0], features[1], features[2], 
                         state[-1], self.position, reward])#self.step_pnl])
        self.data = np.vstack((self.data, data))

    def scale_and_reshape(self, data):
        feature_vector = data.iloc[:, :-1]
        #col_names = feature_vector.columns
        #scaler = MinMaxScaler(feature_range=(0, 1))
        #feature_vector = pd.DataFrame(scaler.fit_transform(feature_vector), columns = col_names)
        # RESHAPING
        y_ind = np.arange(self.nn_parameters["n_timepoints"], len(data.iloc[:, -1]), self.nn_parameters["n_timepoints"])
        y = data.iloc[:, -1][y_ind]
        rows_x = len(y)
        x = feature_vector.iloc[range(self.nn_parameters["n_timepoints"] * rows_x), :]
        x = x.to_numpy()
        feature_vector_reshaped = np.reshape(x, (rows_x, self.nn_parameters["n_timepoints"], self.nn_parameters["n_features"]))
        return feature_vector_reshaped
        
    def train_model(self):
        data = pd.DataFrame(self.data, columns = ['ma1', 'ma2', 'trend_feature',
                                                  'stock_prices', 'action', 'reward'])
        reward_vector = data.iloc[:, -1].values
        
        disc_reward = np.zeros_like(reward_vector)
        cum_reward = 0
        for i in reversed(range(len(reward_vector))):
            cum_reward = cum_reward * self.gamma + reward_vector[i]
            disc_reward[i] = cum_reward
        
        disc_reward = reward_vector
        
        y_ind = np.arange(self.nn_parameters["n_timepoints"], len(disc_reward), self.nn_parameters["n_timepoints"])
        disc_reward = np.array(disc_reward[y_ind])
        data = data[["stock_prices", "action", "reward"]]
        feature_vector_reshaped = self.scale_and_reshape(data)
        
        # TRAIN
        self.model.fit(feature_vector_reshaped, disc_reward, epochs=self.nn_parameters["n_epochs"], batch_size=1, verbose=2)
        # for i in range(self.train_batch_size, feature_vector_reshaped.shape[0]):
        #     train_batch = np.reshape(feature_vector_reshaped[i,:,:], (1, self.nn_parameters["n_timepoints"], self.nn_parameters["n_features"]))
        #     self.model.train_on_batch(train_batch, np.reshape(disc_reward[i], (1,)), reset_metrics = False)

    
    def rnn_model(self):
        model = Sequential()
        model.add(LSTM(self.nn_parameters["n_units"], batch_input_shape=(self.nn_parameters["batch_size"], self.nn_parameters["n_timepoints"], self.nn_parameters["n_features"]), stateful=True, return_sequences=True))
        model.add(LSTM(self.nn_parameters["n_units"], batch_input_shape=(self.nn_parameters["batch_size"], self.nn_parameters["n_timepoints"], self.nn_parameters["n_features"]), stateful=True))
        model.add(Dense(units=self.nn_parameters["dense_units"], activation="linear"))
        model.compile(loss='mean_squared_error', optimizer="adam")#keras.optimizers.Adam(learning_rate=self.lr))
        return model
   
    def predict_state(self):
        state_feature_vector = self.data[-(self.nn_parameters["n_timepoints"]*2):, :]
        state_feature_vector = state_feature_vector[:, (-3, -2, -1)]
        #display("state_feat: ", state_feature_vector[-1,:])
        #display("state", self.new_state[-1])
        action_list = [-1, 0, 1]
        predictions = []
        # TODO CREATE VECTOR INSTEAD OF FORLOOP
        for act in action_list:
            state_feature_vector[:,1] = act
            data = pd.DataFrame(state_feature_vector)
            state_feature_vector_reshaped = self.scale_and_reshape(data)
            #print(state_feature_vector_reshaped.shape)
            predictions.append(self.model.predict(state_feature_vector_reshaped, batch_size = 1))
        #display("preds: ", predictions)
        action = action_list[np.argmax(predictions)]
        self.predicted_reward = np.max(predictions)
        return action
    
    def take_action(self, random_action = False):
        if random_action == True or random.random() < self.epsilon:
            self.action = random.randint(-1, 1)
            self.position = self.action
        else:    
            self.action = self.predict_state()
            self.position = self.action
        
    def reset_data(self):
        self.data = np.ndarray(shape = (0, self.nn_parameters["n_features"] + 4))
        self.total_pnl = 0
        self.position = 0
        
    def update(self, state, reward, random_action = False) -> NoReturn:
        """
        Updates agents position
        :param state: market state information
        :return: NoReturn
        """
        self.step_pnl = reward
        self.new_state = state
        self.take_action(random_action)
        self.store_data(state)
        
        
class RLStockAgentTwo():
    """
    RL Agent which trades according to a RNN 
    """

    def __init__(self,
#                 agent_id: int = None,
                 delta: float = None,
                 position: int = 0,
                 pnl: float = None,
                 buy_price: float = None,
                 sell_price: float = None,
                 all_trades: np.ndarray = None,
                 buy_volume: float = None,
                 sell_volume: float = None,
                 epsilon: float = 0.2,
                 state_size: int = 2,
                 lr: float = 0.01,
                 alpha: float = 0.6,
                 noise_range: Tuple = [0.01, 0.1],
                 gamma: float = 0.99,
                 train_batch_size: int = 60,
                 nn_parameters: dict = {"n_layers": 10, "n_units": 20, "n_features": 2, "batch_size": 1,
                                        "dense_units": 1, "n_timepoints": 16, "n_epochs": 20},
                 function_approximator: str = "rnn"):

        """
        Constructor
        :param latency: latency when matching agents in the market environment
        """
        self.agent_class = "RL"
#        self.agent_id = agent_id
        self.delta = delta
        self.latency = delta
        self.position = position
        self.step_pnl = pnl
        self.total_pnl = 0
        self.all_trades = all_trades if all_trades else np.array([0, 0])
        self.random_agent_price = None
        self.epsilon = epsilon
        self.lr = lr
        self.alpha = alpha
        self.state_size = state_size
        self.gamma = gamma
        self.nn_parameters = nn_parameters
        self.data = np.ndarray(shape = (0, 6))
        if function_approximator == "rnn":
            self.model = self.rnn_model()
        self.position = 0
        self.train_batch_size = train_batch_size
        self.predicted_reward = 0

    def calculate_profit_and_loss(self, state) -> NoReturn:
        """
        Calculates profit and loss

        :param state: market state information
        :return: total profit and loss
        """
        #realized_value = np.sum(self.all_trades[:, 0] * self.all_trades[:, 1])
        #step_return = self.position * (state[-1] - state[-2])
        #self.step_pnl = step_return
        self.total_pnl += self.step_pnl

    def store_data(self, state) -> NoReturn:
        self.calculate_profit_and_loss(state)
        features = state
        #reward = state[-1] - state[-2] - self.predicted_reward
        reward = self.step_pnl - self.predicted_reward
        data = np.array([state[-1], self.position, reward])#self.step_pnl])
        self.data = np.vstack((self.data, data))

    def scale_and_reshape(self, data):
        feature_vector = data.iloc[:, :-1]
        #col_names = feature_vector.columns
        #scaler = MinMaxScaler(feature_range=(0, 1))
        #feature_vector = pd.DataFrame(scaler.fit_transform(feature_vector), columns = col_names)
        # RESHAPING
        y_ind = np.arange(self.nn_parameters["n_timepoints"], len(data.iloc[:, -1]), self.nn_parameters["n_timepoints"])
        y = data.iloc[:, -1][y_ind]
        rows_x = len(y)
        x = feature_vector.iloc[range(self.nn_parameters["n_timepoints"] * rows_x), :]
        x = x.to_numpy()
        feature_vector_reshaped = np.reshape(x, (rows_x, self.nn_parameters["n_timepoints"], self.nn_parameters["n_features"]))
        return feature_vector_reshaped
        
    def train_model(self):
        data = pd.DataFrame(self.data, columns = ['ma1', 'ma2', 'trend_feature',
                                                  'stock_prices', 'action', 'reward'])
        reward_vector = data.iloc[:, -1].values
        
        disc_reward = np.zeros_like(reward_vector)
        cum_reward = 0
        for i in reversed(range(len(reward_vector))):
            cum_reward = cum_reward * self.gamma + reward_vector[i]
            disc_reward[i] = cum_reward
        
        disc_reward = reward_vector
        
        y_ind = np.arange(self.nn_parameters["n_timepoints"], len(disc_reward), self.nn_parameters["n_timepoints"])
        disc_reward = np.array(disc_reward[y_ind])
        data = data[["stock_prices", "action", "reward"]]
        feature_vector_reshaped = self.scale_and_reshape(data)
        
        # TRAIN
        self.model.fit(feature_vector_reshaped, disc_reward, epochs=self.nn_parameters["n_epochs"], batch_size=1, verbose=2)
        # for i in range(self.train_batch_size, feature_vector_reshaped.shape[0]):
        #     train_batch = np.reshape(feature_vector_reshaped[i,:,:], (1, self.nn_parameters["n_timepoints"], self.nn_parameters["n_features"]))
        #     self.model.train_on_batch(train_batch, np.reshape(disc_reward[i], (1,)), reset_metrics = False)

    
    def rnn_model(self):
        def custom_loss(y_true, y_pred):
            #y_pred_clip = K.clip(y_pred, 1e-8, 1-1e-8)
            abs_max = np.amax(np.abs(y_pred))
            y_pred_clip = y_pred * (1.0 / abs_max)
            log_likelihood = y_true*np.log(y_pred_clip)
            return np.sum(-log_likelihood*discounted_rewards)
        
        model = Sequential()
        model.add(LSTM(self.nn_parameters["n_units"], batch_input_shape=(self.nn_parameters["batch_size"], self.nn_parameters["n_timepoints"], self.nn_parameters["n_features"]), stateful=True, return_sequences=True))
        model.add(LSTM(self.nn_parameters["n_units"], batch_input_shape=(self.nn_parameters["batch_size"], self.nn_parameters["n_timepoints"], self.nn_parameters["n_features"]), stateful=True))
        model.add(Dense(units=self.nn_parameters["dense_units"], activation="linear"))
        model.compile(loss='mean_squared_error', optimizer="adam")#keras.optimizers.Adam(learning_rate=self.lr))
        return model
   
    def predict_state(self):
        state_feature_vector = self.data[-(self.nn_parameters["n_timepoints"]*2):, :]
        state_feature_vector = state_feature_vector[:, (-3, -2, -1)]
        #display("state_feat: ", state_feature_vector[-1,:])
        #display("state", self.new_state[-1])
        action_list = [-1, 0, 1]
        predictions = []
        # TODO CREATE VECTOR INSTEAD OF FORLOOP
        for act in action_list:
            state_feature_vector[:,1] = act
            data = pd.DataFrame(state_feature_vector)
            state_feature_vector_reshaped = self.scale_and_reshape(data)
            #print(state_feature_vector_reshaped.shape)
            predictions.append(self.model.predict(state_feature_vector_reshaped, batch_size = 1))
        #display("preds: ", predictions)
        action = action_list[np.argmax(predictions)]
        self.predicted_reward = np.max(predictions)
        return action
    
    def take_action(self, random_action = False):
        if random_action == True or random.random() < self.epsilon:
            self.action = random.randint(-1, 1)
            self.position = self.action
        else:    
            self.action = self.predict_state()
            self.position = self.action
        
    def reset_data(self):
        self.data = np.ndarray(shape = (0, self.nn_parameters["n_features"] + 4))
        self.total_pnl = 0
        self.position = 0
        
    def update(self, state, reward, random_action = False) -> NoReturn:
        """
        Updates agents position
        :param state: market state information
        :return: NoReturn
        """
        self.step_pnl = reward
        self.new_state = state
        self.take_action(random_action)
        self.store_data(state)
        
        
        
        
        
        
