import random
from datetime import datetime
import time
#import resource
import pickle
import os
import pdb

import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.regularizers import l2
import tensorflow.keras.backend as K

# originally built on old TensorFlow and Keras which didn't support eager execution
tf.compat.v1.disable_eager_execution()

import plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from IPython.display import clear_output, display, HTML

from matplotlib import pyplot as plt
import seaborn as sns

# set seeds for reproducibility
# np.random.uniform(0,10000) 4465
#random.seed(4465)
#np.random.seed(4465)
#tf.random.set_seed(4465)

print("TensorFlow %s" % tf.__version__)
print("Keras %s" % keras.__version__)
print("plotly %s" % plotly.__version__)
print("pandas %s" % pd.__version__)
print("numpy %s" % np.__version__)

# If model save directory isn't made yet, make it
if not os.path.exists('model_output'):
    os.makedirs('model_output')
if not os.path.exists('model_output/trading'):
    os.makedirs('model_output/trading')

# show memory usage (some versions of TensorFlow gave memory issues)
def sizeof_fmt(num, suffix='B'):
    """given memory as int format as memory units eg KB"""
    for unit in ['', 'K', 'M', 'G', 'T', 'P', 'E', 'Z']:
        if abs(num) < 1024.0:
            return "%3.1f %s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f %s%s" % (num, 'Y', suffix)

# def memusage():
#     """print memory usage"""
#     return sizeof_fmt(int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))


RENDER = False
OUTPUT_DIR = 'model_output/trading/'

class Agent:
    """abstract base class for agents"""

    def __init__(self, state_size, action_size, filename="model",
                 *args, **kwargs):
        self.state_size = state_size
        self.action_size = action_size
        self.filename = filename
        self.timestep = 0
        self.total_reward = 0
        self.save_interval = 10

        raise NotImplementedError

    def build_model(self, *args, **kwargs):
        """build the relevant model"""
        raise NotImplementedError

    def reset(self):
        """reset agent for start of episode"""
        self.timestep = 0
        self.total_reward = 0

    def increment_time(self):
        """increment timestep counter"""
        self.timestep += 1

    def remember(self, *args, **kwargs):
        """store the states and rewards needed to fit the model"""
        raise NotImplementedError

    def train(self, *args, **kwargs):
        """train the model on experience stored by remember"""
        raise NotImplementedError

    def act(self, *args, **kwargs):
        """pick an action using model"""
        raise NotImplementedError

    def save_score(self):
        """save score of each episode"""
        self.results.append(self.total_reward)

    def score_episode(self, episode_num, n_episodes):
        """output results and save"""
        self.save_score()
        avglen = min(len(self.results), self.save_interval)
        formatstr = "{} episode {}/{}:, score: {}, {}-episode avg: {:.1f} Memory: {}        "
        print(formatstr.format(time.strftime("%H:%M:%S"), len(self.results),
                               n_episodes, self.total_reward, avglen,
                               sum(self.results[-avglen:])/avglen, 0),
              end="\r", flush=False)

    def run_episode(self, env, render=RENDER):
        """run a full episode"""
        #global env

        self.reset()
        self.state = env.reset(self.state_size)
        self.done = False

        while not self.done:
            if render:
                env.render()
            self.action = self.act(self.state.reshape([1, self.state_size]), env)
            self.next_state, self.reward, self.done, _ = env.step(self.action, self.state_size, self.mu_zero)
            self.total_reward += self.reward

            self.remember()
            self.state = self.next_state
            self.increment_time()
            
        if render:
            env.render()
            
        self.train()
   
    def save(self, *args, **kwargs):
        """save agent to disk"""
        raise NotImplementedError

    def load(*args, **kwargs):
        """load agent from disk"""
        raise NotImplementedError

    # def view(self):
    #     """Run an episode without training, with rendering"""
    #     state = env.reset()
    #     state = np.reshape(state, [1, self.state_size])
    #     done = False

    #     # run an episode
    #     self.timestep = 0
    #     r = 0
    #     nstocks = 1
    #     while not done:
    #         env.render()
    #         action = self.act(state)
    #         lastmarket = self.state[0, nstocks-1]
    #         state, reward, done, _ = env.step(action)
    #         newmarket = self.state[0, nstocks-1]
    #         print("prev mkt: %.4f action: %d, new mkt %f, reward %f" % (lastmarket, action, newmarket, reward))
    #         r += reward
    #         state = np.reshape(state, [1, self.state_size])
    #         self.timestep += 1
    #     env.render()
    #     print(r)
    #     env.close()
    #     return self.timestep

    def rlplot(self, title='Trading Agent Training Progress'):
        """plot training progress"""
        df = pd.DataFrame({'timesteps': self.results})
        df['avg'] = df['timesteps'].rolling(10).mean()

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index,
                                 y=df['timesteps'],
                                 mode='markers',
                                 name='timesteps',
                                 marker=dict(
                                     color='mediumblue',
                                     size=4,
                                 ),
                                ))

        fig.add_trace(go.Scatter(x=df.index,
                                 y=df['avg'],
                                 mode='lines',
                                 line_width=3,
                                 name='moving average'))

        fig.update_layout(
            title=dict(text=title,
                       x=0.5,
                       xanchor='center'),
            xaxis=dict(
                title="Episodes",
                linecolor='black',
                linewidth=1,
                mirror=True
            ),
            yaxis=dict(
                title="Total Reward per Episode",
                linecolor='black',
                linewidth=1,
                mirror=True
            ),
            legend=go.layout.Legend(
                x=0.01,
                y=0.99,
                traceorder="normal",
                font=dict(
                    family="sans-serif",
                    size=12,
                    color="black"
                ),
                #bgcolor="LightSteelBlue",
                bordercolor="Black",
                borderwidth=1,
            ),
        )

        return fig.show()


class REINFORCE_Agent(Agent):
    """REINFORCE policy gradient method using deep Keras NN"""
    def __init__(self, state_size=4, action_size=5, learning_rate=0.0005,
                 discount_rate=0, n_hidden_layers=2, hidden_layer_size=16,
                 activation='relu', reg_penalty=0, bias_reg=0, dropout=0, filename="kreinforce",
                 verbose=True, epsilon = 0.1, mu_zero = False):
        self.state_size = state_size
        self.action_size = action_size
        self.action_space = list(range(action_size))
        self.learning_rate = learning_rate
        self.discount_rate = discount_rate

        self.n_hidden_layers = n_hidden_layers
        self.hidden_layer_size = hidden_layer_size
        self.activation = activation
        self.reg_penalty = reg_penalty
        self.bias_reg = bias_reg
        self.dropout = dropout
        self.verbose = verbose
        self.filename = filename

        self.train_model, self.predict_model = self.policy_model()
        self.results = []
        self.save_interval = 10
        self.reset()
        self.epsilon = epsilon
        self.mu_zero = mu_zero

    def reset(self):
        """reset agent for start of episode"""
        self.timestep = 0
        # truncate memory
        self.state_memory = []
        self.action_memory = []
        self.reward_memory = []
        self.total_reward = 0

    def policy_model(self):
        """set up NN model for policy.
        predict returns probs of actions to sample from.
        train needs discounted rewards for the episode, so we define custom loss.
        when training use train_model with custom loss and multi input of training data and rewards.
        when predicting use predict_model with single input.
        """
        
        def custom_loss(y_true, y_pred):
            y_pred_clip = K.clip(y_pred, 1e-8, 1-1e-8)
            log_likelihood = y_true*K.log(y_pred_clip)
            return K.sum(-log_likelihood*discounted_rewards)

        inputs = Input(shape=(self.state_size,), name="Input")
        discounted_rewards = Input(shape=(1,), name="Discounted_rewards")
        last_layer = inputs

        for i in range(self.n_hidden_layers):
            if self.verbose:
                formatstr = "layer %d size %d, %s, reg_penalty %.8f, dropout %.3f"
                print(formatstr % (i + 1,
                                   self.hidden_layer_size,
                                   self.activation,
                                   self.reg_penalty,
                                   self.dropout,
                                   ))
            # add dropout, but not on inputs, only between hidden layers
            if i > 0 and self.dropout:
                last_layer = Dropout(self.dropout, name="Dropout%02d" % i)(last_layer)

            #last_layer = Dense(units=self.hidden_layer_size,
            last_layer = Dense(units=int(self.hidden_layer_size / (i + 1)),
                               activation=self.activation,
                               kernel_initializer=glorot_uniform(),
                               kernel_regularizer=keras.regularizers.L2(self.reg_penalty),
                               bias_regularizer=keras.regularizers.L2(self.bias_reg),
                               name="Dense%02d" % i)(last_layer)

        outputs = Dense(self.action_size, activation='softmax', name="Output")(last_layer)
        train_model = Model(inputs=[inputs, discounted_rewards], outputs=[outputs])
        train_model.compile(optimizer=Adam(learning_rate=self.learning_rate), loss=custom_loss)

        predict_model = Model(inputs=[inputs], outputs=[outputs])

        if self.verbose:
            print(predict_model.summary())

        return train_model, predict_model

    def act(self, state, env):
        """pick an action using predict_model"""
        probabilities = self.predict_model.predict(state)
        if random.random() < self.epsilon:
            min_rand = 0
            max_rand = 4
            if env.pos_size >= 10:
                max_rand = 2
            elif env.pos_size <= -10:
                min_rand = 2
            action = random.randint(min_rand, max_rand)
        else:
            if env.pos_size <= -10:
                actions = self.action_space[2:]
                probs = probabilities[0][2:] / np.sum(probabilities[0][2:])
                if np.any(np.isnan(probs)):
                    probs = [1, 0, 0]
                action = np.random.choice(actions, p=probs)
            elif env.pos_size >= 10:
                actions = self.action_space[:3]
                probs = probabilities[0][:3] / np.sum(probabilities[0][:3])
                if np.any(np.isnan(probs)):
                    probs = [0, 0, 1]
                action = np.random.choice(actions, p=probs)
            else:
                action = np.random.choice(self.action_space, p=probabilities[0])


        return action

    def remember(self):
        """at each step save state, action, reward for future training"""
        
        self.state_memory.append(self.state)
        self.action_memory.append(self.action)
        self.reward_memory.append(self.reward)

    def train(self):
        """train the model on experience stored by remember"""
        state_memory = np.array(self.state_memory)
        state_memory = state_memory.reshape((len(self.state_memory),self.state_size))
        action_memory = np.array(self.action_memory)
        reward_memory = np.array(self.reward_memory)

        # one-hot actions
        actions = np.zeros([len(action_memory), self.action_size])
        actions[np.arange(len(action_memory)), action_memory] = 1

        disc_rewards = np.zeros_like(reward_memory)
        cumulative_rewards = 0
        for i in reversed(range(len(reward_memory))):
            cumulative_rewards = cumulative_rewards * self.discount_rate + reward_memory[i]
            disc_rewards[i] = cumulative_rewards

        # standardize
        disc_rewards -= np.mean(disc_rewards)
        disc_rewards /= np.std(disc_rewards) if np.std(disc_rewards) > 0 else 1

        # train states v. actions, (complemented by disc_rewards_std)
        cost = self.train_model.train_on_batch([state_memory, disc_rewards], actions)

        return cost

    # def view(self):
    #     """Run an episode without training, with rendering"""
    #     state = env.reset()
    #     state = np.reshape(state, [1, self.state_size])
    #     done = False

    #     # run an episode
    #     self.timestep = 0
    #     r = 0
    #     retarray = []
    #     while not done:
    #         action = self.act(state)
    #         lastmarket = state[0, self.state_size//2-1]
    #         state, reward, done, _ = env.step(action)
    #         newmarket = state[0, self.state_size//2-1]
    #         print("prev mkt: %.4f action: %d, new mkt %.4f, reward %f" % (lastmarket, action, newmarket, reward))
    #         r += reward
    #         state = np.reshape(state, [1, self.state_size])
    #         self.timestep += 1
    #         retarray.append((self.timestep, action, lastmarket, newmarket, reward))
    #     print(r)
    #     env.close()
    #     return retarray

    def save(self):
        "save agent: pickle self and use Keras native save model"
        fullname = "%s%s%05d" % (OUTPUT_DIR, self.filename, len(self.results))
        self.predict_model.save("%s_predict.h5" % fullname)
        # can't save / load train model due to custom loss
        pickle.dump(self, open("%s.p" % fullname, "wb"))

    def load(filename, memory=True):
        "load saved agent"
        self = pickle.load(open("%s.p" % filename, "rb"))
        self.predict_model = load_model("%s_predict.h5" % filename)
        print("loaded %d results, %d rows of memory, epsilon %.4f" % (len(self.results),
                                                                      len(self.memory),
                                                                      self.epsilon))




