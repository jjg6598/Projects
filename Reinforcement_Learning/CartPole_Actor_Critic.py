#!/usr/bin/env python
# coding: utf-8

# In[1]:


import collections
import gym
import numpy as np
import statistics
import tensorflow as tf
import tqdm

from matplotlib import pyplot as plt
from tensorflow.keras import layers
from typing import Any, List, Sequence, Tuple


# In[2]:


env = gym.make('CartPole-v0')

seed = 42
env.seed(seed)
tf.random.set_seed(seed)
np.random.seed(seed)

eps = np.finfo(np.float32).eps.item()


# In[4]:


class ActorCritic(tf.keras.Model):
    
    def __init__(self, num_actions: int, num_hidden_units: int):
        super().__init__()
        
        self.common = layers.Dense(num_hidden_units, activation='relu')
        self.actor = layers.Dense(num_actions)
        self.critic = layers.Dense(1)
        
    def call(self, input: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        x = self.common(inputs)
        return self.actor(x), self.critic(x)
                 


# In[5]:


num_actions = env.action_space.n
num_hidden_units = 128

model = ActorCritic(num_actions, num_hidden_units)


# In[6]:


def env_step(action: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    
    state, reward, done, _ = env.step(action)
    return (state.astype(np.float32),
           np.array(reward, np.int32), 
           np.array(done, np.int32))

def tf_env_step(action: tf.Tensor) -> List[tf.Tensor]:
    return tf.numpy_function(env_step, [action], 
                            [tf.float32, tf.int32, tf.int32])


# In[ ]:




