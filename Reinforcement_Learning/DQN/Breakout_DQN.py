#!/usr/bin/env python
# coding: utf-8

# In[2]:


import gym


# In[3]:


env = gym.make('BreakoutDeterministic-v4')

frame = env.reset()

env.render()

is_done = False
while not is_done:
    frame, reward, is_done, _ = env.step(env.action_space.sample())
    
    env.render()


# In[4]:


def to_grayscale(img):
    return np.mean(img, axis=2).astype(np.uint8)

def downsample(img):
    return img[::2, ::2]

def preprocess(img):
    return to_grayscale(downsample(img))


# In[5]:


def transform_reward(reward):
    return np.sign(reward)


# In[7]:


def fit_batch(model, gamme, start_states, actions, rewards, next_states,
             is_terminal):
    
    next_Q_values = model.predict([next_states, np.ones(actions.shape)])
    
    next_Q_values[is_terminal] = 0
    
    Q_values = rewards + gamma * np.max(next_Q_values, axis=1)
    
    model.fit(
        [start_states, actions], actions * Q_values[:, None],
        nb_epoch=1, batch_size=len(start_states), verbose=0
    )


# In[8]:


def atari_model(n_actions):
    
    ATARI_SHAPE = (4, 105, 80)
    
    frames_input = keras.layers.Input(ATARI_SHAPE, name='frames')
    actions_input = keras.layers.Input((n_actions,), name='mask')
    
    normalized = keras.layers.Lambda(lambda x: x / 255.0)(frames_input)
    
    conv_1 = keras.layers.convolutional.Convolution2D(
        16, 8, 8, subsample = (4,4), activation = 'relu'
    )(normalized)
    
    conv_2 = keras.layers.convolutional.Convolution2D(
        16, 8, 8, subsample = (4,4), activation = 'relu'
    )(conv_1)
    
    conv_flattened = keras.layers.core.Flatten()(conv_2)
    
    hidden = keras.layers.Dense(256, activation = 'relu')(conv_flattened)
    
    filtered_output = keras.layers.merge([output, actions_input], mode = 'mul')
    
    self.model = keras.models.Model(input=[frames_input, actions_input], 
                                    output = filtered_output)
    optimizer = keras.optimizers.RMSprop(lr=0.00025, rho=0.95, epsilon=0.01)
    self.model.compile(optimizer, loss = 'mse')


# In[9]:


class RingBuf:
    def __init__(self, size):
        self.data = [None] * (size+1)
        self.start = 0
        self.end = 0
        
    def append(self, element):
        self.data[self.end] = element
        self.end = (self.end + 1) % len(self.data)
        
        if self.end == self.start:
            self.start = (self.start + 1) % len(self.data)
            
    def __getitem__(self):
        if self.end < self.start:
            return self.end + len(self.data) - self.start
        
        else:
            return self.end - self.start
        
    def __iter__(self):
        for i in range(len(self)):
            yield self[i]


# In[ ]:




