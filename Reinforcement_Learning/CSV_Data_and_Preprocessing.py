#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import pydot
import graphviz

np.set_printoptions(precision=3, suppress=True)

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing


# In[2]:


abalone_train = pd.read_csv(
    "https://storage.googleapis.com/download.tensorflow.org/data/abalone_train.csv",
    names=["Length", "Diameter", "Height", "Whole weight", "Shucked weight",
           "Viscera weight", "Shell weight", "Age"])

abalone_train.head()


# In[3]:


abalone_train.shape


# In[4]:


abalone_features = abalone_train.copy()
abalone_labels = abalone_features.pop('Age')


# In[5]:


abalone_features = np.array(abalone_features)
abalone_features


# In[6]:


abalone_model = tf.keras.Sequential([
    layers.Dense(64),
    layers.Dense(1)
])

abalone_model.compile(loss = tf.losses.MeanSquaredError(),
                     optimizer = tf.optimizers.Adam())


# In[7]:


abalone_model.fit(abalone_features, abalone_labels, epochs=10)


# In[8]:


normalize = preprocessing.Normalization()


# In[9]:


normalize.adapt(abalone_features)


# In[10]:


norm_abalone_model = tf.keras.Sequential([
    normalize, 
    layers.Dense(64),
    layers.Dense(1)
])

norm_abalone_model.compile(loss = tf.losses.MeanSquaredError(), 
                          optimizer = tf.optimizers.Adam())

norm_abalone_model.fit(abalone_features, abalone_labels, epochs=10)


# In[ ]:







# In[11]:


titanic = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/train.csv')
titanic.head()


# In[12]:


titanic_features = titanic.copy()
titanic_labels = titanic_features.pop('survived')


# In[13]:


input = tf.keras.Input(shape=(), dtype=tf.float32)

result = 2*input + 1

result


# In[14]:


calc = tf.keras.Model(inputs=input, outputs=result)


# In[15]:


print(calc(1).numpy())
print(calc(2).numpy())


# In[16]:


inputs = {}

for name, column in titanic_features.items():
  dtype = column.dtype
  if dtype == object:
    dtype = tf.string
  else:
    dtype = tf.float32

  inputs[name] = tf.keras.Input(shape=(1,), name=name, dtype=dtype)

inputs


# In[17]:


numeric_inputs = {name:input for name,input in inputs.items()
                 if input.dtype == tf.float32}

x = layers.Concatenate()(list(numeric_inputs.values()))
norm = preprocessing.Normalization()
norm.adapt(np.array(titanic[numeric_inputs.keys()]))
all_numeric_inputs = norm(x)

all_numeric_inputs


# In[18]:


preprocessed_inputs = [all_numeric_inputs]


# In[19]:


for name, input in inputs.items():
    if input.dtype == tf.float32:
        continue
        
    lookup = preprocessing.StringLookup(vocabulary = np.unique(titanic_features[name]))
    onehot = preprocessing.CategoryEncoding(num_tokens=lookup.vocabulary_size())
    
    x = lookup(input)
    x = onehot(x)
    preprocessed_inputs.append(x)


# In[20]:


preprocessed_inputs_cat = layers.Concatenate()(preprocessed_inputs)

titanic_preprocessing = tf.keras.Model(inputs, preprocessed_inputs_cat)

tf.keras.utils.plot_model(model = titanic_preprocessing , rankdir="LR", dpi=72, show_shapes=True)


# In[21]:


titanic_features_dict = {name: np.array(value) 
                         for name, value in titanic_features.items()}


# In[22]:


features_dict = {name:values[:1] for name, values in titanic_features_dict.items()}
titanic_preprocessing(features_dict)


# In[23]:


def titanic_model(preprocessing_head, inputs):
  body = tf.keras.Sequential([
    layers.Dense(64),
    layers.Dense(1)
  ])

  preprocessed_inputs = preprocessing_head(inputs)
  result = body(preprocessed_inputs)
  model = tf.keras.Model(inputs, result)

  model.compile(loss=tf.losses.BinaryCrossentropy(from_logits=True),
                optimizer=tf.optimizers.Adam())
  return model

titanic_model = titanic_model(titanic_preprocessing, inputs)


# In[24]:


titanic_model.fit(x=titanic_features_dict, y=titanic_labels, epochs=10)

