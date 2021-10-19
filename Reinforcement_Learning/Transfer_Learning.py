#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os 
import numpy as np

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds

print("Version: ", tf.__version__)
print("Eager mode: ", tf.executing_eagerly())
print("Hub version: ", hub.__version__)
print("GPU is", "available" if tf.config.list_physical_devices("GPU") else "NOT AVAILABLE")


# In[2]:


train_data, validation_data, test_data = tfds.load(
    name = 'imdb_reviews',
    split = ('train[:60%]', 'train[60%:]', 'test'),
    as_supervised=True)


# In[3]:


train_examples_batch, train_labels_batch = next(iter(train_data.batch(10)))
train_examples_batch


# In[4]:


train_labels_batch


# In[6]:


embedding = 'https://tfhub.dev/google/nnlm-en-dim50/2'
hub_layer = hub.KerasLayer(embedding, input_shape=[],
                          dtype=tf.string, trainable=True)
hub_layer(train_examples_batch[:3])


# In[8]:


model = tf.keras.Sequential([
    hub_layer,
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1)
])

model.summary()


# In[9]:


model.compile(optimizer='adam',
             loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
             metrics=['accuracy'])


# In[10]:


history = model.fit(train_data.shuffle(10000).batch(512),
                    epochs = 10,
                    validation_data = validation_data.batch(512),
                    verbose = 1)


# In[11]:


results = model.evaluate(test_data.batch(512), verbose = 2)

for name, value in zip(model.metrics_names, results):
    print('%s: %.3f' % (name, value))

