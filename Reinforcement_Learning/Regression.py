#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

np.set_printoptions(precision=3, suppress=True)


# In[3]:


import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing

print(tf.__version__)


# In[4]:


url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'
column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight',
                'Acceleration', 'Model Year', 'Origin']

raw_dataset = pd.read_csv(url, names=column_names,
                          na_values='?', comment='\t',
                          sep=' ', skipinitialspace=True)


# In[5]:


dataset = raw_dataset.copy()
dataset.tail()


# In[6]:


dataset.isna().sum()


# In[7]:


dataset = dataset.dropna()


# In[8]:


dataset['Origin'] = dataset['Origin'].map({1: 'USA', 2: 'Europe', 3: 'Japan'})


# In[9]:


dataset = pd.get_dummies(dataset, columns=['Origin'], prefix='', prefix_sep='')
dataset.tail()


# In[10]:


train_dataset = dataset.sample(frac=0.8, random_state=0)
test_dataset = dataset.drop(train_dataset.index)


# In[11]:


sns.pairplot(train_dataset[['MPG', 'Cylinders', 'Displacement', 'Weight']], diag_kind='kde')


# In[12]:


train_dataset.describe().transpose()


# In[13]:


train_features = train_dataset.copy()
test_features = test_dataset.copy()

train_labels = train_features.pop('MPG')
test_labels = test_features.pop('MPG')


# In[14]:


train_dataset.describe().transpose()[['mean', 'std']]


# In[17]:


normalizer = preprocessing.Normalization(axis=1)

normalizer.adapt(np.array(train_features))

print(normalizer.mean.numpy())


# In[20]:


first = np.array(train_features[:1])

with np.printoptions(precision=2, suppress=True):
    print('First Example:', first)
    print()
    print('Normalized:', normalizer(first).numpy())


# In[28]:


horsepower = np.array(train_features['Horsepower'])

horsepower_normalizer = preprocessing.Normalization(input_shape=[1,],
                                                   axis=None)

horsepower_normalizer.adapt(horsepower)


# In[29]:


horsepower_model = tf.keras.Sequential([
    horsepower_normalizer,
    layers.Dense(1)
])

horsepower_model.summary()


# In[30]:


horsepower_model.predict(horsepower[:10])


# In[32]:


horsepower_model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.1),
                        loss='mean_absolute_error')


# In[33]:


get_ipython().run_cell_magic('time', '', "\nhistory = horsepower_model.fit(\n    train_features['Horsepower'], train_labels,\n    epochs = 100,\n    verbose = 0,\n    validation_split = 0.2)")


# In[34]:


hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
hist.tail()


# In[35]:


def plot_loss(history):
  plt.plot(history.history['loss'], label='loss')
  plt.plot(history.history['val_loss'], label='val_loss')
  plt.ylim([0, 10])
  plt.xlabel('Epoch')
  plt.ylabel('Error [MPG]')
  plt.legend()
  plt.grid(True)


# In[36]:


plot_loss(history)


# In[37]:


test_results = {}

test_results['horsepower_model'] = horsepower_model.evaluate(
    test_features['Horsepower'],
    test_labels, verbose=0)


# In[39]:


x = tf.linspace(0.0, 250, 251)
y = horsepower_model.predict(x)

def plot_horsepower(x, y):
  plt.scatter(train_features['Horsepower'], train_labels, label='Data')
  plt.plot(x, y, color='k', label='Predictions')
  plt.xlabel('Horsepower')
  plt.ylabel('MPG')
  plt.legend()


# In[40]:


plot_horsepower(x,y)


# In[41]:


linear_model = tf.keras.Sequential([
    normalizer,
    layers.Dense(1)
])

linear_model.predict(train_features[:10])


# In[42]:


linear_model.layers[1].kernel


# In[46]:


linear_model.compile(
    optimizer=tf.optimizers.Adam(learning_rate=0.1),
    loss='mean_absolute_error')


# In[47]:


get_ipython().run_cell_magic('time', '', 'history = linear_model.fit(\n    train_features, train_labels, \n    epochs=100,\n    # suppress logging\n    verbose=0,\n    # Calculate validation results on 20% of the training data\n    validation_split = 0.2)')


# In[48]:


plot_loss(history)


# In[49]:


test_results['linear_model'] = linear_model.evaluate(
    test_features, test_labels, verbose=0)


# In[50]:


def build_and_compile_model(norm):
    
    model = keras.Sequential([
        norm, 
        layers.Dense(64, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(1)
    ])
    
    model.compile(loss='mean_absolute_error',
                 optimizer=tf.keras.optimizers.Adam(0.001))
    
    return model


# In[51]:


dnn_horsepower_model = build_and_compile_model(horsepower_normalizer)

dnn_horsepower_model.summary()


# In[52]:


get_ipython().run_cell_magic('time', '', "history = dnn_horsepower_model.fit(\n    train_features['Horsepower'], train_labels,\n    validation_split=0.2,\n    verbose=0, epochs=100)")


# In[53]:


plot_loss(history)


# In[55]:


x = tf.linspace(0.0, 250, 251)
y = dnn_horsepower_model.predict(x)

plot_horsepower(x, y)


# In[57]:


test_results['dnn_horsepower_model'] = dnn_horsepower_model.evaluate(
    test_features['Horsepower'], test_labels,
    verbose=0)


# In[58]:


dnn_model = build_and_compile_model(normalizer)
dnn_model.summary()


# In[59]:


get_ipython().run_cell_magic('time', '', 'history = dnn_model.fit(\n    train_features, train_labels, \n    validation_split = 0.2,\n    verbose=0, epochs=100)')


# In[60]:


plot_loss(history)


# In[61]:


test_results['dnn_model'] = dnn_model.evaluate(
    test_features, test_labels, verbose=0)


# In[62]:


pd.DataFrame(test_results, index=['Mean absolute error [MPG]']).T


# In[63]:


test_predictions = dnn_model.predict(test_features).flatten()

a = plt.axes(aspect='equal')
plt.scatter(test_labels, test_predictions)
plt.xlabel('True Values [MPG]')
plt.ylabel('Predictions [MPG]')
lims = [0, 50]
plt.xlim(lims)
plt.ylim(lims)
_ = plt.plot(lims, lims)


# In[64]:


error = test_predictions - test_labels
plt.hist(error, bins=25)
plt.xlabel('Prediction Error [MPG]')
_ = plt.ylabel('Count')


# In[65]:


dnn_model.save('dnn_model')


# In[66]:


reloaded = tf.keras.models.load_model('dnn_model')

test_results['reloaded'] = reloaded.evaluate(
    test_features, test_labels, verbose=0)


# In[67]:


pd.DataFrame(test_results, index=['Mean absolute error [MPG]']).T


# In[ ]:












