
# coding: utf-8

# In[2]:


from random import random
from numpy import array
from numpy import cumsum

# create a sequence classification instance
def get_sequence(n_timesteps):
	# create a sequence of random numbers in [0,1]
	X = array([random() for _ in range(n_timesteps)])
	# calculate cut-off value to change class values
	limit = n_timesteps/4.0
	# determine the class outcome for each item in cumulative sequence
	y = array([0 if x < limit else 1 for x in cumsum(X)])
	return X, y


# In[13]:


n_timesteps = 10


# In[ ]:


X,y = get_sequence(n_timesteps)


# In[35]:


X = [[0,1,0,1,2], [1,2,4,0,0,1,2], [0,1,2]]
y = [[0,1],[0,0],[0,1],[0,1],[0,0]], [[0,1],[1,1],[0,0],[1,1],[0,0],[1,0],[0,1]], [[0,1],[1,1],[0,0]]


# In[38]:


from keras.preprocessing import sequence
x = sequence.pad_sequences(X, value=-1)
y = sequence.pad_sequences(y, value=-1)


# In[39]:


zip(x,y)


# In[15]:


import keras
from keras.utils.np_utils import to_categorical
y = to_categorical(y)


# In[16]:


# reshape input and output data to be suitable for LSTMs
X = X.reshape(1, n_timesteps, 1)
y = y.reshape(1, n_timesteps, 2)


# In[18]:


from keras.models import Sequential
from keras.layers import LSTM, TimeDistributed, Dense
# define LSTM
model = Sequential()
model.add(LSTM(20, input_shape=(10, 1), return_sequences=True))
model.add(TimeDistributed(Dense(2, activation='softmax')))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])


# In[49]:


from keras.models import Model
from keras.layers import Input, Masking

# reshape input and output data to be suitable for LSTMs
x = x.reshape(3, 7, 1)
y = y.reshape(3, 7, 2)

inp = Input(shape=(7,1))
mask = Masking(mask_value=-1)(inp)
out = TimeDistributed(Dense(1,activation='linear'))(mask)
model = Model(input=inp,output=out)
q = model.predict(x)
print (q[0])


# In[19]:


model.summary()


# In[20]:


# train LSTM
for epoch in range(1000):
	# generate new random sequence
	# fit model for one epoch on this sequence
	model.fit(X, y, epochs=1, batch_size=1, verbose=2)

