
# coding: utf-8

# # <span style='color:orange'> Apply Machine Learning models to predict best hypothesis translation </span>

# #### For Final Project
# #### Course Name: CMPT 825 Natural Language Processing

# ## Load the data

# In[1]:


from __future__ import print_function
import numpy as np
import cPickle as pickle
import numpy as np
from sklearn.model_selection import train_test_split


# In[2]:


whole_data = np.array(pickle.load(open('./result_vector_sleek.p')) ).reshape(-1,6)
x = pickle.load(open('./result_vector.p')) 
x[0:2]


# In[3]:


y = []
with open('./data/dev.answers', 'rb') as f:
    for true_val in f.readlines():
        y.append(int(true_val))


# In[4]:


y[0:2]


# # Convert to Numpy arrays and set up for training

# In[5]:


# since we have labels only for the first ~26000 sentences, get only that that data
x = x[0:len(y)]


# In[14]:


x_train, x_test, y_train, y_test = train_test_split(np.array(x).reshape(-1, 6), np.array(y).reshape(-1,1), test_size=0.05)


# In[15]:


x_train.shape


# In[16]:


y_train.shape


# In[17]:


x_test.shape


# In[18]:


y_test.shape


# # Start applying machine learning

# ## Logistic Regression

# In[19]:


from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, RidgeClassifierCV
from sklearn.metrics import accuracy_score, classification_report


# In[20]:


log = LogisticRegression(penalty='l2', 
                   dual=False, tol=0.0001, 
                   C=1.0, fit_intercept=True, 
                   intercept_scaling=1, class_weight=None, 
                   random_state=None, solver='saga', 
                   max_iter=1000, multi_class='multinomial', verbose=0, 
                   warm_start=False, n_jobs=1)


# In[21]:


log.fit(x_train, y_train)
pred = log.predict(x_test)
acc = accuracy_score(y_test, pred, normalize=True, sample_weight=None)
print('accuracy {}'.format(acc))


# In[22]:


log.coef_


# In[23]:


log.classes_


# In[25]:


pred_whole_data = log.predict(whole_data)

with open('./output_ML_logistic', 'wb') as f:
    for num in pred_whole_data:
        print(num, file=f)


# ## Support Vector Machines

# In[26]:


from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC


# In[27]:


x_train.shape


# In[28]:


y_train.shape


# In[29]:


x_train.shape


# In[30]:


clf = SVC()
clf.fit(x_train, y_train)


# In[31]:


pred = clf.predict(x_test)
acc = accuracy_score(y_test, pred, normalize=True, sample_weight=None)
print('accuracy {}'.format(acc))


# In[32]:


pred_whole_data = clf.predict(whole_data)


# In[33]:


with open('./output_ML_SVM_new', 'wb') as f:
    for num in pred_whole_data:
        print(num, file=f)


# ## Neural Network

# In[35]:


import keras
from keras.layers import Dense, Input
from keras.models import Model
from keras.utils.np_utils import to_categorical


# In[36]:


inp = Input(shape=(6,))
x = Dense(200, activation='relu')(inp)
x = Dense(200, activation='relu')(x)
x = Dense(200, activation='relu')(x)
x = Dense(3, activation='softmax')(x)

model = Model(inputs=inp, outputs=x)


# In[37]:


model.summary()


# In[38]:


model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])


# In[39]:


y_train_cat = to_categorical(y_train, num_classes=3)


# In[40]:


model.fit(x_train, y_train_cat, epochs=20, validation_data=[x_test, to_categorical(y_test, num_classes=3)])


# In[42]:


cat_classes = np.argmax(model.predict(whole_data), axis=1)
pred_classes = np.zeros((np.shape(whole_data)[0],1))

for idx, cat in enumerate(cat_classes):
    if cat == 0:
        pred_classes[idx] = 0
    if cat == 1:
        pred_classes[idx] = 1
    if cat == 2:
        pred_classes[idx] = -1

pred_whole_data = pred_classes
with open('./output_ML_NN_new', 'wb') as f:
    for num in pred_whole_data:
        print(int(num), file=f)

