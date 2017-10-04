
# coding: utf-8

# # <span style='color:orange'> Chinese Word Segmentation using Bidirectional LSTMs </span>

# This problem definition of chinese word segmentation closely mimics unigram segmentation method, where the label of the next word is predicted only by looking at the current word, without any context. However the non-existence of context cannot be accurately stated since the network learns the internal representation of the characters as well as their dependencies to give a more informed prediciton compared to simple unigram model. 

# # <span style='color:green'> Building the Dataset for Training <span>

# In[ ]:


from __future__ import print_function
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional
from keras.utils.np_utils import to_categorical
import collections


# # Use the Wseg_1M Dataset

# #### Write a wrapper for the below function to incorporate the 1M chinese word segm dataset

# In[ ]:


# we basically parse the input file and find all spaces. As soon as we find a space we output the word into a text
# file called word_file_1M which has the same format as the count_1w file.

with(open('/local-scratch/wseg_simplified_cn.txt', 'rb')) as f:
    word_file_1M = open('/local-scratch/word_file_1M', 'wb')
    for line in f:
        line = unicode(line, 'utf-8')
        line = line.replace('\n', '')
        line = line.split(' ')
        for word in line:
            word_file_1M.write(word.encode('utf-8') + '\t'.encode('utf-8') + str(0).encode('utf-8') +                                       '\n'.encode('utf-8'))
    f.close()
word_file_1M.close()


# # Assigning labels to characters

# Labels:
# 
# 0 - Beginning <br>
# 1 - Middle <br>
# 2 - End <br>
# 3 - Single Character word

# In[ ]:


# we open the count file, get the word and assign labels to each character
# cannot use dictionaries, since the same character may appear again and overwrites the value at its place in dict.
with(open('/local-scratch/word_file_1M', 'rb')) as f:
    label = []
    for line in f:
        word, count = line.split('\t')
        
        # making sure the parsing is going fine
        assert int(count) == 0
        
        word = unicode(word, 'utf-8')
        if len(word) == 1:
            label.append((word[0], 3))
        else:
            for i, character in enumerate(word):
                if i == 0: # this is the first letter
                    label.append((character, 0))
                elif i == (len(word) - 1): # this is the last letter
                    label.append((character, 2))
                else: # this is somewhere in the middle
                    label.append((character, 1))
                    
    f.close()


# # Building initial integer embeddings for characters

# In[ ]:


all_words = [y[0] for y in label]


# In[ ]:


def build_dataset(words):
    count = collections.Counter(words).most_common()
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return dictionary, reverse_dictionary


# ### Testing

# In[ ]:


orig_dict, ret_dict = build_dataset(all_words)


# In[ ]:


ret_dict[orig_dict[u'\u6753']]


# In[ ]:


len(orig_dict)


# In[ ]:


x_train = [orig_dict[y[0]] for y in label]
y_train = [y[1] for y in label]


# ### Visualize training data

# In[ ]:


zip(x_train, y_train)


# Change y_train into categorical one-hot-vector encoding

# In[ ]:


y_train = to_categorical(y_train)


# In[ ]:


print(np.shape(x_train), 'train sequence shape')
print(np.shape(y_train), 'labels shape')


# # Build and Train a BLSTM

# In[ ]:


model = Sequential()
# first argument is the size of the vocabulary, second argument is the size of embedding, third argument is the
# number of features in the text, we only have 1 character. 
model.add(Embedding(len(orig_dict), 200, input_length=1))
model.add(Bidirectional(LSTM(10, return_sequences=True)))
model.add(Bidirectional(LSTM(10)))
model.add(Dropout(0.2))
model.add(Dense(4, activation='softmax'))

model.compile('adadelta', 'categorical_crossentropy', metrics=['accuracy'])

print('Train...')
model.fit(np.array(x_train), y_train,
          batch_size=20,
          epochs=1,
          validation_data=[x_train, y_train])


# # Testing on test set and generating output file suitable for scoring

# In[ ]:


# open the input file, open the output file
inp_file = open('../data/input', 'rb')
out_file = open('../data/output_rnn', 'wb')

# read the first line

for line in inp_file:
    # divide the line into characters
    line = unicode(line, 'utf-8')
    line = line.replace('\n','')
    print(line)
    init_embedding = [orig_dict[char] for char in line]
        
    # change the character into initial embedding using orig_dict
    
    # feed into RNN
        
    pred_labels = model.predict_classes(init_embedding)
    
    # get the class label
    
    for num, char in enumerate(line):
        if pred_labels[num] >= 2:
            out_file.write(char.encode('utf-8') + ' '.encode('utf-8'))
        else:
            out_file.write(char.encode('utf-8'))
    
    out_file.write('\n'.encode('utf-8'))
    
out_file.close()
    # if label == 2 or label == 3, add a trailing space after the character

    # output the line into a file
    # f = open('myfile', 'w')
    # f.write('hi there\n')  # python will convert \n to os.linesep



# In[ ]:


pred_labels

