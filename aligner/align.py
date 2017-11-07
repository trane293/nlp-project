
# coding: utf-8

# # IBM Model 1 with Expectation Maximization

# ## Author: Anmol Sharma

# **Open the files**

# In[1]:


from __future__ import print_function
import itertools, sys
import numpy as np


# In[2]:


print('Opening files..', file=sys.stderr)
eng_set = open('./data/hansards.en', 'rb')
fr_set = open('./data/hansards.fr', 'rb')


# **Perform Preprocessing**

# 1. **Split the data into two different sets, and split each sentence into words**
# 2. **Add a NONE character inside every english sentence**

# In[3]:


src_sent = []
dest_sent = []

for line_en, line_fr in zip(eng_set, fr_set):
    # split each sentence into a list of words for easy processing
    src_sent.append(line_fr.split())
    dest_sent.append(line_en.split())


# We can see the words contain many "\xc3\xa9es"... which are basically the unicode codes for special accented symbols in french. Nothing to worry.
#
# Also, the punctuation marks are left as it is as "words" which map directly to the punctuation in the destination language.

# In[4]:


print('French sentences..', file=sys.stderr)
print(src_sent[5:10], file=sys.stderr)


# In[5]:


print('English sentences..', file=sys.stderr)
print(dest_sent[5:10], file=sys.stderr)


# ## **We need to find the probability $t_k(f_i|e_j)$ where $f_i$ = french word (source language) and $e_j$ = english word (destination language)**

# Find all the unique words in french data

# In[6]:


# convert the source list into a chain of iterables, and then convert it to a set to only retain unique elements.
# further convert to list for easy processing
fr_vocab = list(set(itertools.chain.from_iterable(src_sent)))
en_vocab = list(set(itertools.chain.from_iterable(dest_sent)))


# In[7]:


print('Some unique french words..', file=sys.stderr)
print(fr_vocab[0:5], file=sys.stderr)


# In[8]:


print('Some unique english words..', file=sys.stderr)
print(en_vocab[0:5], file=sys.stderr)


# # Start the training process..

# **We cannot initialize the $t_k$ values to uniform due to memory constraints. A better way to do this is to first check if the key exists or not, and if it doesn't, then initialize it to uniform probability. This saves a huge memory and computational overhead of permuting through all $f_i$ and $e_j$ and setting them uniform, many of which will not even appear in the training text**

# In[9]:


k = 0
t_k = {}
count_comb = {}
count_e = {}
uni_prob = 1.0 / np.shape(fr_vocab)[0]
epochs = 1

for _i in range(epochs):
    print('Currently on training epoch {}..'.format(_i+1), file=sys.stderr)
    # iterate over all training examples
    for src_sent_eg, dest_sent_eg in zip(src_sent, dest_sent):
        for f_i in src_sent_eg:
            Z = 0.0
            for e_j in dest_sent_eg:

                if (f_i, e_j) not in t_k:
#                     print('({}, {}) not in t_k, initializing to uniform!'.format(f_i, e_j))
                    t_k[(f_i, e_j)] = 1.0 / uni_prob

                Z += t_k[(f_i, e_j)]
            for e_j in dest_sent_eg:
                c = t_k[(f_i, e_j)] / Z

                # initialize counts on the fly
                if (f_i, e_j) not in count_comb:
#                     print('({}, {}) not in count_comb, initializing to 0!'.format(f_i, e_j))
                    count_comb[(f_i, e_j)] = 0

                # initialize counts on the fly
                if e_j not in count_e:
#                     print('({}) not in count_e, initializing to 0!'.format(e_j))
                    count_e[e_j] = 0

                count_comb[(f_i, e_j)] += c
                count_e[e_j] += c
    print('Updating t_k counts...', file=sys.stderr)
    for f_e_keys in count_comb:
        # f_e_keys[0] = f_i, f_e_keys[1] = e_j
        t_k[(f_e_keys[0], f_e_keys[1])] = count_comb[f_e_keys] / count_e[f_e_keys[1]]


# # Make predictions using this trained model..

# In[11]:


print('Aligning...', file=sys.stderr)
print('Source | Destination', file=sys.stderr)
for src_sent_eg, dest_sent_eg in zip(src_sent, dest_sent):
    i = 0
    for f_i in src_sent_eg:
        bestp = 0
        bestj = 0
        j = 0
        for e_j in dest_sent_eg:
            if t_k[(f_i, e_j)] > bestp:
                bestp = t_k[(f_i, e_j)]
                bestj = e_j
                j += 1
        sys.stdout.write('{}-{} '.format(i,j))
        i += 1
    sys.stdout.write('\n')
