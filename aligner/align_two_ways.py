
# coding: utf-8

# # IBM Model 1 with Expectation Maximization

# ## Author: Anmol Sharma

# **Open the files**

# In[1]:


from __future__ import print_function
import itertools, sys, optparse, os
import numpy as np


# In[3]:

optparser = optparse.OptionParser()
optparser.add_option("-d", "--datadir", dest="datadir", default="data", help="data directory (default=data)")
optparser.add_option("-p", "--prefix", dest="fileprefix", default="europarl", help="prefix of parallel data files (default=europarl)")
optparser.add_option("-e", "--english", dest="english", default="en", help="suffix of Destination (target language) filename (default=en)")
optparser.add_option("-f", "--french", dest="french", default="de", help="suffix of Source (source language) filename (default=de)")
optparser.add_option("-l", "--logfile", dest="logfile", default=None, help="filename for logging output")
optparser.add_option("-t", "--threshold", dest="threshold", default=0.5, type="float", help="threshold for alignment (default=0.5)")
optparser.add_option("-i", "--epochs", dest="epochs", default=5, type="int", help="Number of epochs to train for (default=5)")
optparser.add_option("-n", "--num_sentences", dest="num_sents", default=100000, type="int", help="Number of sentences to use for training and alignment")
(opts, _) = optparser.parse_args()

f_data = "%s.%s" % (os.path.join(opts.datadir, opts.fileprefix), opts.french)
e_data = "%s.%s" % (os.path.join(opts.datadir, opts.fileprefix), opts.english)


print('Opening files..', file=sys.stderr)
src_set = open(f_data)
des_set = open(e_data)


# **Perform Preprocessing**

# 1. **Split the data into two different sets, and split each sentence into words**
# 2. **Add a NONE character inside every english sentence**

# In[4]:


src_sent = []
dest_sent = []

for line_des, line_src in zip(des_set, src_set):
    # split each sentence into a list of words for easy processing
    src_sent.append(line_src.split() + ['nOnE'])
    # src_sent.append(line_src.split())
    # print(src_sent[0:10])
    dest_sent.append(line_des.split() + ['nOnE'])


# We can see the words contain many "\xc3\xa9es"... which are basically the unicode codes for special accented symbols in french. Nothing to worry. 
# 
# Also, the punctuation marks are left as it is as "words" which map directly to the punctuation in the destination language. 

# In[5]:


print('Source sentences..', file=sys.stderr)
print(src_sent[5:10], file=sys.stderr)


# In[6]:


print('Destination sentences..', file=sys.stderr)
print(dest_sent[5:10], file=sys.stderr)


# ## **We need to find the probability $t_k(f_i|e_j)$ where $f_i$ = source word and $e_j$ = destination word**

# Find all the unique words in french data

# In[7]:


# convert the source list into a chain of iterables, and then convert it to a set to only retain unique elements.
# further convert to list for easy processing
src_vocab = list(set(itertools.chain.from_iterable(src_sent)))
des_vocab = list(set(itertools.chain.from_iterable(dest_sent)))


# In[8]:


print('Some unique source words..', file=sys.stderr)
print(src_vocab[0:5], file=sys.stderr)


# In[9]:


print('Some unique destination words..', file=sys.stderr)
print(des_vocab[0:5], file=sys.stderr)


# # Start the training process..

# **We cannot initialize the $t_k$ values to uniform due to memory constraints. A better way to do this is to first check if the key exists or not, and if it doesn't, then initialize it to uniform probability. This saves a huge memory and computational overhead of permuting through all $f_i$ and $e_j$ and setting them uniform, many of which will not even appear in the training text**

# In[ ]:


k = 0
t_k = {}
t_k_2 ={}
uni_prob = 1.0 / np.shape(src_vocab)[0]
uni_prob_2 = 1.0 / np.shape(des_vocab)[0]
epochs = opts.epochs
print('Starting training for {} epochs..'.format(epochs), file=sys.stderr)
for _i in range(epochs):
    print('Currently on training epoch {}..'.format(_i+1), file=sys.stderr)
    count_comb = {}
    count_comb_2 = {}
    count_e_2 = {}
    count_e = {}
    # iterate over all training examples
    for src_sent_eg, dest_sent_eg in zip(src_sent, dest_sent):
        for f_i in src_sent_eg:
            Z = 0.0
            for e_j in dest_sent_eg:

                 # initialize counts on the fl
                if (f_i, e_j) not in t_k:
#                     print('({}, {}) not in t_k, initializing to uniform!'.format(f_i, e_j))
                    t_k[(f_i, e_j)] = uni_prob

                Z += t_k[(f_i, e_j)]
            for e_j in dest_sent_eg:
                c = t_k[(f_i, e_j)] / Z

                # initialize counts on the fly
                if (f_i, e_j) not in count_comb:
#                     print('({}, {}) not in count_comb, initializing to 0!'.format(f_i, e_j))
                    count_comb[(f_i, e_j)] = 0.0

                # initialize counts on the fly
                if e_j not in count_e:
#                     print('({}) not in count_e, initializing to 0!'.format(e_j))
                    count_e[e_j] = 0.0

                count_comb[(f_i, e_j)] += c
                count_e[e_j] += c


    print('Updating t_k counts...', file=sys.stderr)
    for f_e_keys in count_comb:
        # f_e_keys[0] = f_i, f_e_keys[1] = e_j
        lmda = 0.01
        t_k_2[(f_e_keys[0], f_e_keys[1])] = \
            lmda * (count_comb['nOnE', f_e_keys[1]] / count_e[f_e_keys[1]]) \
            + (1 - lmda) * count_comb[f_e_keys] / count_e[f_e_keys[1]]
        # count_comb[f_e_keys] / count_e[f_e_keys[1]]

    # =============+ROUND2+=========================
    # for src_sent_eg, dest_sent_eg in zip(src_sent, dest_sent):
    for src_sent_eg, dest_sent_eg in zip(dest_sent, src_sent):
        for f_i in src_sent_eg:
            Z_2 = 0.0
            for e_j in dest_sent_eg:
                # initialize counts on the fl
                if (f_i, e_j) not in t_k_2:
                    # print('({}, {}) not in t_k, initializing to uniform!'.format(f_i, e_j))
                    t_k_2[(f_i, e_j)] = uni_prob_2

                Z_2 += t_k_2[(f_i, e_j)]
            for e_j in dest_sent_eg:
                c_2 = t_k_2[(f_i, e_j)] / Z_2

                # initialize counts on the fly
                if (f_i, e_j) not in count_comb_2:
                    # print('({}, {}) not in count_comb, initializing to 0!'.format(f_i, e_j))
                    count_comb_2[(f_i, e_j)] = 0.0

                # initialize counts on the fly
                if e_j not in count_e_2:
                    # print('({}) not in count_e, initializing to 0!'.format(e_j))
                    count_e_2[e_j] = 0.0

                count_comb_2[(f_i, e_j)] += c_2
                count_e_2[e_j] += c_2

    print('Updating t_k counts...', file=sys.stderr)
    for f_e_keys in count_comb_2:
        # f_e_keys[0] = f_i, f_e_keys[1] = e_j
        lmda=0.01
        t_k_2[(f_e_keys[0], f_e_keys[1])] = \
            lmda * (count_comb_2['nOnE', f_e_keys[1]] / count_e_2[f_e_keys[1]]) \
            + (1-lmda) * count_comb_2[f_e_keys] / count_e_2[f_e_keys[1]]
            # count_comb[f_e_keys] / count_e[f_e_keys[1]]

# # Make predictions using this trained model..

# In[ ]:


print('Aligning...', file=sys.stderr)
print('Source | Destination', file=sys.stderr)
for src_sent_eg, dest_sent_eg in zip(src_sent, dest_sent):
    for i, f_i in enumerate(src_sent_eg):
        if f_i == 'nOnE':
            print(str(i), file=sys.stderr)
            continue
        bestp = 0
        bestj = 0
        fr_word = ''
        en_word = ''
        for j, e_j in enumerate(dest_sent_eg):
            if t_k[(f_i, e_j)] > bestp:
                bestp = t_k[(f_i, e_j)]
                bestj = j
                fr_word = f_i
                en_word = e_j
    for k, e_k in enumerate(dest_sent_eg):
        bestp2 = 0
        bestl2 = 0
        fr_word2 = ''
        en_word2 = ''
        for l,f_l in enumerate(src_sent_eg):
            if t_k_2[(e_k, f_l)]:
                bestp2 = t_k_2[(e_k, f_l)]
                bestl2 = l
                fr_word2 = f_l
                en_word2 = e_k
    if fr_word2 == fr_word & en_word == en_word2:
        if fr_word != 'nOnE' & en_word != 'nOnE':
            sys.stdout.write('{}-{} '.format(i, bestj))
    sys.stdout.write('\n')