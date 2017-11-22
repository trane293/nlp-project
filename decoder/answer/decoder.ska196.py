# -*- coding: utf-8 -*-
import optparse
import sys
import models
from collections import namedtuple
import imp
models = imp.load_compiled("models","models/models.pyc")

# 4 seems to be the general idea. Could be different. Need to find out
d = distortion_limit = 4


# Start working on a single sentence generalize later
sentence = "honorables sénateurs , que se est - il passé ici , mardi dernier ?"
sentence = sentence.split(' ')


# Get a translation for each set of words using tm
translation_model = models.TM("../data/tm", 10)

# (',', 'et', 'je', 'y') [phrase(english=', and I will', logprob=0.0)]
# translation_model type -- dict {(tuple of french words),list of [(namedtuple --  phrase, englishword logprob))]

# tm[('que', 'se', 'est')] = [
#   phrase(english='what has', logprob=-0.301030009985),
#   phrase(english='what has been', logprob=-0.301030009985)]

# print translation_model

found_indices = []
phrases = []
for key, value in translation_model.iteritems():
    indices = [sentence.index(word) for word in key if word in sentence]

    if len(key) == len(indices):
        pass
    else:
        continue

    prev_ix = indices[0]
    for ix in indices[1:]:
        if ix - prev_ix != 1:
            break
        if ix == indices[-1]:
            print indices
            phrases.append(value)
            found_indices.append(indices)

        prev_ix = ix


# List of english phrases found for this sentence
print found_indices
print 

