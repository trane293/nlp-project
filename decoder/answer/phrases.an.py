# -*- coding: utf-8 -*-
import models
import itertools

# 4 seems to be the general idea. Could be different. Need to find out
d = distortion_limit = 4

# Start working on a single sentence generalize later
sentence_orig = u"honorables sénateurs , que se est - il passé ici , mardi dernier ?"
sentence = sentence_orig.split(' ')

# Get a translation for each set of words using tm
tm = models.TM("../data/tm", 1)

# We need to find all combinations of the input sentence, and then check if a combination exists in the
# translation model. Earlier implementation had a search scheme going on, which i) did not check for all
# combinations, ii) was highly inefficient since it was searching through the dictionary, which is an O(n)
# operation. Indexing a dictionary is O(1).

# define script P, set of all possible phrases in a sentence
sc_P = []

for i in range(0, len(sentence)):
    for j in range(i, len(sentence)):
        if i == j:
            subset = (sentence[i],)
        else:
            subset = tuple(sentence[i:j+1])
        try:
            tm_output = tm[subset]
            sc_P.append((i, j, tm_output))
        except KeyError:
            continue

# found_indices = []
# phrases = []
# for key, value in translation_model.iteritems():
#     indices = [sentence.index(word) for word in key if word in sentence]
#
#     if len(key) == len(indices):
#         pass
#     else:
#         continue
#
#     prev_ix = indices[0]
#     for ix in indices[1:]:
#         if ix - prev_ix != 1:
#             break
#         if ix == indices[-1]:
#             print indices
#             phrases.append(value)
#             found_indices.append(indices)
#
#         prev_ix = ix
#
#
# # List of english phrases found for this sentence
# print found_indices
# print

