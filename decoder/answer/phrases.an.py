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

for L in range(1, len(sentence) + 1):
    for subset in itertools.combinations(list(enumerate(sentence_orig.split(' '))), L):
        # to ensure we preserve the order of combination, we check if the combination
        # actually appears in the sentence
        indices = [x[0] for x in subset]
        flag = 0
        # first, the subset should be in the sentence
        if u' '.join(tuple([x[1] for x in subset])) in sentence_orig:
            # then we ensure that the chosen subset is actually adjacent.
            for t in range(len(indices) - 1):
                if indices[t + 1] - indices[t] > 1:
                    flag = 1
            if flag == 0:
                try:
                    #                         print(tm[tuple([x[1] for x in subset])])
                    #                         print(subset)
                    sc_P.append((subset, tm[tuple([x[1] for x in subset])]))
                except KeyError:
                    continue

sc_P_indices = []
ind = 0  # index is at zero place
for inst in range(len(sc_P)):
    indices = []
    for source_phrase in sc_P[inst][0]:  # source phrase is at pos 0
        indices.append(source_phrase[ind])
    start = min(indices)
    end = max(indices)
    sc_P_indices.append((start, end, sc_P[inst][1]))

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

