# -*- coding: utf-8 -*-
import optparse
import sys
import models
from operator import itemgetter
from collections import namedtuple
from collections import defaultdict
import numpy as np
import imp
models = imp.load_compiled("models","models/models.pyc")

translation_model = models.TM("../data/tm", 10)

#------------------ Function Declarations ------------------------#

''' Get list of all states with score = alpha* - beta 
    Returns a list of states '''


def beam(list_of_states, beta):
    return [state for state in list_of_states if state.score >= (max(list_of_states, key=itemgetter(4)).score - beta)]

# TODO Verify this function works. I think it may be wrong somewhere
'''Gets a list of possible states given the current state'''
def phrases(given_state, sentence):
    # Use distortion
    # Make sure every word only has one translation.
    print given_state
    found_indices = []
    list_of_phrases = []

    words = [word if value == 0 else '1' for value, word in zip(given_state.bitstring, sentence)]

    # Used for generating sequences
    prev_index = 0
    for i, word in enumerate(words):
        sequence = words[prev_index:i]
        for key, value in translation_model.iteritems():
            indices = [sequence.index(word) for word in key if word in sequence]

            if len(indices) == 0:
                # if no matching key found in translation_model
                continue

            prev_ix = indices[0]
            # For making sure the sequence is good: Credits to Amirali
            for ix in indices[1:]:
                if ix - prev_ix != 1:
                    # Using 1 also rules out sequences like [7,1] which breaks the sequence of the key
                    break
                if ix == indices[-1]:
                    print indices
                    print key
                    print sentence
                    list_of_phrases.append(value)
                    found_indices.append(indices)

                prev_ix = ix

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



#-------------------- end ----------------------------------------#

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


states = []

bitstring = [0]*len(sentence)
# # e1 e2 is the last two translated english words, bitstring, r is the last index of the english phrase, score
# is the score for the current state

State = namedtuple('State',"e1 e2 bitstring r score")

# # state_dict the capital Q
# # Each entry list holds a set of states
# # {}
state_holders = {}


# # q0 is the initial state -> q0 = (*,*,00000,0,0). Use <s> for start symbol instead
# # q1 = State('<s>', '<s>', bitstring, 0, 2)
# # add q0 to 0 in state holders


q0 = State('<s>', '<s>', bitstring, 0, 0)
state_holders[0] = [q0]

# print beam(state_holders[0], 0)
for state in beam(state_holders[0], 0):
    phrases(state, sentence)
# # for each state q in beam, for each phrase in p in ph(q)
# # get all possible derivates for p




# for iteration in range(len(sentence)):
    # get q from beam of Q



# def

