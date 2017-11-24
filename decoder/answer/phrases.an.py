# -*- coding: utf-8 -*-
import models
from collections import namedtuple

# 4 seems to be the general idea. Could be different. Need to find out
d = distortion_limit = 4

# Start working on a single sentence generalize later
sentence_orig = u"honorables sénateurs , que se est - il passé ici , mardi dernier ?"
sentence = sentence_orig.split(' ')

# Get a translation for each set of words using tm
tm = models.TM("../data/tm", 1)

sc_P = []

for i in range(0, len(sentence)):
    for j in range(i, len(sentence)):
        if i == j:
            subset = (sentence[i],)
        else:
            subset = tuple(sentence[i:j + 1])
        try:
            tm_output = tm[subset]
            sc_P.append((i, j, tm_output[0]))
        except KeyError:
            continue

# define bitstring as a list of zeros
bitstring = [0] * len(sentence)

# define a numedtuple which will be the state vector q
State = namedtuple('State', "e1 e2 bitstring r alpha")

# this is a container that will hold all states
state_holder = {}

# initial state
q0 = State('<s>', '<s>', bitstring, 0, 0)

# append initial state to state_holder
state_holder[0] = [q0]

# define bitstring as a list of zeros
bitstring = [0] * len(sentence)

# define a numedtuple which will be the state vector q
State = namedtuple('State', "e1 e2 bitstring r alpha")

# this is a container that will hold all states
state_holder = {}

# initial state
q0 = State('<s>', '<s>', bitstring, 0, 0)

# append initial state to state_holder
state_holder[0] = [q0]

def ph(q, d=4):
    ph_states = []
    for state in sc_P:
        flag = True  # we assume it as a valid state
        s = state[0]
        t = state[1]

        orig_bitstring = q.bitstring

        '''
        Step 1: Ensure bit string is not overlapped
        '''
        if s == t:
            # invalid state
            if orig_bitstring[s] != 0:
                flag = False
        else:
            # individial bits s and t are 0, but we also have to check in between them
            if orig_bitstring[s] == 0 and orig_bitstring[t] == 0:
                for _num in range(s, t + 1):
                    if orig_bitstring[_num] != 0:
                        flag = False
            else:
                flag = False

        '''
        Step 2: Ensure distortion limit is still obeyed
        '''

        r = q.r

        if not (abs(r + 1 - s) <= d):
            flag = False

        if flag == True:
            ph_states.append(state)

    return ph_states
