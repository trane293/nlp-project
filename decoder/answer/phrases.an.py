# coding=utf-8
import models
import itertools
from copy import deepcopy
from collections import namedtuple
import numpy as np

# 4 seems to be the general idea. Could be different. Need to find out
d = distortion_limit = 4

# Start working on a single sentence generalize later
sentence_orig = "honorables sénateurs , que se est - il passé ici , mardi dernier ?"
sentence = sentence_orig.split(' ')

# Get a translation for each set of words using tm
tm = models.TM("../data/tm", 1)
lm = models.LM("../data/lm")

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

# define a numedtuple which will be the state vector q
State = namedtuple('State', "e1 e2 bitstring r alpha")


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
                for _num in range(s + 1, t):
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


def next(q, p, eta=-1):
    s = p[0]
    t = p[1]

    num_translated_words = len(p[2].english.split(' '))

    if num_translated_words >= 2:
        last_word = p[2].english.split(' ')[-1]
        second_last = p[2].english.split(' ')[-2]
    elif num_translated_words == 1:
        last_word = p[2].english.split(' ')[-1]
        second_last = q.e2

    # Calculate new_bitstring

    new_bitstring = deepcopy(q.bitstring)

    for i in range(s, t+1):
        # just a safeguard..check whether earlier these values were zeros or not

        if q.bitstring[i] == 1:
            print('MAJOR ERROR!')
            print('The Ph(q) function has issues if you can read this...')
        new_bitstring[i] = 1

    '''
    Calculate new logprob (alpha value)
    '''

    # CALCULATE LANGUAGE MODEL PROBABILITY

    e1 = q.e1
    e2 = q.e2
    prob = 0
    if len(p[2].english.split(' ')) == 1: # just a single word
#         print('Single english word')
        # first try bigram probability
        word = p[2].english.split(' ')[0]
#         print('\n\tFinding probability of {}, {}, {}'.format(e1, e2, word))
        try:
    #         print('Finding probability of {}, {}, {}'.format(e1, e2, word))
            # the index 1 contains the logprob, 0 index contains the repeated state
            prob += lm.score((e1, e2), word)[1]
        except KeyError:
#             print("\t\tCouldn't find trigram prob..backing off to bigram")
            try:
#                 print('\t\t\tFinding probability of {}, {}'.format(e2, word))
                prob += lm.score((e2, ), word)[1]
            except KeyError:
                prob += lm.score((), word)[1]
    else: # there are multiple words
#         print('Multiple english words')
        for _num in range(0, len(p[2].english.split(' '))):
            word = p[2].english.split(' ')[_num]
#             print('\n\tFinding probability of {}, {}, {}'.format(e1, e2, word))
            try:
                prob += lm.score((e1, e2), word)[1]
            except KeyError:
#                 print("\t\tCouldn't find trigram prob..backing off to bigram")
                try:
#                     print('\t\t\tFinding probability of {}, {}'.format(e2, word))
                    prob += lm.score((e2, ), word)[1]
                except KeyError:
                    prob += lm.score((), word)[1]

            e1 = deepcopy(e2)
            e2 = deepcopy(word)

    # CALCULATE G(x) LOGPROB

    g_x = p[2].logprob

    # CALCULATE DISTORTION VALUE
    dist_val = eta * abs(q.r + 1 - s)

    # UPDATE ALPHA VALUE

    new_alpha = q.alpha + g_x + prob + dist_val

    new_state = State(e1=second_last, e2=last_word, bitstring=new_bitstring, r=p[1], alpha=new_alpha)
    return new_state


def eq(q1, q2):
    if q1.e1 != q2.e1:
        return False
    elif q1.e2 != q2.e2:
        return False
    elif q1.bitstring != q2.bitstring:
        return False
    elif q1.r != q2.r:
        return False
    else:
        return True


def add(Q_main, index, q_new, q, valid_phrase, back_pointer):
    for q_dd in Q_main[index]: # q double dash
        if eq(q_new, q_dd) == True:
            # print('Found similar q..')
            if q_new.alpha > q_dd.alpha: # score of new thing is greater than older
                # print('Changing the older q for the newer one..')
                Q_main[index].remove(q_dd)
                Q_main[index].append(q_new)
                back_pointer.append((q_new, q, valid_phrase))
                return
            else:
                return
    Q_main[index].append(q_new)
    back_pointer.append((q_new, q, valid_phrase))
    return


def beam(Q_main, index, beam_width=5):
    running_max = -500

    # find the highest scoring state in the set
    for q in Q_main[index]:
        if q.alpha > running_max:
            running_max = q.alpha
            curr_max_state = q

    final_beam = running_max - beam_width

    final_list = []
    for q in Q_main[index]:
        if q.alpha >= final_beam:
            final_list.append(q)

    return final_list


bitstring = [0]*len(sentence)

q0 = State('<s>', '<s>', bitstring, 0, 0)
Q_main = {k: [] for k in range(len(sentence)+1)}
Q_main[0] = [q0]
back_pointer = []

for i in range(0, len(sentence)-1):
    for q in beam(Q_main, i, beam_width=5):
        for valid_phrase in ph(q):
#             print('\ncurrent q: {}'.format(q))
#             print('current valid_phrase: {}\n'.format(valid_phrase))
            q_new = next(q, valid_phrase)
            index = len(np.nonzero(q_new.bitstring)[0])
            add(Q_main, index, q_new, q, valid_phrase, back_pointer)

print(Q_main)