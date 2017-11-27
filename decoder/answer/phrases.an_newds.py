# coding=utf-8
from __future__ import print_function
import models
import itertools
from copy import deepcopy
from collections import namedtuple
import numpy as np
import sys
import time

class NotFound(Exception):
    pass

def createQfromHashable(q0, alpha):
    q_orig = State(q0[0], q0[1], list(q0[2]), q0[3], alpha)
    return q_orig

def createHashable(q0):
    q_hash = (q0.e1, q0.e2, tuple(q0.bitstring), q0.r)
    return q_hash, q0.alpha

def ph(sc_P, q, d=4):
    ph_states = []
    for state in sc_P:
        flag = True  # we assume it as a valid state
        s = state[0]
        t = state[1]

        orig_bitstring = q.bitstring

        '''
        Step 1: Ensure bit string is not overlapped
        '''
        for _num in range(s, t+1):
            if orig_bitstring[_num] != 0:
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
            print('MAJOR ERROR!', file=sys.stderr)
            print('The Ph(q) function has issues if you can read this...', file=sys.stderr)
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

    new_state = State(e1=second_last, e2=last_word, bitstring=new_bitstring, r=t, alpha=new_alpha)
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


def addHashable(Q_main, index, q_new, q, valid_phrase, back_pointer):

    q_new_hash, q_new_alpha = createHashable(q_new)

    if q_new_hash in Q_main[index]: # there is a state
        q_dd_alpha = Q_main[index][q_new_hash]

        if q_new_alpha > q_dd_alpha:
            # remove the older one
            _tm = Q_main[index].pop(q_new_hash, None)
            assert _tm != None
            Q_main[index][q_new_hash] = q_new_alpha
            back_pointer.append((q_new, q, valid_phrase))
        else:
            return
    Q_main[index][q_new_hash] = q_new_alpha
    back_pointer.append((q_new, q, valid_phrase))
    return


def beamHash(Q_main, index, beam_width=5):

    max_alpha = max(Q_main[index].values())

    final_beam = max_alpha - beam_width

    for q, alpha in Q_main[index].items():
        if alpha < final_beam:
            _tmp = Q_main[index].pop(q, None)
            assert _tmp != None

    return Q_main[index]


def getBestPerformerStateHash(Q_main, ind):
    best_state = max(Q_main[ind], key=Q_main[ind].get)
    best_state_alpha = Q_main[ind][best_state]
    return createQfromHashable(best_state, best_state_alpha)

def search(back_pointer, value, which_index):
    for entry in back_pointer:
        if value == entry[which_index]:
            return entry[2], entry[1]
    return False, False

# Start working on a single sentence generalize later
inp_file = open('../data/input')
outfile = open('beam_search_output_tmux', 'wb')

# sentence_orig = "honorables sénateurs , que se est - il passé ici , mardi dernier ?"
# Get a translation for each set of words using tm
tm = models.TM("../data/tm", 1)
lm = models.LM("../data/lm")

# define a numedtuple which will be the state vector q
State = namedtuple('State', "e1 e2 bitstring r alpha")
sent_num = 0

# runs = [16]
for sentence_orig in inp_file:
    # start = time.time()
    # if sent_num+1 not in runs:
    #     sent_num += 1
    #     continue
    print('sentence {}'.format(sent_num+1), file=sys.stderr)
    sentence = sentence_orig.split(' ')[0:-1]
    sc_P = []

    for i in range(0, len(sentence)):
        for j in range(i, len(sentence)):
            if i == j:
                subset = (sentence[i],)
            else:
                subset = tuple(sentence[i:j + 1])
            try:
                tm_output = tm[subset]
                if len(tm_output) > 1:
                    for k in range(0, len(tm_output)):
                        sc_P.append((i, j, tm_output[k]))
                else:
                    sc_P.append((i, j, tm_output[0]))
            except KeyError:
                continue
    try:
        for d in [4,6,8]:
            for beam_value in [14, 18, 24, 30]:
                print('using beam_value {}'.format(beam_value), file=sys.stderr)
                print('using d {}'.format(d), file=sys.stderr)
                bitstring = [0]*len(sentence)

                q0 = State('<s>', '<s>', bitstring, 0, 0)
                Q_main = {k: {} for k in range(len(sentence)+1)}
                q0_hash, q0_alpha = createHashable(q0)
                Q_main[0] = {q0_hash:q0.alpha}
                back_pointer = []

                for i in range(0, len(sentence)-1):
                    beam_output = beamHash(Q_main, i, beam_width=beam_value)
                    for q_hash, alpha in beam_output.items():
                        ph_output = ph(sc_P=sc_P, q=createQfromHashable(q_hash, alpha), d=d) # earlier running with 4
                        for valid_phrase in ph_output:
                            q_new = next(createQfromHashable(q_hash, alpha), valid_phrase)
                            index = len(np.nonzero(q_new.bitstring)[0])
                            addHashable(Q_main, index, q_new, createQfromHashable(q_hash, alpha), valid_phrase, back_pointer)
                    sys.stderr.write('.')

                if Q_main[len(sentence)] != {}:
                    print('worked! moving on..', file=sys.stderr)
                    raise Exception

        print('[BACKOFF] SUBOPTIMAL TRANSLATION', file=sys.stderr)

        if Q_main[len(sentence)-1] != {}:
            print('getting second last..', file=sys.stderr)
            end_point = getBestPerformerStateHash(Q_main, len(sentence)-1)
        elif Q_main[len(sentence)-2] != {}:
            print('getting third last..', file=sys.stderr)
            end_point = getBestPerformerStateHash(Q_main, len(sentence) - 2)

        ptr = end_point
        phrase = [1, 1]
        phrase_list_final = []
        print('backtracking..', file=sys.stderr)
        while (phrase[0] != 0 and phrase[1] != 0):
            phrase, ptr = search(back_pointer=back_pointer, value=ptr, which_index=0)
            phrase_list_final.append(phrase)
        print('sorting the backtrace..', file=sys.stderr)
        phrase_list_final = sorted(phrase_list_final, key=lambda tup: tup[0])

        print('writing to file..', file=sys.stderr)
        for phrase in phrase_list_final:
            sys.stdout.write(phrase[2].english + ' ')
            outfile.write(phrase[2].english + ' ')
        sys.stdout.write('\n')
        outfile.write('\n')

        sent_num += 1

    except Exception:
        print('getting best performer in the translated list', file=sys.stderr)
        end_point = getBestPerformerStateHash(Q_main, len(sentence))

        ptr = end_point
        phrase = [1, 1]
        phrase_list_final = []
        print('backtracking..', file=sys.stderr)
        while (phrase[0] != 0 and phrase[1] != 0):
            phrase, ptr = search(back_pointer=back_pointer, value=ptr, which_index=0)
            phrase_list_final.append(phrase)
        print('sorting the backtrace..', file=sys.stderr)
        phrase_list_final = sorted(phrase_list_final, key=lambda tup: tup[0])

        print('writing to file..', file=sys.stderr)
        for phrase in phrase_list_final:
            sys.stdout.write(phrase[2].english + ' ')
            outfile.write(phrase[2].english + ' ')
        sys.stdout.write('\n')
        outfile.write('\n')

        sent_num += 1
    # end = time.time()
    # print('Time taken for this sentence: {}'.format(end-start))
