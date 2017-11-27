import optparse
import sys
import models
from collections import namedtuple
from math import log10

optparser = optparse.OptionParser()
optparser.add_option("-d","--distortion limit",dest="distort")
optparser.add_option("-i", "--input", dest="input", default="../data/input", help="File containing sentences to translate (default=data/input)")
optparser.add_option("-t", "--translation-model", dest="tm", default="../data/tm", help="File containing translation model (default=data/tm)")
optparser.add_option("-l", "--language-model", dest="lm", default="../data/lm", help="File containing ARPA-format language model (default=data/lm)")
optparser.add_option("-n", "--num_sentences", dest="num_sents", default=sys.maxint, type="int", help="Number of sentences to decode (default=no limit)")
optparser.add_option("-k", "--translations-per-phrase", dest="k", default=10, type="int", help="Limit on number of translations to consider per phrase (default=1)")
optparser.add_option("-s", "--stack-size", dest="s", default=3000, type="int", help="Maximum stack size (default=1)")
optparser.add_option("-v", "--verbose", dest="verbose", action="store_true", default=False,  help="Verbose mode (default=off)")
opts = optparser.parse_args()[0]

tm = models.TM(opts.tm, opts.k)
lm = models.LM(opts.lm)

french = [tuple(line.strip().split()) for line in open(opts.input).readlines()[:opts.num_sents]]

# tm should translate unknown words as-is with probability 1
for word in set(sum(french, ())):
    if (word,) not in tm:
        tm[(word,)] = [models.phrase(word, 0.0)]

def get_predecessor(q):
    return "" if q.predecessor is None else "%s%s " % (get_predecessor(q.predecessor), q.phrase.english)


def beam(Q, beta=9):
    states = sorted(Q.itervalues(), key=lambda h: -(h.logprob + h.future_cost))
    max = states[0].logprob + states[0].future_cost
    return [hypothesis for hypothesis in states if
           (hypothesis.logprob + hypothesis.future_cost) >= (max - beta)]


def get_future_cost_matrix(f_line, tm):
    cost_matrix = [[0 for _ in f_line] for _ in f_line]
    for length in range(1, len(f_line)+1):
        for start in range(0,len(f_line)- length+1):
            end = start + length
            cost_matrix[start][end-1] = -1*sys.maxint
            if f_line[start:end] in tm:
                e_phrase = max(tm[f_line[start:end]], key=lambda q: q.logprob)
                cost_matrix[start][end-1] = max(tm[f_line[start:end]], key=lambda q: q.logprob).logprob
                lm_state = ("<s>",)
                for word in e_phrase.english.split():
                    (lm_state, word_logprob) = lm.score(lm_state, word)
                    cost_matrix[start][end - 1] += word_logprob

            for ind in xrange(start, end-1):
                if cost_matrix[start][ind] + cost_matrix[ind+1][end-1] > cost_matrix[start][end-1]:
                    cost_matrix[start][end-1] = cost_matrix[start][ind] + cost_matrix[ind+1][end-1]

            return cost_matrix


for number, f_line in enumerate(french):
    sys.stderr.write("Working on {}\n".format(number + 1))

    future_cost_matrix = get_future_cost_matrix(f_line, tm)
    # print future_cost_matrix

    eta = 1.0
    mstack_size = 3000
    d = 4

    # Change the stack size, eta and distortion limit if the length of the sentence changes
    if len(f_line) < 9:
      eta = 0.8
      d = 4

    if len(f_line) > 13:
      eta = 1.2
      d = 6

    bit_vec = [0] * len(f_line)

    hypothesis = namedtuple("hypothesis", "logprob, lm_state, bit_vec, last_ind, predecessor, phrase, future_cost")
    initial_hypothesis = hypothesis(0.0, lm.begin(), bit_vec, 0, None, None, 0)

    Q_vec  = [{} for _ in f_line] + [{}]

    bitstring = ''.join(str(bit) for bit in bit_vec)
    Q_vec[0][bitstring] = initial_hypothesis

    for ind, Q in enumerate(Q_vec[:-1]):
        # Beam the Q_vec and prune
        beam_list = beam(Q,beta=9)
        beam_list = beam_list[:mstack_size]

        for q in beam_list:
            # ph function inlined
            for i in range(0, len(f_line)):
                for j in range(i+1, len(f_line)+1):

                    flag_1 = False
                    for bit in q.bit_vec[i:j]:
                        if bit == 1:
                            flag_1 = True

                    if flag_1 == True:
                        continue

                    # if 1 in q.bit_vec[i:j]:
                    #     continue

                    if ind == 0 and abs(1 - i) > d:
                        break

                    phrase_french = f_line[i:j]

                    if phrase_french in tm:
                        new_bitvec = q.bit_vec[:]

                        # Calculate the lm scores and update the lmstate
                        for bit in range(i,j):
                            new_bitvec[bit] = 1

                        for phrase in tm[phrase_french]:
                            logprob = q.logprob + phrase.logprob
                            lm_state  = q.lm_state

                            for word in phrase.english.split():
                                (lm_state, word_logprob) = lm.score(lm_state, word)
                                logprob += word_logprob
                            if 0 not in new_bitvec:
                                logprob += lm.end(lm_state)
                            if abs(q.last_ind - i + 1) > d:
                                logprob += log10(eta) * abs(q.last_ind - i + 1)

                            st = -1
                            future_cost = 0
                            for bit in xrange(0, len(new_bitvec)):
                                if st == -1 and new_bitvec[bit] == 0:
                                    st = bit
                                if st != -1 and new_bitvec[bit] == 1:
                                    future_cost += future_cost_matrix[st][bit - 1]
                                    st = -1
                            if st != -1:
                                future_cost += future_cost_matrix[st][len(new_bitvec) - 1]

                            new_hypothesis = hypothesis(logprob, lm_state, new_bitvec, j - 1, q, phrase, future_cost)

                            bitstring = ''.join(str(bit) for bit in new_bitvec)

                            if bitstring not in Q_vec[ind + j - i] or Q_vec[ind + j - i][bitstring].logprob < logprob:
                                Q_vec[ind+j-i][bitstring] = new_hypothesis

    best =  max(Q_vec[-1].itervalues(), key=lambda q: q.logprob)


    print get_predecessor(best)

