from __future__ import print_function
from itertools import islice # slicing for iterators
from nltk.stem import *
from nltk.corpus import wordnet as wn
import sys
import argparse # optparse is deprecated
inp = 'data/hyp1-hyp2-ref'
def sentences():
    with open(inp) as f:
        for pair in f:
            yield [sentence.strip().split() for sentence in pair.split(' ||| ')]

def exactMatching(h1, h2, bitstring_h1, bitstring_h2, alignments):
    for ind_h1_word, word in enumerate(h1):
        try:
            ind_h2_word = h2.index(word)
            if bitstring_h2[ind_h2_word] == 0:
                # this is a candidate alignment
                alignments.append((ind_h1_word, ind_h2_word))
                bitstring_h2[ind_h2_word] = 1
                bitstring_h1[ind_h1_word] = 1
            else:
                # this is not a candidate alignment
                # search for next occurences
                occurences = [i for i, j in enumerate(h2) if j == word]
                for i in occurences:
                    if bitstring_h2[i] == 0:
                        alignments.append((ind_h1_word, i))
                        bitstring_h2[i] = 1
                        bitstring_h1[ind_h1_word] = 1
        except ValueError:
            continue

    alignments = sorted(alignments, key=lambda x: x[0])

    return h1, h2, bitstring_h1, bitstring_h2, alignments

def stemmerMatching(h1, h2, bitstring_h1, bitstring_h2, alignments):
    un_matched_h1 = []
    un_matched_h2 = []

    for idx, word in enumerate(h1):
        if bitstring_h1[idx] == 0:
            un_matched_h1.append((idx, word))

    for idx, word in enumerate(h2):
        if bitstring_h2[idx] == 0:
            un_matched_h2.append((idx, word))

    stemmer = PorterStemmer()

    stemmed_h1 = []
    stemmed_h2 = []

    if un_matched_h1 == [] or un_matched_h2 == []:
#         print('no further alignment is possible..')
        return h1, h2, bitstring_h1, bitstring_h2, alignments

    for ind_h1, word_h1 in un_matched_h1:
        stemmed_h1.append((ind_h1, stemmer.stem(word_h1.decode('utf-8'))))

    for ind_h2, word_h2 in un_matched_h2:
        stemmed_h2.append((ind_h2, stemmer.stem(word_h2.decode('utf-8'))))

    for id1, wd1 in stemmed_h1:
        for id2, wd2 in stemmed_h2:
            if wd1 == wd2 and (bitstring_h1[id1] == 0 and bitstring_h2[id2] == 0):
                alignments.append((id1, id2))
                bitstring_h1[id1] = 1
                bitstring_h2[id2] = 1

    return h1, h2, bitstring_h1, bitstring_h2, alignments

def synonymMatching(h1, h2, bitstring_h1, bitstring_h2, alignments):
    un_matched_h1 = []
    un_matched_h2 = []

    for idx, word in enumerate(h1):
        if bitstring_h1[idx] == 0:
            un_matched_h1.append((idx, word))

    for idx, word in enumerate(h2):
        if bitstring_h2[idx] == 0:
            un_matched_h2.append((idx, word))

    if un_matched_h1 == [] or un_matched_h2 == []:
#         print('no further alignment is possible..')
        return h1, h2, bitstring_h1, bitstring_h2, alignments

    synonyms_h1 = []
    synonyms_h2 = []
    empty_set = set({})
    for id1, wd1 in un_matched_h1:
        wd1_synset = set(wn.synsets(wd1.decode('utf-8')))
        for id2, wd2 in un_matched_h2:
            if wd1_synset.intersection(wn.synsets(wd2.decode('utf-8'))) != empty_set and (bitstring_h1[id1] == 0 and bitstring_h2[id2] == 0):
                alignments.append((id1, id2))
                bitstring_h1[id1] = 1
                bitstring_h2[id2] = 1

    alignments = sorted(alignments, key=lambda x: x[0])

    return h1, h2, bitstring_h1, bitstring_h2, alignments

def chunk(alignments):
    chunks = []
    for i in range(0, len(alignments)):
        # start the chunk, but check if its already in previous chunk:

        if i > 0 and alignments[i] in chunks[-1]:
#             print('this alignment already belongs to previous chunk..moving on!')
            continue

#         print('starting chunk {}'.format(alignments[i]))
        chunks.append([alignments[i]])

        for j in range(i+1, len(alignments)):
            if alignments[j-1][1] - alignments[j][1] == -1:
                # append to current chunk
#                 print('adding unigram alignment {} to previous chunk {}'.format(alignments[j], chunks[-1]))
                chunks[-1].append(alignments[j])
            else:
#                 print('moving on to next alignment')
                break

    return chunks, len(chunks)

def scoreMETEOR(h, ref, num_chunks, alignment, alpha=0.9, beta=3.0, gamma=0.5):
    m = float(len(alignment))
    r = float(len(ref))
    t = float(len(h))
    ch = float(num_chunks)

    P = m / t
    R = m / r

    if P == 0.0 and R == 0.0:
        return 0.0
    else:
        F_mean = P*R / (alpha * P + (1 - alpha)*R)

        frag = ch / m

        penalty = gamma * (frag ** beta)

        score = (1 - penalty) * F_mean

        return score

def main():
    parser = argparse.ArgumentParser(description='Evaluate translation hypotheses.')
    parser.add_argument('-i', '--input', default='data/hyp1-hyp2-ref',
            help='input file (default data/hyp1-hyp2-ref)')
    parser.add_argument('-n', '--num_sentences', default=None, type=int,
            help='Number of hypothesis pairs to evaluate')
    # note that if x == [1, 2, 3], then x[:None] == x[:] == x (copy); no need for sys.maxint
    opts = parser.parse_args()

    alignments_h1_ref = {}
    alignments_h2_ref = {}
    t = 0
    # vals = []
    num_chunks_h1 = {}
    num_chunks_h2 = {}
    score = {}

    for h1, h2, ref in islice(sentences(), opts.num_sentences):
        sys.stderr.write(str(t) + ' ')
        bitstring_h1 = [0]*len(h1)
        bitstring_ref = [0]*len(ref)
        alignments_h1_ref[t] = []

        h1, ref, bitstring_h1, bitstring_ref, alignments_h1_ref[t] = exactMatching(h1, ref, bitstring_h1, bitstring_ref, alignments_h1_ref[t])
        h1, ref, bitstring_h1, bitstring_ref, alignments_h1_ref[t] = stemmerMatching(h1, ref, bitstring_h1, bitstring_ref, alignments_h1_ref[t])
        h1, ref, bitstring_h1, bitstring_ref, alignments_h1_ref[t] = synonymMatching(h1, ref, bitstring_h1, bitstring_ref, alignments_h1_ref[t])

        _tm, num_chunks_h1[t] = chunk(alignments_h1_ref[t])

        score_h1 = scoreMETEOR(h1, ref, num_chunks_h1[t], alignments_h1_ref[t], alpha=0.9, beta=3.0, gamma=0.5)

        bitstring_h2 = [0]*len(h2)
        bitstring_ref = [0]*len(ref)
        alignments_h2_ref[t] = []

        h2, ref, bitstring_h2, bitstring_ref, alignments_h2_ref[t] = exactMatching(h2, ref, bitstring_h2, bitstring_ref, alignments_h2_ref[t])
        h2, ref, bitstring_h2, bitstring_ref, alignments_h2_ref[t] = stemmerMatching(h2, ref, bitstring_h2, bitstring_ref, alignments_h2_ref[t])
        h2, ref, bitstring_h2, bitstring_ref, alignments_h2_ref[t] = synonymMatching(h2, ref, bitstring_h2, bitstring_ref, alignments_h2_ref[t])

        _tm, num_chunks_h2[t] = chunk(alignments_h2_ref[t])

        score_h2 = scoreMETEOR(h2, ref, num_chunks_h2[t], alignments_h2_ref[t], alpha=0.9, beta=3.0, gamma=0.5)

        score[t] = [score_h1, score_h2]
        print(1 if score[t][0] > score[t][1] else # \begin{cases}
                    (0 if score[t][0] == score[t][1]
                        else -1), file=sys.stdout)
        t += 1

# convention to allow import of this file as a module
if __name__ == '__main__':
    main()
