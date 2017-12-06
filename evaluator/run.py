
import cPickle as pickle

# import custom implementations
from METEOR_score import *
from ROGUE_score import *
from BLEU_score import *

def main():
    parser = argparse.ArgumentParser(description='Evaluate translation hypotheses.')
    parser.add_argument('-i', '--input', default='data/hyp1-hyp2-ref',
                        help='input file (default data/hyp1-hyp2-ref)')
    parser.add_argument('-r', '--reference', default='data/dev.answers',
                        help='input file (default data/hyp1-hyp2-ref)')
    parser.add_argument('-n', '--num_sentences', default=None, type=int,
                        help='Number of hypothesis pairs to evaluate')
    parser.add_argument('-a', '--alpha', default=0.9, type=float,
            help='Alpha parameter for METEOR score calculation')
    parser.add_argument('-b', '--beta', default=3.0, type=float,
            help='Beta parameter for METEOR score calculation')
    parser.add_argument('-g', '--gamma', default=0.5, type=float,
            help='Gamma parameter for METEOR score calculation')

    # note that if x == [1, 2, 3], then x[:None] == x[:] == x (copy); no need for sys.maxint
    opts = parser.parse_args()

    # we create a generator and avoid loading all sentences into a list
    def sentences():
        with open(opts.input) as f:
            for pair in f:
                yield [sentence.strip() for sentence in pair.split(' ||| ')]

    alignments_h1_ref = {}
    alignments_h2_ref = {}
    t = 0
    # vals = []
    num_chunks_h1 = {}
    num_chunks_h2 = {}
    score = {}

    result_vector = []
    # note: the -n option does not work in the original code
    for h1, h2, ref in islice(sentences(), opts.num_sentences):
        # Getting METEOR results
        if t % 1000 == 0:
            sys.stderr.write(str(t) + ' ')

        # Getting ROUGE scores

        h1_rogue = score_ROGUE(h1, ref, beta=1.2)
        h2_rogue = score_ROGUE(h2, ref, beta=1.2)


        # Getting BLUE scores

        h1_bleu = BLEU([h1], [[ref]])
        h2_bleu = BLEU([h2], [[ref]])

        # convert data to suitable form so that meteor implementation can work
        h1 = h1.split()
        h2 = h2.split()
        ref = ref.split()

        bitstring_h1 = [0]*len(h1)
        bitstring_ref = [0]*len(ref)
        alignments_h1_ref[t] = []

        h1, ref, bitstring_h1, bitstring_ref, alignments_h1_ref[t] = exactMatching(h1, ref, bitstring_h1, bitstring_ref, alignments_h1_ref[t])
        h1, ref, bitstring_h1, bitstring_ref, alignments_h1_ref[t] = stemmerMatching(h1, ref, bitstring_h1, bitstring_ref, alignments_h1_ref[t])
        h1, ref, bitstring_h1, bitstring_ref, alignments_h1_ref[t] = synonymMatching(h1, ref, bitstring_h1, bitstring_ref, alignments_h1_ref[t])

        _tm, num_chunks_h1[t] = chunk(alignments_h1_ref[t])

        score_h1 = scoreMETEOR(h1, ref, num_chunks_h1[t], alignments_h1_ref[t], alpha=opts.alpha, beta=opts.beta, gamma=opts.gamma)

        bitstring_h2 = [0]*len(h2)
        bitstring_ref = [0]*len(ref)
        alignments_h2_ref[t] = []

        h2, ref, bitstring_h2, bitstring_ref, alignments_h2_ref[t] = exactMatching(h2, ref, bitstring_h2, bitstring_ref, alignments_h2_ref[t])
        h2, ref, bitstring_h2, bitstring_ref, alignments_h2_ref[t] = stemmerMatching(h2, ref, bitstring_h2, bitstring_ref, alignments_h2_ref[t])
        h2, ref, bitstring_h2, bitstring_ref, alignments_h2_ref[t] = synonymMatching(h2, ref, bitstring_h2, bitstring_ref, alignments_h2_ref[t])

        _tm, num_chunks_h2[t] = chunk(alignments_h2_ref[t])

        score_h2 = scoreMETEOR(h2, ref, num_chunks_h2[t], alignments_h2_ref[t], alpha=0.9, beta=3.0, gamma=opts.gamma)

        # create the resultant vector

        result_vector.append([h1_bleu, h2_bleu, h1_rogue, h2_rogue, score_h1, score_h2])

        t += 1

    print('dumping result vector..')
    pickle.dump(result_vector, open('./result_vector_sleek.p', 'wb'))
    print('Done!')

if __name__ == '__main__':
    main()