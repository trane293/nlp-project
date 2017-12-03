"""
Computes the BLEU, ROUGE
using the COCO metrics scripts
"""
# from bleu.bleu import Bleu
from bleu.bleu_imp import *
from rouge.rouge import Rouge
import glob

import argparse
from itertools import islice

inp = 'data/hyp1-hyp2-ref'

def load_textfiles(references, hypothesis):
    hypo = {idx: [lines.strip()] for (idx, lines) in enumerate(hypothesis)}
    # take out newlines before creating dictionary
    raw_refs = [map(str.strip, r) for r in zip(references)]
    refs = {idx: rr for idx, rr in enumerate(raw_refs)}
    # sanity check that we have the same number of references as hypothesis
    if len(hypo) != len(refs):
        raise ValueError("There is a sentence number mismatch between the inputs")
    return refs, hypo


def score(ref, hypo):
    """
    ref, dictionary of reference sentences (id, sentence)
    hypo, dictionary of hypothesis sentences (id, sentence)
    score, dictionary of scores
    """
    scorers = [
        (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
        (Rouge(), "ROUGE_L"),
    ]
    final_scores = {}
    for scorer, method in scorers:
        score, scores = scorer.compute_score(ref, hypo)
        if type(score) == list:
            for m, s in zip(method, score):
                final_scores[m] = s
        else:
            final_scores[method] = score
    return final_scores


def compute_metrics(reference, hypothesis):
    ref, hypo = load_textfiles(list([reference]), list([hypothesis]))
    return score(ref, hypo)



def main():
    parser = argparse.ArgumentParser(description='Evaluate translation hypotheses.')
    parser.add_argument('-i', '--input', default='data/hyp1-hyp2-ref',
                        help='input file (default data/hyp1-hyp2-ref)')
    parser.add_argument('-n', '--num_sentences', default=None, type=int,
                        help='Number of hypothesis pairs to evaluate')
    parser.add_argument('-m', '--model', default='Bleu_1',
                        help='Choose between models',
                        choices=['Bleu_1', 'Bleu_2', 'Bleu_3', 'Bleu_4', 'ROUGE_L'])
    # note that if x == [1, 2, 3], then x[:None] == x[:] == x (copy); no need for sys.maxint
    opts = parser.parse_args()

    # we create a generator and avoid loading all sentences into a list
    def sentences():
        with open(opts.input) as f:
            for pair in f:
                yield [sentence.strip() for sentence in pair.split(' ||| ')]

    # note: the -n option does not work in the original code
    for h1, h2, ref in islice(sentences(), opts.num_sentences):

        # Getting BLUE and ROUGE data

        h1_scores = compute_metrics(ref, h1)
        h2_scores = compute_metrics(ref, h2)


        # Getting BLUE custom implementation
        h1_bleu = BLEU([h1], [[ref]])
        h2_bleu = BLEU([h2], [[ref]])



        # print(1 if h1_scores[opts.model]  > h2_scores[opts.model] else # \begin{cases}
        #         (0 if h1_scores[opts.model] == h2_scores[opts.model]
        #             else -1)) # \end{cases}

        # print(1 if h1_score > h2_score else # \begin{cases}
        #         (0 if h1_score == h2_score
        #             else -1)) # \end{cases}

if __name__ == '__main__':
    main()

