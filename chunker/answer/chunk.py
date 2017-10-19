"""
You have to write the perc_train function that trains the feature weights using the perceptron algorithm for the CoNLL 2000 chunking task.
Each element of train_data is a (labeled_list, feat_list) pair.

Inside the perceptron training loop:

    - Call perc_test to get the tagging based on the current feat_vec and compare it with the true output from the labeled_list

    - If the output is incorrect then we have to update feat_vec (the weight vector)

    - In the notation used in the paper we have w = w_0, w_1, ..., w_n corresponding to \phi_0(x,y), \phi_1(x,y), ..., \phi_n(x,y)

    - Instead of indexing each feature with an integer we index each feature using a string we called feature_id

    - The feature_id is constructed using the elements of feat_list (which correspond to x above) combined with the output tag (which correspond to y above)

    - The function perc_test shows how the feature_id is constructed for each word in the input, including the bigram feature "B:" which is a special case

    - feat_vec[feature_id] is the weight associated with feature_id

    - This dictionary lookup lets us implement a sparse vector dot product where any feature_id not used in a particular example does not participate in the dot product

    - To save space and time make sure you do not store zero values in the feat_vec dictionary which can happen if \phi(x_i,y_i) - \phi(x_i,y_{perc_test}) results in a zero value

    - If you are going word by word to check if the predicted tag is equal to the true tag, there is a corner case where the bigram 'T_{i-1} T_i' is incorrect even though T_i is correct.

"""

from __future__ import print_function
import perc
import sys, optparse, os
from collections import defaultdict


def perc_train(train_data, tagset, numepochs):
    # initialize feature vector to defaultdict with ints
    feat_vec = defaultdict(int)

    # defauklt tag is B-NP, which is at first place in tagset
    default_tag = tagset[0]
    print('Starting training process...', file=sys.stderr)
    for e in range(numepochs):
        # variable to track mistakes made in this epoch and total words seen in the current epoch
        mistakes = total_instances = 0
        print('Currently in Epoch {}'.format(e), file=sys.stderr)
        # iterate over each sentence
        for (labeled_list, feat_list) in train_data:
            # feat_index should go back to zero after one sentence is complete, this was a major bug which lead to
            # overflow of feat_index when the sentence changed...
            feat_index = 0

            # get the ground truth targets for this particular sentence
            ground_truth = []
            for t in labeled_list:
                ground_truth.append(t.split(' ')[2])

            # get the output from the viterbi algorithm
            output = perc.perc_test(feat_vec, labeled_list, feat_list, tagset, default_tag)

            # just an assertion to satiate my paranoia
            assert len(ground_truth) == len(output)
            if output != ground_truth:
                # iterate over each word in the sentence
                for num in range(len(output)):
                    total_instances += 1

                    # get the features for this particular word, and return index where next word's features start
                    (feat_index, feats) = perc.feats_for_word(feat_index, feat_list)

                    # if we made a mistake on this word, update weight vectors
                    if output[num] != ground_truth[num]:
                        mistakes += 1

                        # iterate over all the features (we have 20 features)
                        for f_num in range(len(feats)):
                            # print("INFO:root:updating feat_vec with feature_id: ({}, {}) value: 1".format(feats[f_num], ground_truth[num]), file=sys.stderr)
                            feat_vec[feats[f_num], ground_truth[num]] += 1
                            # print("INFO:root:updating feat_vec with feature_id: ({}, {}) value: -1".format(feats[f_num], output[num]), file=sys.stderr)
                            feat_vec[feats[f_num], output[num]] -= 1
        print('number of mistakes: {}'.format(mistakes), file=sys.stderr)
        print('current error on training set: {}'.format(float(mistakes*100)/total_instances), file=sys.stderr)
        # print('feat_vec: {}'.format(feat_vec), file=sys.stderr)
    return feat_vec

if __name__ == '__main__':
    optparser = optparse.OptionParser()
    optparser.add_option("-t", "--tagsetfile", dest="tagsetfile", default=os.path.join("../data", "tagset.txt"), help="tagset that contains all the labels produced in the output, i.e. the y in \phi(x,y)")
    optparser.add_option("-i", "--trainfile", dest="trainfile", default=os.path.join("../data", "train.txt.gz"), help="input data, i.e. the x in \phi(x,y)")
    optparser.add_option("-f", "--featfile", dest="featfile", default=os.path.join("../data", "train.feats.gz"), help="precomputed features for the input data, i.e. the values of \phi(x,_) without y")
    optparser.add_option("-e", "--numepochs", dest="numepochs", default=int(30), help="number of epochs of training; in each epoch we iterate over over all the training examples")
    optparser.add_option("-m", "--modelfile-save", dest="modelfile", default=os.path.join("groupNLP.model"), help="filename to store the trained weights in")
    (opts, _) = optparser.parse_args()

    tagset = perc.read_tagset(opts.tagsetfile)
    print("reading train data ...", file=sys.stderr)
    train_data = perc.read_labeled_data(opts.trainfile, opts.featfile)
    print("done.", file=sys.stderr)
    print('starting training for {} epochs...'.format(opts.numepochs))
    feat_vec = perc_train(train_data, tagset, opts.numepochs)
    print('writing weights to the model file...')
    perc.perc_write_to_file(feat_vec, opts.modelfile)
    print('done writing to model file {}!'.format(opts.modelfile))
