## Assignment 1
## CMPT 825 Natural Language Processing
* Student Name: Anmol Sharma
* Student ID: asa224
* Student Email ID: asa224@sfu.ca

## Contributions:

1. Implemented the base "chartEntry" datastructure with comparison operators.
2. Implemented wrapper for heapq default python datastructure to make it secure against unintentional underlying list modification.
3. Implemented the heap initialization part of the algorithm.
4. Posed the problem of chinese word segmentation as a one-to-one prediction problem suitable for LSTMs.
  1. Implemented the one-to-one word segmentation problem by preparing the dataset and training an LSTM.
  2. Obtained results very close to what we were getting on Unigram based segmentation baseline.
5. Reposed the problem as a sequence-to-sequence prediction problem which leverages the LSTM's time unrolling properties and it's long term memory.
  1. Prepared the dataset for this problem statement.
  2. Implemented the idea using an LSTM based RNN using Python and Keras.
  3. Observed comparable results, and meticulously documented the experiment.
6. Wrote detailed documentation for all the codes written by all members of the team.
7. Wrote detailed markdown based file highlighting my LSTM based implementations, available in documented form in answer/Experiments named `README_RNN_EXPERIMENTS.asa224.md`. For easy readability, the markdown version of this detailed file is also provided in the form of an HTML version, with the name `README_RNN_EXPERIMENTS.asa224.md.html`.
8. Implemented unigram and bigram count extracting code from 1M chinese sentence dataset.
