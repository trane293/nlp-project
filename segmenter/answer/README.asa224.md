## Assignment 1
## CMPT 825 Natural Language Processing
* Student Name: Anmol Sharma
* Student ID: asa224
* Student Email ID: anmol_sharma@sfu.ca

## Contributions:

1. Implemented the base "chartEntry" datastructure with comparison operators.
2. Implemented wrapper for heapq default python datastructure to make it secure against unintentional underlying list modification.
3. Implemented the heap initialization part of the algorithm.
4. Posed the problem of chinese word segmentation as a one-to-one prediction problem suitable for LSTMs.
  1. Implemented the one-to-one word segmentation problem by preparing the dataset and training an LSTM.
  2. Obtained results very close to what we were getting on Unigram based segmentation baseline.
5. Reposed the problem as a sequence-to-sequence prediction problem which leverages the LSTM's time unrolling properties and it's long term memory.
  1. Prepared the dataset for this problem statement with idea from [^1].
  2. Implemented the idea using a Bidirectional LSTM using Python and Keras.
  3. Observed (what results)
6. Wrote detailed documentation for all the codes written by all members of the team.
7. Wrote detailed markdown based file highlighting my LSTM based implementations, available in highly documented form in answer/Experiments, accompanied by the README file named `README_RNN_EXPERIMENTS.asa224.md`.
