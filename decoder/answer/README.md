# Decoding, Homework 4
# CMPT 825 Natural Language Processing
##### Group Name: GroupNLP

### Running the code


    ~~~
    python2 decoder.py > file.out
    python2 baseline.py > file.out
    ~~~

### Testing and Evaluation phase

    ~~~
    python2 score-decoder < answer/file.out
    ~~~

## Main Implementation
### Algorithm Description

The team implemented _*two algorithms*_. The first one was vanilla beam search algorithm as described in [1],
implemented independently by Anmol Sharma. The implementation closely follows the exact algorithm description in the text.

The performance of this implementation was not upto the mark. It performed barely above baseline, and was largely
untunable even after repeated changes in beam_width and distortion penalties.

Later, Shreeashish and Amirali independently reimplemented the beam search algorithm, which also made use of the
future cost value as described in [2]. The algorithm although mostly similar in nature to vanilla beam search,
has an extra "future cost" element in the state vector.
The parameter is later used to prune the search space. From any state, an estimate of the cost of translating the rest of the sentence calculated and the low cost translations are picked. Intuitively, this biases the next translation to be made towards the parts of the sentence which are common and would be considered easy.

## References
[1] http://anoopsarkar.github.io/nlp-class/assets/mcollins-notes/pb.pdf
[2] http://anoopsarkar.github.io/nlp-class/assets/slides/06-decoding.pdf
