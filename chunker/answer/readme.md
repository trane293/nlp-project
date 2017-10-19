
# Phrasal Chunking, Homework 2
# CMPT 825 Natural Language Processing
##### Group Name: GroupNLP

### Running the code

### Training phase

    `python chunk.py -m groupNLP.model`

### Testing and Evaluation phase

    ~~~
    python perc.py -m groupNLP.model > output
    python score-chunks.py -t output
    ~~~

## Main Implementation
### Algorithm Description

The implemented algorithm in the `default.py` file is the Perceptron algorithm for Hidden Markov Model. The general Perceptron algorithm is given as:

1. Initialize a weight vector `w` with `d` dimensions, same as the number of features in the training set.
2. Repeat until convergence:
  1. For each example `x^{j}` in the training set `T`, calculate the following:
    1. Calculate the output of current model:
          1. `prediction(t)` = `w(t).x^{j}`
    2. Update the weights to penalize wrong decision:
          1. `w_{d}(t+1) = w_{d}(t) + (target_{j} - prediction_{j}(t))x^{j}_{d}`


To get the `prediction(t)`, we use the Viterbi Algorithm to search efficiently over exponential space and find the best tag sequence for a sentence which maximizes the quantity that is the product of the weight vector and the features.

Once a prediction is made, we update the weights using the following method:

1. For each `prediction_value` in `zip(prediction(t), target_{j})`:
  1. if `prediction_value` != `target_value`:
    1. `weights[x^{j}_{d}, target_value] += 1`  # add value to features that are correct
    2. `weights[x^{j}_{d}, prediction_value] -= 1`  # penalize the feature-prediction combo that is incorrect.

## Other Experiments

Other than baseline implementation, we also performed other experiments with different machine learning models by posing the problem of chunking as a supervised learning algorithm, where given a word and its POS_tag, we predict its chunk tag. The detailed description of the experiments and code is inside `answer/Experiments` folder. The description is in the form of a markdown file which is accompanied by an easy to read HTML file named as `README_EXPERIMENTS.asa224.md` and `README_EXPERIMENTS.asa224.html` respectivly. The code is inside `ML_Chunking.ipynb` which also has an HTML version `ML_Chunking.html`.
