Authors:
=========
- Annmol Sharma
- Amirali Sharifian
- Shreeashish Kumar


Summary
====================
Automatic  evaluation  metrics  are  fundamen-tally  important  for  Machine  Translation,  al-lowing  comparison  of  systems  performanceand efficient training. Current evaluation met-rics fall into two classes: heuristic approaches,like BLEU or METEOR, and those using su-pervised learning trained on human judgmentdata. Evaluating a machine translation systemusing  such  heuristic  metrics  is  much  faster,easier and cheaper compared to human eval-uations, which require trained bilingual evalu-ators.In this report, we present our approach to com-bine  multiple  heuristic  automatic  evaluationtechniques  using  machine  learning  and  thencompare the quality of translation of two dif-ferent MT system. Our results show that whileeach  heuristic  approach  can  provide  a  validcomparison between two system, but combin-ing the techniques using our machine learningapproach provide higher accuracy.

File
====================
Each file conatains implimentation of one of the techniques, the mathematical description and implimentation details are presented inside the report, here we briefly only name these files.

1. BLEU_score.py: `BLEU implimentation`
2. METEOR_score.py: `METEOR implimentation`
3. ROUGE_score.py: `ROUGE implimentation`
4. run.py: `The main python file`
5. ApplyMachineLearning.py: `Our machine learning implimentation file`

Output
====================
Each file contains our implimentation's result of runnig the evalution technique on our data.


1. output_ML_logistic
2. output_ML_NN
3. output_ML_SVM
4. blue.out.txt
5. meteor_out.txt
6. rouge.out.txt


How to run?
---------

There are three Python programs here (`-h` for usage):

 - `run.py` evaluates pairs of MT output hypotheses by comparing the number of words they match in a reference translation
 - `check.py` checks that the output file is correctly formatted
 - `score-evaluation.py` computes accuracy against human judgements

The commands are designed to work in a pipeline. For instance, this is a valid invocation:

    python default.py | python check.py | python score-evaluation.py

The `data/` directory contains a training set and a test set

 - `data/hyp1-hyp2-ref` is a file containing tuples of two translation hypotheses and a human reference translation.

 - `data/dev.answers` contains human judgements for the first half of the dataset, indicating whether the first hypothesis (hyp1) or the second hypothesis (hyp2) is better or equally good/bad.
