## Assignment 2
## CMPT 825 Natural Language Processing
* Student Name: Anmol Sharma
* Student ID: asa224
* Student Email ID: asa224@sfu.ca

## Contributions:

1. Went through the theoretical underpinnings of the baseline algorithm as well as other material required to implement a solution, and helped the team understand the material.
2. Implemented the initial baseline algorithm after discussion with team.
3. Fixed all the major and minor bugs in the baseline implementation and finalized the code.
4. Cleaned the code, and added inline comments in chunk.py to make it more readable.
5. Updated the `readme.md` in the `answer/` folder with mathematical description of our algorithm.
6. Implemented a machine learning based solution to the chunking problem by cleaning the data myself.
  1. Prepared the dataset for this problem statement.
  2. Implemented the idea by training different classifiers using Python and Scikit-Learn.
  3. Observed comparable results, and documented the experiment.
7. Applied Multi-Class SVM, Random Forest and Naive Bayes towards this problem and achieved the best F1-Score of 80% using the SVM.
8. Wrote detailed markdown based file highlighting my machine learning based implementations, available in documented form in `answer/Experiments` named `README_EXPERIMENTS.asa224.md`.
9. For easy readability, the markdown version of this detailed file is also provided in the form of an HTML version, with the name `README_EXPERIMENTS.asa224.md.html`.

## Commit Identifiers

The following commit identifiers were obtained using the command `git log`. Please note these were made on the branch `new`, not `master`. The current solution will be on `master`.
~~~
commit 52034c1561a9f963674ac4c82505297ef2dba0aa
Author: asa224 <asa224@sfu.ca>
Date:   Wed Oct 18 20:36:05 2017 -0700

    finalized main readme, wrote markdown readme for other experiments, finalized personal readme

commit 2db02a2b3b243322d9dc5be271d0593aba2dec63
Author: asa224 <asa224@sfu.ca>
Date:   Wed Oct 18 19:07:23 2017 -0700

    cleaned working directory, made code standalone, generated model file after 30 epochs, added mathematical documentation in readme, added personal readme

commit 1683f3ff29fe383d66b3ad50c0434a4f55e54158
Author: asa224 <asa224@sfu.ca>
Date:   Wed Oct 18 16:58:05 2017 -0700

    implemented baseline correctly after fixing minor bugs, added helpful comments in code, generated crude output file after 1 epoch of training

commit 89ff2a93a3149d8e1491a3ef877f528a8a7edb9c
Author: asa224 <asa224@cs-mial-31.cs.sfu.ca>
Date:   Wed Oct 18 15:29:06 2017 -0700

    fixed print statements to route everything to stderr

commit d0413632f7a3fe3a5b4bf5aad81fcf862e43865f
Author: asa224 <asa224@cs-mial-31.cs.sfu.ca>
Date:   Wed Oct 18 15:16:24 2017 -0700

    fixed the issues with previous issues and implemented baseline again with correct logic

commit af41d823173c9361f030b1d677e6496dade36d63
Author: amiralis <amirali_sharifian@sfu.ca>
Date:   Wed Oct 18 13:33:39 2017 -0700

    Add some changes

commit 073355e89ee5b0744519eacd2b0fe1328fbb50f8
Author: asa224 <asa224@cs-mial-31.cs.sfu.ca>
Date:   Wed Oct 18 13:08:57 2017 -0700

    changed weight vector keys to only feature names

commit 6cd0e9ee5ce1870bd185a96cf9a3f2ead86da3ab
Author: asa224 <asa224@cs-mial-31.cs.sfu.ca>
Date:   Tue Oct 17 22:58:53 2017 -0700

    code inspection, some changes. need to fix testing data accuracy calculator and weight updation

commit 11ad591fe48c2e90899100765b2ba6b6dca99d57
Author: asa224 <asa224@cs-mial-31.cs.sfu.ca>
Date:   Tue Oct 17 20:43:51 2017 -0700

    fixed weight vector update code

commit 366e4d3d61b18c0032ef29af546c0daab199335f
Author: asa224 <asa224@cs-mial-31.cs.sfu.ca>
Date:   Tue Oct 17 19:43:46 2017 -0700

    added experiments using Naive Bayes, SVM and Random Forest classifiers for chunking task

commit 5dec7d85402dc239346b543dfcaabf71dfdffbf8
:
    changed weight vector keys to only feature names

commit 6cd0e9ee5ce1870bd185a96cf9a3f2ead86da3ab
Author: asa224 <asa224@cs-mial-31.cs.sfu.ca>
Date:   Tue Oct 17 22:58:53 2017 -0700

    code inspection, some changes. need to fix testing data accuracy calculator and weight updation

commit 11ad591fe48c2e90899100765b2ba6b6dca99d57
Author: asa224 <asa224@cs-mial-31.cs.sfu.ca>
Date:   Tue Oct 17 20:43:51 2017 -0700

    fixed weight vector update code

commit 366e4d3d61b18c0032ef29af546c0daab199335f
Author: asa224 <asa224@cs-mial-31.cs.sfu.ca>
Date:   Tue Oct 17 19:43:46 2017 -0700

    added experiments using Naive Bayes, SVM and Random Forest classifiers for chunking task

commit 5dec7d85402dc239346b543dfcaabf71dfdffbf8
Author: amiralis <amirali_sharifian@sfu.ca>
Date:   Tue Oct 17 19:29:56 2017 -0700

    First implimentation of the baseline

~~~
