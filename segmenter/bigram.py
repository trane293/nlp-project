from __future__ import print_function
import sys, codecs, optparse, os
import heapq as heapq
import numpy as np
from heap import *
# pdb.set_trace()

optparser = optparse.OptionParser()
optparser.add_option("-c", "--unigramcounts", dest='counts1w', default=os.path.join('data', 'count_1w.txt'), help="unigram counts")
optparser.add_option("-b", "--bigramcounts", dest='counts2w', default=os.path.join('data', 'count_2w.txt'), help="bigram counts")
optparser.add_option("-i", "--inputfile", dest="input", default=os.path.join('data', 'input'), help="input file to segment")
(opts, _) = optparser.parse_args()

Pw1 = Pdist(opts.counts1w)
Pw2 = Pdist(opts.counts2w)

def bigram(input_filename='data/input', sort_acc_to='log_prob'):

    out_file = open('outfile', 'wb')

    with open(input_filename) as f:

        # iterate over all the lines in the input file
        for line in f:

            # initialize the dynamic programming table chart and heap
            chart = dict()
            heap = Heap()
            utf8line = unicode(line.strip(), 'utf-8')

            """
            Step 1:
              Initializing step
              Finding each word that matches input at position 0
            """

            num_observ = 0
            for key in Pw2:
                # for all keys in the probability distribution
                # check if the sentence starts with this word
                if utf8line.startswith(key):
                    entry = chartEntry("".join(key).encode('utf-8'),
                            start_pos=0,
                            end_pos=len(key)-1,
                            log_prob=np.log2(Pw2("".join(key))),
                            back_ptr=None,
                            sort_acc_to=sort_acc_to)

                    heap.push(entry)
                    num_observ += 1

            """
            Check whether the pattern exists in our learnt data or not.
            If it doesn't exist we move forward one character and push the unseen character to the heap
            with a smoothed probability 1/N (where N = number of elements in the distribution)
            """
            if num_observ == 0:
                heap.push(chartEntry("".join(utf8line[0]).encode('utf-8'),
                    start_pos=0,
                    end_pos=0,
                    log_prob=np.log2(Pw1("".join(utf8line[0]))),
                    back_ptr=None,
                    sort_acc_to=sort_acc_to))

            """
            Start filling the `chart` table iteratively.
            """


            while heap:
                head = heap.pop() # pop the item with highest log-probability
                utf8word = head.get_item('word').decode('utf-8')
                startindex = head.get_item('start_pos')
                endindex   = head.get_item('start_pos') + len(utf8word)-1

                if endindex in chart:
                    # get the previous entry
                    preventry = chart[endindex]

                    if head.get_item('log_prob') > preventry.get_item('log_prob'):
                        chart[endindex] = head
                    else:
                        continue

                else: # there was no previous entry
                    chart[endindex] = head

                num_observ = 0
                # move to the next element in the line
                sub_utf8line  = utf8line[startindex:]
                # print(sub_utf8line)
                for key in Pw2:
                    (u,w) = key.split(' ')
                    # print("WORD: " + head.get_item('word'))
                    search_word = key.replace(" ","")
                    if sub_utf8line.startswith(search_word):
                        num_observ += 1

                        # Computing new probability
                        # Computing p(u)p(w|u)
                        newp = np.log2(Pw1("".join(u)) * ( (Pw2("".join(key))/Pw1("".join(u))) ))

                        newentry = chartEntry("".join(w).encode('utf-8'),
                            start_pos=endindex + 1,
                            end_pos=startindex + len(key) -1,
                            log_prob=newp,
                            back_ptr=head,
                            sort_acc_to=sort_acc_to)


                        heap.push(newentry)


                """
                Check wether the pattern exist in our learn data or no
                If it doesn't exist we move for one character and push that character to the heap
                """
                if num_observ == 0 and len(sub_utf8line) > 1:
                    newentry = chartEntry("".join(utf8line[endindex+1]).encode('utf-8'),
                        start_pos=endindex+1,
                        end_pos=endindex+1,
                        # log_prob=np.log2(smoothed_prob) + head.get_item('log_prob'),
                        log_prob=np.log2(Pw1("".join(w))) + head.get_item('log_prob'),
                        back_ptr=head,
                        sort_acc_to=sort_acc_to)
                    heap.push(newentry)

            finalindex = len(utf8line)-1

            if finalindex in chart:
                finalentry = chart[finalindex]

                """
                Step 3:
                Backtracking and printing the output
                """
                ptr = finalentry.get_item('back_ptr')
                result = finalentry.get_item('word')
                while ptr:

                    # out_file.write(ptr.get_item('word') + ' ')

                    result = ptr.get_item('word') + ' ' + result
                    ptr = ptr.get_item('back_ptr')

                # out_file.write('\n'.encode('utf-8'))
                out_file.write(result+'\n')
                print(result)

            else:
                print(chart)
                print('Not Found!')

"""
Running the algorithem
    1) First fill the PW using count_1w file
    2) Run the baseline algorithem
"""
bigram(input_filename=opts.input)
