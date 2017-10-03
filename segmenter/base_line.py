from __future__ import print_function
# import pandas as pd
# from pandas.io.parsers import read_csv
import heapq as heapq
import numpy as np
import pdb


class chartEntry:
    """
    This class implements the data structure "Entry" described in the baseline
    algorithm. The pseduocode for the description is:

    Entry(word, start-position, end-position, log-probability, back-pointer)

    We define the __lt__ and __eq__ operators to allow comparisons betwen two entries when
    trying to push them into the heap. This operator respects the sort_acc_to variable and
    provides boolean value accordingly.

    It supports operations based on two ideas - i) sorting by start_pos
                                                ii) sorting by log_prob.

    IMPORTANT NOTE: For Comparison, two objects can be compared based upon their start-pos
    or log-prob (depending upon the initial setting of sort_acc_to when defining the objects)
    using the normal comparison operators like < or >. HOWEVER, equals comparison SHOULD ONLY
    BE MADE USING isEquals() function, and NOT ==.


    EXAMPLE USAGE:
        p = chartEntry('Anmol', 0, 4, 0.2, -1, sort_acc_to='start_pos')
        print(p)
        p.get_item('log_prob')
        e1 = chartEntry('Anmol', 0, 4, 0.2, -1)
        e2 = chartEntry("Shreeashish", 5, 10, 0.5, 0)
        e3 = chartEntry('Amir Ali', 11, 16, 0.4, 1)
        p1 = chartEntry('Anmol', 0, 4, 0.2, -1, sort_acc_to='start_pos')
        p2 = chartEntry('Anmol', 0, 4, 0.2, -1, sort_acc_to='start_pos')
        p1.isEqual(p2) [TRUE]

    This work is a part of the Assignment 1 of CMPT 825 Natural Language Processing
    taught by Prof. Anoop Sarkar.

    AUTHOR: Anmol Sharma, GroupNLP
    INSTITUTION: Simon Fraser University
    """

    def __init__(self, word, start_pos, end_pos, log_prob, back_ptr, sort_acc_to='start_pos'):
        self.instance = {}
        self.instance['word'] = word
        self.instance['start_pos'] = start_pos
        self.instance['end_pos'] = end_pos
        self.instance['log_prob'] = log_prob
        self.instance['back_ptr'] = back_ptr
        self.__sort_type = sort_acc_to

    def __repr__(self):
        return "chartEntry({}, {}, {}, {}, {})".format(self.instance['word'], self.instance['start_pos'], \
                                                       self.instance['end_pos'], self.instance['log_prob'], \
                                                       self.instance['back_ptr'])

    def __lt__(self, other_obj):
        if self.__sort_type == 'start_pos':
            return (self.instance['start_pos'] < other_obj.instance['start_pos'])
        else:
            return (self.instance['log_prob'] < other_obj.instance['log_prob'])

    def __gt__(self, other_obj):
        if self.__sort_type == 'start_pos':
            return (self.instance['start_pos'] > other_obj.instance['start_pos'])
        else:
            return (self.instance['log_prob'] > other_obj.instance['log_prob'])

    def __eq__(self, other_obj):
        if self.__sort_type == 'start_pos':
            return (self.instance['start_pos'] == other_obj.instance['start_pos'])
        else:
            return (self.instance['log_prob'] == other_obj.instance['log_prob'])

    def isEqual(self, other_obj):
        for k in self.instance:
            if self.instance[k] == other_obj.instance[k]:
                continue
            else:
                return False
        return True

    def get_item(self, key):
        return self.instance[key] if key in self.instance else "Undefined Key"


class Heap:
    """
    A class wrapper for heapq datastructure implementation of python. Python's default heapq
    implementation requires a list as initialized heap, however it doesn't provide
    any safeguards against the fact that the underlying list may be changed by some function.

    To provide safeguard mechanism, this class wraps the push and pop functions of heapq.

    EXAMPLE USAGE:
        sort_acc_to='log_prob'
        p = chartEntry('Anmol', 0, 4, 0.2, -1, sort_acc_to)
        print(p)
        p.get_item('log_prob')
        e1 = chartEntry('Anmol', 0, 4, 0.6, -1, sort_acc_to)
        e2 = chartEntry("Shreeashish", 5, 10, 0.5, 0, sort_acc_to)
        e3 = chartEntry('Amir Ali', 11, 16, 0.1, 1, sort_acc_to)
        heap1 = Heap()
        heap1.push(e1)
        heap1.push(e2)
        heap1.push(e3)
        heap1.pop()

    This work is a part of the Assignment 1 of CMPT 825 Natural Language Processing
    taught by Prof. Anoop Sarkar.

    AUTHOR: Anmol Sharma, GroupNLP
    INSTITUTION: Simon Fraser University
    """

    def __init__(self, ls=None):
        self.__heap = ls if ls else []
        heapq.heapify(self.__heap)

    def push(self, item):
        heapq.heappush(self.__heap, item)

    def pop(self):
        return heapq.heappop(self.__heap)

    def __len__(self):
        return len(self.__heap)

    def __repr__(self):
        return "{}".format(self.__heap)


class Pdist(dict):
    "A probability distribution estimated from counts in datafile."

    def __init__(self, filename, sep='\t', N=None, missingfn=None):
        self.maxlen = 0
        for line in file(filename):
            (key, freq) = line.split(sep)
            try:
                utf8key = unicode(key, 'utf-8')
            except:
                raise ValueError("Unexpected error %s" % (sys.exc_info()[0]))
            self[utf8key] = self.get(utf8key, 0) + int(freq)
            self.maxlen = max(len(utf8key), self.maxlen)
        self.N = float(N or sum(self.itervalues()))
        self.missingfn = missingfn or (lambda k, N: 1. / N)

    def __call__(self, key):
        if key in self:
            return float(self[key]) / float(self.N)
        # else: return self.missingfn(key, self.N)
        elif len(key) == 1:
            return self.missingfn(key, self.N)
        else:
            return None


Pw = Pdist('data/count_1w.txt')

"""
Baseline function implementation
"""


def baseline_alg(input_filename='data/input', sort_acc_to='log_prob'):
    with open(input_filename) as f:

        heap = Heap()
        chart = dict()

        for line in f:
            utf8line = unicode(line.strip(), 'utf-8')
            # print(utf8line)
            # print(len(utf8line))

            # Step 1:
            # Initializing step
            # Finding each word that matches input at position 0
            for key in Pw:
                # if utf8line.find(key, 0) != -1:
                if utf8line.startswith(key):
                    entry = chartEntry("".join(key).encode('utf-8'), start_pos=0, end_pos=len(key), \
                               log_prob=np.log2(Pw("".join(key))), back_ptr=None, \
                               sort_acc_to=sort_acc_to)
                    # print(entry, '\n')
                    heap.push(entry)

            # Step 2:
            while heap:
                head = heap.pop()
                utf8word = head.get_item('word').decode('utf-8')
                endindex = head.get_item('start_pos') + len(utf8word)-1

                if endindex in chart:
                    preventry = chart[endindex]

                    if head.get_item('log_prob') > preventry.get_item('log_prob'):
                        chart[endindex] = head
                else:
                    chart[endindex] = head

                for key in Pw:
                    sub_utf8line = utf8line[endindex+1:]
                    if sub_utf8line.startswith(key):
                        heap.push(chartEntry("".join(key).encode('utf-8'), start_pos=endindex+1, end_pos=endindex+len(key), \
                                   log_prob=np.log2(Pw("".join(key))), back_ptr=head, \
                                   sort_acc_to=sort_acc_to))

            # print('Original: ', utf8line , '(len: ', len(utf8line),')')
            # print(len(chart))
            # print(chart)
            finalindex = len(utf8line)-1
            if finalindex in chart:
                finalentry = chart[finalindex]
                # print(finalentry)

                # Step 3:
                # Backtracking and printing the output
                ptr = finalentry.get_item('back_ptr')
                result = finalentry.get_item('word')
                while ptr:
                    result = ptr.get_item('word') + ' ' + result
                    ptr = ptr.get_item('back_ptr')

                print('Original: ',utf8line)
                print('Result  : ', result,'\n')

            else:
                print('Original: ',utf8line)
                print('Not Found!')
    # output = [i for i in utf8line]  # segmentation is one word per character in the input
    # print " ".join(output)


baseline_alg()
