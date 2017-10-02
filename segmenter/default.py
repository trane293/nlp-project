
import sys, codecs, optparse, os
from heapq import heappush, heappop
from collections import namedtuple

# import sys
sys.setrecursionlimit(3500)

optparser = optparse.OptionParser()
optparser.add_option("-c", "--unigramcounts", dest='counts1w', default=os.path.join('data', 'count_1w.txt'), help="unigram counts")
optparser.add_option("-b", "--bigramcounts", dest='counts2w', default=os.path.join('data', 'count_2w.txt'), help="bigram counts")
optparser.add_option("-i", "--inputfile", dest="input", default=os.path.join('data', 'input'), help="input file to segment")
(opts, _) = optparser.parse_args()

# Reading the counts1w and make a dictionary of words with their frequencies
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
        self.missingfn = missingfn or (lambda k, N: 1./N)

    def __call__(self, key):
        if key in self: return float(self[key])/float(self.N)
        #else: return self.missingfn(key, self.N)
        elif len(key) == 1: return self.missingfn(key, self.N)
        else: return None

# the default segmenter does not use any probabilities, but you could ...
Pw  = Pdist(opts.counts1w)

old = sys.stdout
sys.stdout = codecs.lookup('utf-8')[-1](sys.stdout)
# ignoring the dictionary provided in opts.counts
with open(opts.input) as f:

    # data = list()
    Entry = namedtuple('Entry', 'start word logp back')
    PHeap = []
    chart = dict()

    for line in f:
        utf8line = unicode(line.strip(), 'utf-8')
        print utf8line

        # Step 1:
        # Initializing step
        # Finding each word that matches input at position 0
        for key in Pw:
            if utf8line.find(key, 0) != -1 :
                heappush(PHeap, Entry(word=key, start=0, logp=Pw(key), back=None))

        # Step 2:
        while PHeap:
            head = heappop(PHeap)
            print len(PHeap)

            endindex = head.start + len(head.word)

            if endindex in chart:
                preventry = chart[endindex]

                if head.logp > preventry.logp:
                    chart[endindex] = head
            else:
                chart[endindex] = head

            for key in Pw:
                if utf8line.find(key, endindex) != -1 :
                    heappush(PHeap, Entry(word=key, start=endindex, logp=Pw(key), back=head))


    finalindex = len(utf8line)
    finalentry = chart[finalindex]
    print finalentry

        # output = [i for i in utf8line]  # segmentation is one word per character in the input
        # print " ".join(output)

sys.stdout = old
