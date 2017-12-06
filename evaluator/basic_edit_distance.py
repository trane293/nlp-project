from nltk.stem.snowball import SnowballStemmer
from nltk.metrics.distance import edit_distance



def sentences():
        with open('data/hyp1-hyp2-ref') as f:
            for pair in f:
                yield [sentence.strip().split() for sentence in pair.split(' ||| ')]

stemmer = SnowballStemmer('english')

for h1, h2, r in sentences():

    h1_string = ' '.join(str(word) for word in h1)
    h2_string = ' '.join(str(word) for word in h2)
    r_string = ' '.join(str(word) for word in r)

    # h1_string = stemmer.stem(h1_string)
    #  h2_string= stemmer.stem(h2_string)
    # r_string = stemmer.stem(r_string)


    if edit_distance(r_string,h1_string) < edit_distance(r_string,h2_string):
        print(1)
    elif edit_distance(r_string,h2_string) < edit_distance(r_string,h1_string):
        print(-1)
    else:
        print(0)
