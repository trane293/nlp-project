# Assignment 3

# Group Name: GroupNLP

To run the alignment script, use the following command to generate the alignments:

`python2 align.py -d ../data -p europarl -e en -f de -i 5 -n 100000`

To run the alignment script which uses null words, use the following command to generate the alignments:

`python2 align_null_forward.py -d ../data -p europarl -e en -f de -i 5 -n 100000`

To run the alignment script which uses null words, use the following command to generate the alignments:

`python2 align_optimized.py -d ../data -p europarl -e en -f de -i 5 -n 100000`

Once the alignment file is generated, do the following to clip the file to only 1000 top sentences:

`head -t 1000 output_de.a > upload_de.a`
