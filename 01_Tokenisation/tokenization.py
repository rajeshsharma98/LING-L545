import sys
from nltk.corpus import words

line = sys.stdin.readline()
# Read File
dictionary_file = open(sys.argv[1], "r")
dictionary_l = dictionary_file.read()
# File to list -> each token as list item
dictionary = dictionary_l.split("\n")

while line != '':
    i= 0
    while i < len(line):
        # create variable to store maxLength
        maxLength = ' '
        for j in range(len(line), i,-1):
            word = line[i:j]
            # compare maxLength and word
            if word in dictionary and len(word) > len(maxLength):
                maxLength = word
        i += len(maxLength)
        if len(maxLength.replace(' ', ''))<1:
            continue
        else:
            sys.stdout.write(maxLength+'\n')# sys out
    line = sys.stdin.readline()

# Reference: 
# 1. https://medium.com/@anshul16/maximum-matching-word-segmentation-algorithm-python-code-3444fe4bd6f9 
# 2. Section 3.9.1 in Jurafsky and Martin (2nd Edition)