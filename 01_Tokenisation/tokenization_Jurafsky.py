''' Algorithm from Section 3.9.1 in Jurafsky and Martin (2nd Edition)'''
# function MAXMATCH(string, dictionary) returns list of tokens T
# if string is empty
    # return empty list
# for i length(sentence) downto 1
    # firstword = first i chars of sentence
    # remainder = rest of sentence
    # if InDictionary(firstword, dictionary)
        # return list(firstword, MaxMatch(remainder,dictionary) )

import sys

line = sys.stdin.readline()

# Read File
dictionary_file = open(sys.argv[1], "r")
dictionary_l = dictionary_file.read()

# File to list -> each token as list item
dictionary = dictionary_l.split("\n")

def maxMatch(line, dictionary):
    if line == '': # if string is empty
        return [] # return empty list
    # for i in range(len(line),1,-1) : this is not considerinf last character   
    for i in range(len(line),0,-1): # for i length(sentence) downto 1
        firstword = line[:i] # firstword = first i chars of sentence
        reaminder = line[i:] # remainder = rest of sentence
        if firstword in dictionary: # if InDictionary(firstword, dictionary)
            print(firstword)
            return maxMatch(reaminder,dictionary) # return list(firstword, MaxMatch(remainder,dictionary) )
    line = sys.stdin.readline()

while line != '':
    # check the line call the maxMatch function
    maxMatch(line,dictionary)
    # read line
    line = sys.stdin.readline()


'''Referecens: Section 3.9.1 in Jurafsky and Martin (2nd Edition) Figure 2.13'''
