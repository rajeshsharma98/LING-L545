import pandas as pd
import urllib.request
import requests
import urllib
import pandas as pd
import time
import ast
import json
import spacy
import stanza
import ast
import os 
import numpy as np
import nltk

from requests_html import HTML
from requests_html import HTMLSession
from tqdm import trange 
from bs4 import BeautifulSoup
from email.headerregistry import ContentTransferEncodingHeader
from pyexpat import model
from tqdm import tqdm
from pyexpat import model
from gensim.parsing.preprocessing import remove_stopwords
from gensim.parsing.preprocessing import strip_punctuation
from gensim.parsing.preprocessing import strip_numeric
from gensim.parsing.preprocessing import strip_multiple_whitespaces
from gensim.parsing.preprocessing import strip_short
from nltk.tokenize import sent_tokenize, word_tokenize
from gensim.models import Word2Vec

nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')
nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('treebank')

# -------- # -------- # -------- # -------- # -------- # -------- #

df1 = pd.read_csv('data/ner_fetched2014.csv')
# Check if any empty list in Persons
ff = df1[df1.Persons == '[]']
print('empty fethced Person list',ff.Target_Public_Status.value_counts())

df = pd.read_csv('data/training/data_2014_fetched.csv')

df = df.loc[:,['Acquiror_Full_Name','Target_Name','Acq CEO', 'Target CEO','Source','Source Link','Links']]

# -------- # -------- # -------- # -------- # -------- # -------- #
''' Drop all rows where Acq CEO = nan'''
c  = 0
d = 0
e = 0
for i in range(df2.shape[0]):
  # list of fetched names
  res = ast.literal_eval(df2.Persons[i])

  # if NER person name extraction not good
  # example: 
  j = []
  for z in res:
    # to remove dot after middle name if present
    z = strip_punctuation(z)

    # if three words in name present
    if len(z.split(' ')) == 3:
      # if middle name present then length also 3 but middle name is one word mostly A. or Al so length of middle name is fixed to 2
      if len(z.split(' ')[1]) > 2:      
        j.append(z.rsplit(' ', 1)[0])
      # else append the original text
      else:
        j.append(z)
    elif len(z.split(' ')) >3:
        # if name is  'Robert G Isaman Stanadyne'
        j.append(z.rsplit(' ', 1)[0])
        # include condition to remove first text when name is 'Robert G Isaman Stanadyne' -> if length of thrid word is of middle name length
    else:
      j.append(z)
  res = j

  # if item is string 
  if isinstance(df2['Acq CEO'][i],str):

    '''ss is the manually labeled result'''
    # strip whitespace at end or first
    ss = df2['Acq CEO'][i].strip()    
    orig = strip_multiple_whitespaces(strip_punctuation(ss))

    if  orig in res:
      # removing short words might remove m,iddle name that must be effective
      c+=1

    # remove middle name from fetched data if middle name present
    # to resolve problem of index 36: Amin Khoury and Amin J Khoury
    elif ss in [strip_short(i) for i in res]:
      c+=1
    # remove middle name from original data and then compare it with fetched data
    elif strip_short(ss) in res:
      c+=1
    
    # else Person name is string but not fetched using algorithm
    else:
      # if only first name or last name is fetched: example fethced 'Wallace' whereas original text has 'Timothy R Wallace'
      # check if splitting the name works
      ss_split = ss.split(' ')
      flag = 0
      for x in ss_split:
        if x in res:
          c+=1
          flag = 1 
          break
        else:
          continue
      # if after running above loop there is no match
      if flag == 0:
        # print(df2['Acq CEO'][i],i)
        e = e+1
  else:
    # these are the cases when instance in nan
    # that is in original data there is no labeling 
    d+=1

print('Tcq CEO:', ' | Correctly identified: ',c,' | Not identified: ',e)


# -------- # -------- # -------- # -------- # -------- #
''' Acq CEO'''
c  = 0
d = 0
e = 0
for i in range(df2.shape[0]):
  # list of fetched names
  res = ast.literal_eval(df2.Persons[i])

  # if NER person name extraction not good
  # example: 
  j = []
  for z in res:
    # to remove dot after middle name if present
    z = strip_punctuation(z)

    # if three words in name present
    if len(z.split(' ')) == 3:
      # if middle name present then length also 3 but middle name is one word mostly
      if len(z.split(' ')[1]) > 2:      
        j.append(z.rsplit(' ', 1)[0])
      # else append the original text
      else:
        j.append(z)
    elif len(z.split(' ')) >3:
        # if name is  'Robert G Isaman Stanadyne'
        j.append(z.rsplit(' ', 1)[0])
        # include condition to remove first text when name is 'Robert G Isaman Stanadyne' -> if length of thrid word is of middle name length

    else:
      j.append(z)
  res = j

  # if item is string 
  if isinstance(df2['Target CEO'][i],str):

    '''ss is the manually labeled result'''
    # strip whitespace at end or first
    ss = df2['Target CEO'][i].strip()    
    orig = strip_multiple_whitespaces(strip_punctuation(ss))

    if  orig in res:
      # removing short words might remove m,iddle name that must be effective
      c+=1

    # remove middle name from fetched data if middle name present
    # to resolve problem of index 36: Amin Khoury and Amin J Khoury
    elif ss in [strip_short(i) for i in res]:
      c+=1
    # remove middle name from original data and then compare it with fetched data
    elif strip_short(ss) in res:
      c+=1
    
    # else Person name is string but not fetched using algorithm
    else:
      # if only first name or last name is fetched: example fethced 'Wallace' whereas original text has 'Timothy R Wallace'
      # check if splitting the name works
      ss_split = ss.split(' ')
      flag = 0
      for x in ss_split:
        if x in res:
          c+=1
          flag = 1 
          break
        else:
          continue
      # if after running above loop there is no match
      if flag == 0:
        print(df2['Target CEO'][i],i)
        print(orig)
        print(res)
        e = e+1
  else:
    # these are the cases when instance in nan
    # that is in original data there is no labeling 
    d+=1

print('Target CEO:', ' | Correctly identified: ',c,' | Not identified: ',e)

# -------- # -------- # -------- # -------- # -------- #

c = 0
for i in range(df2.shape[0]):
  # list of fetched names
  res = ast.literal_eval(df2.Jobs[i])
  if 'CEO' in res:
    c+=1
  elif 'Chief Executive Officer' in res:
    c+=1
  elif 'Executive' in res:
    c+=1
  else:
    print(res)
print('Correctly Identified job titles: ',c)