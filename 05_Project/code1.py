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

# warnings.filterwarnings(action = 'ignore')
stanza.download('en')

def search_google(url):
    '''
    Fetch the response of the URL.

    Args: URL

    Returns: response/data of the url
    '''
    session = HTMLSession()
    response = session.get(url)
    # if response code != 200 give an error else return the response
    if response.status_code != 200:
        print("Error while fetching: ", response.status_code)
        return response
    else:
        return response

def scrape_google(search):
    '''
    Fetch all the website links after google search.

    Args: search keyword/ text to search on google
    
    Returns: links fetched after searching the keyword on Google
    '''
    search = urllib.parse.quote_plus(search)
    response = search_google("https://www.google.com/search?q=" + search)

    # Status code condition in above function have to handle it here only
    '''If response status !=200 -> handle error'''
    if response.status_code != 200: 
        links = []
    else:
        links = list(response.html.absolute_links)

    '''can remove google related websites here also'''
    return links,response.status_code

def algo(list_of_websites,acquirer, target):
    '''
    Filter out the links fetched and will keep only the useful one.
    
    Args: list of all websites fetched from google search
          acquirer and target company names.

    Returns: All the useful links to gather further detailed information about the merger.
    '''
    l = list_of_websites

    websites = []
    # First search for sec.gov websites 
    websites.extend([i for i in l if "sec.gov" in i])
    l_sec = len([s for s in l if "sec.gov" in s])

    # Second:- search for acquirer companies website
    # acquirer = remove last words as they are of no use
    websites.extend([s for s in l if acquirer.split()[0] in s])
    websites.extend([s for s in l if acquirer.split()[0].lower() in s])
    l_acq = len([s for s in l if acquirer in s])

    # get first two words -> combine them and search for link -> words in lower case
    acq = acquirer.split()[:2]
    websites.extend([s for s in l if ''.join(acq).lower() in s])

    # Third:- search for target companies website
    websites.extend([s for s in l if target.split()[0] in s])
    websites.extend([s for s in l if target.split()[0].lower() in s])

    # get first two words -> combine them and search for link -> words in lower case
    tar = target.split()[:2]
    websites.extend([s for s in l if ''.join(tar).lower() in s])

    # Fourth 
    # search for rest of the websites
    ''' Can use loop'''
    websites.extend([i for i in l if 'businesswire' in i])
    # print([s for s in l if 'buisnesswire' in s])
    websites.extend([s for s in l if 'prenewswire' in s])
    # print([s for s in l if 'prenewswire' in s])
    websites.extend([s for s in l if 'globenewswire' in s])
    # print([s for s in l if 'globenewswire' in s])
    websites.extend([s for s in l if 'nytimes' in s])
    # print([s for s in l if 'nytimes' in s])
    websites.extend([i for i in l if 'crunchbase' in i])
    # print([s for s in l if 'crunchbase' in s])
    websites.extend([i for i in l if 'bizjournal' in i]) 
    # print([s for s in l if 'bizjournal' in s])
    websites.extend([i for i in l if 'mergr' in i]) 
    # print([s for s in l if 'mergr' in s])
    websites.extend([i for i in l if 'finance.yahoo' in i]) 
    # print([s for s in l if 'yahoo finance' in s])
    '''canb add CNN, techcrunch, bloomberg'''
    return websites

def count_categories():
    df = pd.read_excel('data/testing/mergers.xlsx')
    print(df.groupby(['Target_Public_Status']).size)

def filter(all_list,removal):
    '''
    Remove google related websites like google support/maps/advanced google search

    Args: actual list that contains all data items
          all sub-Strings that if found in any list data item -> data item to be removed
    '''
    for url in all_list:
        for start in removal:
            if url.startswith(start):
                try:
                    all_list.remove(url)
                except:
                    continue
            else:
                continue
    return all_list

def list_file(name, list):
    with open(name+'.txt', 'w') as f:
        for line in set(list):
            f.write(f"{line}\n")
    print('File Created: ',name,'.txt')

def get_jobs():
    '''
    This function get list of all job position on website
    Paramter: nothing
    Return: list with all the job-titles
    '''
    url = 'https://www.joblist.com/b/all-jobs'
    reqs = requests.get(url)
    soup = BeautifulSoup(reqs.text, 'html.parser')

    # ff = pd.DataFrame()
    lst = []
    count = 0
    for link in soup.find_all('a'):
        # print(link.get('href'))
        # ff.at[count,'Jobs'] = link.get('href')
        lst.append(link.get('href'))
        count += 1
    
    links = []
    for i in lst:
        if i.startswith('/b'):
            # remove starting '/b/' tag and also replace '-' with white space
            item = i[3: ].replace('-',' ')
            links.append(item)
        else:
            continue
    
    # save file
    with open("data/job_lists.txt", "w") as output:
        output.write(str(links))
    
    print('File stored at: ','data/job_lists.txt')
    return links

def read_job():
    '''
    This function reads all the job lists title from the text file and convert them to list format
    Returns: all job titles in as list items
    '''
    f = open("data/job_lists.txt", "r")
    job_list = ast.literal_eval(f.read())
    return job_list

def pre(combined):
    '''
    Apply pre processing tasks on textual data
    Parameter: textual data in string format
    Return: textual data in string format after pre processing'''
    combined = combined.lower()
    combined = remove_stopwords(combined)
    ## word2vec consider sentences so we can not remove punctuations
    combined = strip_punctuation(combined)
    # print(filtered_sentence[:120])
    combined = strip_multiple_whitespaces(combined)
    # print(filtered_sentence[:120])
    combined = strip_numeric(combined)
    # print(filtered_sentence[:100])
    combined = strip_short(combined)
    # print(filtered_sentence[:100])
    return combined
# ---------------------------------------------------------------- Algorithm 
def step1():
    '''
    Collect website links
    '''
    df = pd.read_excel('data/testing/mergers.xlsx')

    # df = pd.read_excel('data_2014.xlsx')

    removal = ['https://www.google.','https://www.google.com/search?q','https://google.','https://webcache.googleusercontent.', 'https://support.google.','http://webcache.googleusercontent.', 'https://maps.google.']

    # end_df =pd.DataFrame(columns=['Acquirer','Target','Links','Year'])
    df['Links'] = ''
    # Read the database and execute the above functions 
    # To fetch useful websites

    for i in trange(df.shape[0]):
        if i%100 and i!=0:
            print(i)
            # time.sleep(120)

        ''' Remove Inc or something like that words'''
        ret = scrape_google(df.Acquiror_Full_Name[i] + " acquire "+ df.Target_Name[i])
        
        links = ret[0] 
        status = ret[1]
        print('\nStatus',status,'\n')
        if status!=200:
            df.to_csv('data1.csv')
        else:
            result = algo(links,df.Acquiror_Full_Name[i], df.Target_Name[i])            
            # Remove unnecessary links
            # Remove empty lists -. if Any
            result = [item for item in result if item != []]

            # Have to repeat below steps as there are duplicates in list
            result = filter(result,removal)
            print(result)
            # result = filter(result,removal)
            # result = filter(result,removal)
            # result = filter(result,removal)

            # Everything in a DatFrame
            # end_df.loc[i,'Acquirer'] = df.Acquiror_Full_Name[i]
            # end_df.loc[i,'Target'] = df.Target_Name[i]
            # end_df.loc[i,'Links'] = str(result)
            df.loc[i,'Links'] = str(result)
    df.to_csv('data/data_fetchedLinks.csv')

def step1_2014():
    '''
    Collect website links for 2014 year only
    '''
    # df = pd.read_excel('data/testing/mergers.xlsx')

    df = pd.read_excel('data/training/data_2014.xlsx')

    removal = ['https://www.google.','https://www.google.com/search?q','https://google.','https://webcache.googleusercontent.', 'https://support.google.','http://webcache.googleusercontent.', 'https://maps.google.']

    df['Links'] = ''
    # Read the database and execute the above functions 
    # To fetch useful websites
    ''' Run this for loop from 0-200 and 200-400 and 400-600 and 600-last: this won't give 429 error '''
    ''' or give it a sleep after 100 '''
    for i in trange(df.shape[0]):
        if i%200 and i!=0:
            print(i)
            time.sleep(120)

        ''' Remove Inc or something like that words'''
        ret = scrape_google(df.Acquiror_Full_Name[i] + " acquire "+ df.Target_Name[i])
        
        links = ret[0] 
        status = ret[1]
        print('\nStatus',status,'\n')
        if status!=200:
            df.to_csv('data1.csv')
        else:
            result = algo(links,df.Acquiror_Full_Name[i], df.Target_Name[i])            
            # Remove unnecessary links
            # Remove empty lists -. if Any
            result = [item for item in result if item != []]

            # Have to repeat below steps as there are duplicates in list
            result = filter(result,removal)
            # Everything in a DatFrame

            df.loc[i,'Links'] = str(result)
    df.to_csv('data/data_fetchedLinks2014.csv')
    return df

def step2():
    '''
    Collect websites data 
    '''
    df = pd.read_csv('data/testing/data_fetchedLinks.csv')
    
    ## Need to drop all those rows only where LInk is na - isntead of droping can use try and except while fetching
    df = df[df['Links'].notna()] # drop all rows that do not have any links data
    df.reset_index(inplace = True) # need to do this after droping nan values

    # Store important columns in seperate file after reseting the index
    df1 = df.loc[:,['Date_Announced', 'Target_Public_Status','Acquiror_Full_Name', 'Target_Name']]
    # df1 = df.loc[:,['Date_Announced', 'Target_Public_Status','Acquiror_Full_Name', 'And', 'Target_Name']]

    df1['FileName'] = ''

    headers = requests.utils.default_headers()
    headers.update({'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.93 Safari/537.36'})

    # # Next step is to remove duplicates
    # for k in range(df.shape[0]):
    #     lst = ast.literal_eval(df.Links[k])
    #     df.loc[k,'Links'] = [*set(lst)]
    # # can use set(list)

    ''' pass the link and we will get the paragraph content'''
    for i in trange(df.shape[0]):
        if i%200 == 0 and i!=0:
            time.sleep(60)
        websites = ast.literal_eval(df.Links[i])
        websites = [*set(websites)]
        
        # remove all google related websites
        removal = ['https://www.google.','https://www.google.com/search?q','https://google.','https://webcache.googleusercontent.', 'https://support.google.','http://webcache.googleusercontent.', 'https://maps.google.']
        websites = filter(websites,removal)

        headers = requests.utils.default_headers()
        headers.update({'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.93 Safari/537.36'})
        '''storing data in json is good'''
        dictionary = {}
        for link in websites:
           
            ''' Handle this pdf and sec gov isssue''' 
            if link[-4:] == '.pdf':
                print(i,' ',link)
            
            # if link is of sec website : use following to fetch data
            # soup1 = BeautifulSoup(html_page, 'html5lib')
 
            else:
                try:
                    resp = requests.get(link,headers=headers) 

                    if resp.status_code != 200:
                        print('Error ', resp.status_code,' Link ',link,' Acquiror Name ',df.Acquiror_Full_Name[i])
                        dictionary[link] = 'Error_'+str(resp.status_code)
                        continue

                    else:
                        html_page = resp.content
                        soup = BeautifulSoup(html_page, 'html.parser')
                        content = soup.findAll('p')
                        value = ''
                        for txt in content:
                            value += txt.getText()
                        dictionary[link] = value 

                except: # if SSL error then run the following
                    try:
                        resp = requests.get(link,headers=headers, verify =False) 

                        if resp.status_code != 200:
                            print('Error ', resp.status_code,' Link ',link,' Acquiror Name ',df.Acquiror_Full_Name[i])
                            dictionary[link] = 'Error_'+str(resp.status_code)
                            continue

                        else:
                            html_page = resp.content
                            soup = BeautifulSoup(html_page, 'html.parser')
                            content = soup.findAll('p')
                            value = ''
                            for txt in content:
                                value += txt.getText()
                            dictionary[link] = value 
                        print(i,' error')
                    
                    # If code gets inside the except -> some other error
                    except:
                        print(i,'Eror but not SSL Certificate')

        json_object = json.dumps(dictionary)

        # Writing to sample.json
        # str(df.Target_Name[i].split()[0]).replace('/','-') # replace '/' in the name with '-'
        filename = str(df.Target_Name[i].split()[0]).replace('/','-')
        ''' File name in which paragraph tag are stored : Target company name and their index in thre dataset file'''
        filename = filename.replace('.','-')
 
        with open('data/fetched2014/'+filename+' '+str(i)+".json", "w") as outfile:
            outfile.write(json_object)

        df1.loc[i,'FileName'] = filename
    df1.to_csv('metadata_jsons.csv')

def step2_2014():
    '''
    Collect websites data for 2014 year only
    '''
    df = pd.read_csv('data/training/data_2014_fetched.csv')

    ## Need to drop all those rows only where LInk is na - isntead of droping can use try and except while fetching
    df = df[df['Links'].notna()] # drop all rows that do not have any links data
    df.reset_index(inplace = True) # need to do this after droping nan values

    df1 = df.loc[:,['Date_Announced', 'Target_Public_Status','Acquiror_Full_Name', 'And', 'Target_Name']]
    df1['FileName'] = ''
    # # Next step is to remove duplicates
    # for k in range(df.shape[0]):
    #     lst = ast.literal_eval(df.Links[k])
    #     df.loc[k,'Links'] = [*set(lst)]
    # # can use set(list)
    headers = requests.utils.default_headers()
    headers.update({'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.93 Safari/537.36'})

    ''' pass the link and we will get the paragraph content'''
    for i in trange(df.shape[0]):
        if i%200 == 0 and i!=0:
            time.sleep(60)
        websites = ast.literal_eval(df.Links[i])
        websites = [*set(websites)]
        
        # remove all google related websites
        removal = ['https://www.google.','https://www.google.com/search?q','https://google.','https://webcache.googleusercontent.', 'https://support.google.','http://webcache.googleusercontent.', 'https://maps.google.']
        websites = filter(websites,removal)

        '''storing data in json is good'''
        dictionary = {}
        for link in websites:
           
            ''' Handle this pdf and sec gov isssue''' 
            if link[-4:] == '.pdf':
                print(i,' ',link)
            
            # if link is of sec website-> soup = BeautifulSoup(html_page, 'html5lib')
            else:
            # if last of websites not htm or html move ahead
                try:
                    resp = requests.get(link,headers=headers) 

                    if resp.status_code != 200:
                        print('Error ', resp.status_code,' Link ',link,' Acquiror Name ',df.Acquiror_Full_Name[i])
                        dictionary[link] = 'Error_'+str(resp.status_code)
                        continue

                    else:
                        html_page = resp.content
                        soup = BeautifulSoup(html_page, 'html.parser')
                        content = soup.findAll('p')
                        value = ''
                        for txt in content:
                            value += txt.getText()
                        dictionary[link] = value 

                except: # if SSL error then run the following
                    try:
                        resp = requests.get(link,headers=headers, verify =False) 

                        if resp.status_code != 200:
                            print('Error ', resp.status_code,' Link ',link,' Acquiror Name ',df.Acquiror_Full_Name[i])
                            dictionary[link] = 'Error_'+str(resp.status_code)
                            continue

                        else:
                            html_page = resp.content
                            soup = BeautifulSoup(html_page, 'html.parser')
                            # soup1 = BeautifulSoup(html_page, 'html5lib')
                            content = soup.findAll('p')
                            value = ''
                            for txt in content:
                                value += txt.getText()
                            dictionary[link] = value 
                        print(i,' error')
                    
                    # If code gets inside the except -> some other error
                    except:
                        print(i,'Error but not SSL Certificate')

        json_object = json.dumps(dictionary)

        # Writing to sample.json
        # str(df.Target_Name[i].split()[0]).replace('/','-') # replace '/' in the name with '-'
        filename = str(df.Target_Name[i].split()[0]).replace('/','-')
        ''' File name in which paragraph tag are stored : Target company name and their index in thre dataset file'''
        filename = filename.replace('.','-')

        with open('data/fetched2014/'+filename+str(i)+".json", "w") as outfile:
            outfile.write(json_object)

        df1.loc[i,'FileName'] = filename+str(i)+".json"
    df1.to_csv('data/metadata_jsons2014.csv')

def step3_2014():
    ''' 
    named Entity Recognition and store all the fetched Person Names, Job Names and Organization Names as extra columns in metadata
    '''

    # use metadata_csv file
    df1 = pd.read_csv('data/metadata_jsons2014.csv')
    df1 = df1.dropna()
    files = df1.FileName.to_list()

    # get all values to do NER and all
    df1['Persons'] = ''
    df1['Organization'] = ''
    df1['Jobs'] = ''
    # Go for each json file fetched
    c = 0

    stanza.download('en')
    nlp = stanza.Pipeline(lang='en', processors='tokenize,ner', download_method = None, use_gpu=True, verbose = 0)
    for file in tqdm(files):
        print(c,file)

        # Check Desktop Service Store file present or not
        if file != '.DS_Store':
            f = open('data/fetched2014/'+file)
            dictionary = json.load(f)
            
            # Stop Word removal and Remove Punctuations 
            value = list(dictionary.values())
            
            '''Combining the data good or not'''
            # Combine all textual data 
            combined = ''
            for i in value:
                combined += i
            combined = pre(combined)

            '''Extract all Person names from the dataset'''
            # only need tokenization lemmatiuzation and NER
            # p = []
            org = []
            prsn = []

            doc = nlp(combined) # executing this takes time

            for sentence in doc.sentences:
                for word in sentence.ents:
                    if word.type == 'ORG':
                        # print('ORG', word.text, ' ', word.type)
                        org.append(word.text)
                    elif word.type == 'PERSON':
                        # print('PERSON', word.text, ' ', word.type)
                        prsn.append(word.text)
            df1.loc[c,'Persons'] = str(list(set(prsn)))
            c += 1
            # p.append(prsn)
            
            ''' Get organizations name'''
            # already in the dataset
            df1.loc[c,'Organization'] = str(list(set(org)))

            ''' Get job lists'''              
            # jobs = read_job()
            # For now read subset of the job titles manually selected else jobs variable length is 5k and taking time to process
            dd = pd.read_csv('data/jobLists_subset.csv')
            jobs = dd.Jobs.to_list()
            job_titles = []
            for i in jobs:
                if i in combined:
                    job_titles.append(i)

            df1.loc[c,'Jobs'] = str(job_titles)

        else:
            continue
        df1.to_csv('ner_fetched2014.csv')

    return prsn, org, job_titles

def word2vec(file_name):
    '''
    This function applied word2vec model on the data and return the model
    Paramter: File name (json file of fetched textual data)
    Returns: trained model 
    '''
    # file = 'Stream2.json'
    file = filename
    f = open('/data/fetched2014/'+file)
    dictionary = json.load(f)
            
    # Stop Word removal and Remove Punctuations 
    value = list(dictionary.values())

    '''Combining the data good or not'''
    # Combine all textual data 
    combined = ''
    for i in value:
        combined += i

    ''' Split sentences into sepretate list and apply pre processing'''
    # a = combined.split('\n')
    sentences = combined.split('. ')

    processed = []
    for line in sentences:
        processed.append(pre(line))

    # remove empty list items 
    while("" in processed):
        processed.remove("")

    tokenized_sents = [word_tokenize(i) for i in processed]
    
    w2v_model = Word2Vec(min_count=2, window=10, size=524, sample=6e-5, alpha=0.03, sg=1, min_alpha=0.0007, negative=20)

    w2v_model.build_vocab(tokenized_sents, progress_per=10000)
    w2v_model.train(tokenized_sents, total_examples=w2v_model.corpus_count, epochs=50, report_delay=1)
    # w2v_model.wv.most_similar('chief')
    return w2v_model
# ---------------------------------------------------------------- Main- Call functions

def function_calls():
    step1_df = step1_2014()
    step2_df = step2_2014()

'''References: '''
# https://tedboy.github.io/bs4_doc/9_specifying_the_parser.html
# https://github.com/roomylee/awesome-relation-extraction 
# https://radimrehurek.com/gensim_3.8.3/parsing/preprocessing.html#gensim.parsing.preprocessing.strip_punctuation 