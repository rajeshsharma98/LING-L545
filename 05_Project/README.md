# Project  : Delegates Information Extraction 
Merger and Acquisition (M & A) activity is exponentially increasing with time and has increased from $1400 billion to $2600 billion solely in the USA in the past 15 years only. Information of M&A deals between two organizations is readily available on the internet, but to get further insights into the data, one must go further into the press release or blogs to get information about the delegations involved. In this work, we consider extracting the delegate’s information like name and job position of the people that were involved in the process using various NLP techniques.



## Files:

Code1.py: main code file  
metrics.py: code file to check NER results  

data: all data is inside this folder  
- data/metadata.csv: are files that contains company name and the file name in which there fetched textual data is stored  
- data/ner_fetched2014.csv: update of metadata.csv -> after pre processing : contains columns with person names, organziation names, and job titles  
- data/job_lists.txt: all job titles fethced from the website  
- data/jobLists_subset.csv: all job titles of buisness domain only. this file is subset of above file(job_lists.txt)    

data/fetched2014  : this folder conatins all json(fethced textual data for each deal) files (step3())   

data/training : initial data  
- data_2014: all deals of 2014  
- data_2014_fetched: all deals of 2014 with website links (step1())     
