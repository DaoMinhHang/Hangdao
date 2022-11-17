# -*- coding: utf-8 -*-
"""
Created on Sun Jan 16 19:52:03 2022

@author: Dell
"""

import pandas as pd 
from nltk.corpus import stopwords
from nltk.tokenize import wordpunct_tokenize, sent_tokenize, word_tokenize
import nltk
import csv


import gensim
import numpy
from gensim.utils import simple_preprocess
def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations
import pandas as pd 
from nltk.corpus import stopwords
from nltk.tokenize import wordpunct_tokenize, sent_tokenize, word_tokenize
import nltk

from nltk.corpus import stopwords

data=pd.read_csv('data job posts.csv')

data_preprocessed= data.drop(['date', 'jobpost','JobRequirment', 'Company',	'AnnouncementCode',	'Term',	'Eligibility', 'Audience',	'StartDate', 'Duration', 'Location', 'Salary',	'ApplicationP',	'OpeningDate',	'JobDescription','Deadline',	'Notes'	,'AboutC',	'Attach',	'Month'
], axis=1)
IT_data=data_preprocessed[data_preprocessed.IT ==True]
IT_data=IT_data.dropna()
IT_data['Year'].value_counts()

IT_data_2005_2011 = IT_data[IT_data['Year'].isin([2005, 2006, 2007, 2008, 2009, 2010, 2011]) ]
IT_data_2012_2015 = IT_data[IT_data['Year'].isin([2012, 2013, 2014, 2015]) ]
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()


qualifications_2012_15=IT_data_2012_2015['RequiredQual']
qualifications_2012_15=qualifications_2012_15.tolist()
qualifications_2005_11=IT_data_2005_2011['RequiredQual']
qualifications_2005_11=qualifications_2005_11.tolist()


data_words = list(sent_to_words(qualifications_2005_11))
from nltk.corpus import stopwords
stop_words=stopwords.words('english')
stop_words.extend(['candidates','implementation','qualification', 'qualifications', 'education'
                   'understanding', 'tasks','task','responsibility', 'understand', 'willingness','requirement', 'requirements','years', 
                   'work','year', 'knowledge','computer', 'must', 'skill', 'skills', 'experience', 'background', 'members', 'member',
                   'candidate', 'work', 'education', 'should', 'ability', 'languages', 'development', 'plus',
                   'application','technology', 'degree', 'problem', 'field', 'science', 'environment', 'project', 'management',
                   'quality', 'tool', 'personality', 'familiarity', 'strong', 'remuneration', 'note', 'notes', 'competency', 'organization', 'organisation',
                   'language', 'proficiency', 'advantage', 'solving', 'program', 'application', 'applications', 'professionalism', 'competency', 'person', 'people',
                   'passion','passionate','understanding','expertise'])
pos_tags_qual_2005_11=[]
for q in qualifications_2005_11:
    q_tokenized_2005_11= word_tokenize(q)
    pos_tag_qual_2005_11=nltk.pos_tag(q_tokenized_2005_11)
    pos_tags_qual_2005_11.append(pos_tag_qual_2005_11)

content_list_2005_11=[]
for pos_pairs in pos_tags_qual_2005_11:
    qual_content_2005_11=[]
    for p in pos_pairs:
        if p[1]=='NNP' or p[1]=='NN' or p[1]== 'NNS' or p[1]=='NNPS':
            qual_content_2005_11.append(p[0])
    content_list_2005_11.append(qual_content_2005_11)
    
removestopword_2005_11_list=[]
for l in content_list_2005_11:
    removestopword_2005_11=[]
    for w in l:
        w=w.lower()
        if w not in stop_words:
            removestopword_2005_11.append(w)
    removestopword_2005_11_list.append(removestopword_2005_11)
            

   
# from nltk.stem.porter import *

content_list_filtered_2005_11=[]
for l in removestopword_2005_11_list:
    content_filtered_2005_11=[]
    for w in l:

        w=w.lower()
    # w=lemmatizer.lemmatize(w)
        w=lemmatizer.lemmatize(w)
        content_filtered_2005_11.append(w)
    content_list_filtered_2005_11.append(content_filtered_2005_11)
            
converted_list_2005_11=[]
for l in content_list_filtered_2005_11:
    joined_l=' '.join(l)
    converted_list_2005_11.append(joined_l)

filtered_2005_11=[]
for sent in converted_list_2005_11:
    nums='0123456789'
    puncs='''~`!@$%^&*()-_=:;<>,.?\|'"'''
    sent_filtered=''
    for c in sent:
        if c=='/':
            c=' '
        if c in nums:
            c=''
        if c in puncs:
            c=''
        sent_filtered+=c
    filtered_2005_11.append(sent_filtered)
data_lemmatized =[]
for i in filtered_2005_11:
  i=i.split()
  data_lemmatized.append(i)


import gensim.corpora as corpora
# Create Dictionary

id2word = corpora.Dictionary(data_lemmatized)
id2word.filter_extremes(no_below=3, no_above=0.5)

# Create Corpus
texts = data_lemmatized
# Term Document Frequency
corpus = [id2word.doc2bow(text) for text in texts]

# lda_model = gensim.models.LdaMulticore(corpus=corpus,
#                                        id2word=id2word,
#                                        num_topics=10, 
#                                        random_state=100,
#                                        chunksize=100,
#                                        passes=10,
#                                        per_word_topics=True)

# from pprint import pprint
# # Print the Keyword in the 10 topics
# pprint(lda_model.print_topics(num_words=20))
# doc_lda = lda_model[corpus]

from gensim.models import CoherenceModel
# # Compute Coherence Score
# coherence_model_lda = CoherenceModel(model=lda_model, texts=data_lemmatized, dictionary=id2word, coherence='c_v')
# coherence_lda = coherence_model_lda.get_coherence()
# print('\nCoherence Score: ', coherence_lda)

# supporting function
def compute_coherence_values(corpus, dictionary, k, a, b):
    
    lda_model = gensim.models.LdaMulticore(corpus=corpus,
                                           id2word=dictionary,
                                           num_topics=k, 
                                           random_state=100,
                                           chunksize=100,
                                           passes=10,
                                           alpha=a,
                                           eta=b)
    
    coherence_model_lda = CoherenceModel(model=lda_model, texts=data_lemmatized, dictionary=id2word, coherence='c_v')
    
    return coherence_model_lda.get_coherence()

import numpy as np
import tqdm

grid = {}
grid['Validation_Set'] = {}

# Topics range
min_topics = 2
max_topics = 11
step_size = 1
topics_range = range(min_topics, max_topics, step_size)

# Alpha parameter
alpha = list(np.arange(0.01, 1, 0.5))
alpha.append('symmetric')
alpha.append('asymmetric')

# Beta parameter
beta = list(np.arange(0.01, 1, 0.5))
beta.append('symmetric')

# Validation sets
num_of_docs = len(corpus)
corpus_sets = [gensim.utils.ClippedCorpus(corpus, int(num_of_docs*0.75)), 
               corpus]

corpus_title = ['75% Corpus', '100% Corpus']

model_results = {'Validation_Set': [],
                 'Topics': [],
                 'Alpha': [],
                 'Beta': [],
                 'Coherence': []
                }

# Can take a long time to run
if 1 == 1:
    pbar = tqdm.tqdm(total=(len(beta)*len(alpha)*len(topics_range)*len(corpus_title)))
    
    # iterate through validation corpuses
    for i in range(len(corpus_sets)):
        # iterate through number of topics
        for k in topics_range:
            # iterate through alpha values
            for a in alpha:
                # iterare through beta values
                for b in beta:
                    # get the coherence score for the given parameters
                    cv = compute_coherence_values(corpus=corpus_sets[i], dictionary=id2word, 
                                                  k=k, a=a, b=b)
                    # Save the model results
                    model_results['Validation_Set'].append(corpus_title[i])
                    model_results['Topics'].append(k)
                    model_results['Alpha'].append(a)
                    model_results['Beta'].append(b)
                    model_results['Coherence'].append(cv)
                    
                    pbar.update(1)
    pd.DataFrame(model_results).to_csv('lda_tuning_results_2005_11.csv', index=False)
    pbar.close()
    
lda_model8= gensim.models.LdaMulticore(corpus=corpus,
                                      id2word=id2word,
                                      num_topics=8, 
                                      random_state=100,
                                      chunksize=100,
                                      passes=10,
                                        per_word_topics=True)

from pprint import pprint
# Print the Keyword in the x topics
pprint(lda_model8.print_topics(num_words=30))

import pyLDAvis