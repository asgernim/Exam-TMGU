# -*- coding: utf-8 -*-
"""
@author: Asger Nim 
Exam in TMGU 
"""
#####SECTION 1.0#####

from __future__ import division
import os, re
import numpy as np
import pandas as pd
import pickle

import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

from nltk.tag import pos_tag
from gensim import corpora, models

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn import metrics
from sklearn.linear_model import LogisticRegressionCV

wd ='C:\\Users\\reno0006\\Desktop\\tutorial_py\\group_sos' #wd - absolute path
os.chdir(wd)

import textMiningModule as tm 
import textminer as tm1 #a bit sloppy I know, should probably merge the two merge the two modules together...

#importing original data
df = pd.read_csv('fake_or_real_news.csv', encoding = 'utf-8')
print df.label.value_counts() #balanced dataset # Real: 3171. Fake: 3164

#removing bad rows
bad_rows = []
for row in range(len(df)):              #loop through all rows in the df
    if len(df.loc[row, 'text']) < 100:   #if the len of the text is > 100:
        bad_rows.append(row)            #save the row in the bad_row list

df = df.drop(df.index[bad_rows]) #remove all the bad rows
print len(bad_rows) #print the ammount of removed data # 128 bad rows 
print df.label.value_counts() #still a balanced dataset

# Making a column with cleaned text 

text_clean = []
for text in df['text']:
    text = re.sub(r'[^a-zA-Z]', ' ', text) #replace everything that is not alfapethical with a space
    text = re.sub(r' +', ' ', text) #replace one or more whitespaces with a whitespace
    text = text.rstrip() #remove newlines and other escapecharacters
    text_clean.append(text)

df['text_clean'] = text_clean #adding the clean text to the df


df.to_csv('fake_or_real_news_cleaned_sent.csv', index = False, encoding = 'utf-8')
#
#### section 1.1 #####

#absolute sentiment score 
text_scored = []
for text in df['text']:
    sent_score = tm.labMT_sent(text)
    text_scored.append(sent_score)

df['abs_sent'] = text_scored #adding the scored text to the df

#relative sentiment score 
text_scored = []
for text in df['text']:
    sent_score = tm.labMT_sent(text, rel = True)
    text_scored.append(sent_score)

df['rel_sent'] = text_scored #adding the scored text to the df

#overall mean
df['abs_sent'].mean() 
df['abs_sent'].loc[df['label'] == 'FAKE'].mean() 
df['abs_sent'].loc[df['label'] == 'REAL'].mean() 
#relative score mean calculations
df['rel_sent'].mean() #overall mean
df['rel_sent'].loc[df['label'] == 'FAKE'].mean() 
df['rel_sent'].loc[df['label'] == 'REAL'].mean() 
####

###Graphs to compare in discussion ###
ax = df['rel_sent'].loc[df['label'] == 'FAKE'].plot(kind='line', title ="Fake News", figsize=(15, 10), legend=True, fontsize=12)
ax.set_xlabel("nr.", fontsize=12)
ax.set_ylabel("Relative sentiment score", fontsize=12)
plt.show()

ax = df['rel_sent'].loc[df['label'] == 'REAL'].plot(kind='line', title ="Real News", figsize=(15, 10), legend=True, fontsize=12)
ax.set_xlabel("nr.", fontsize=12)
ax.set_ylabel("Relative sentiment score", fontsize=12)
plt.show()

###

#### Section 1.2 #### 

df = pd.read_csv('fake_or_real_news_cleaned_sent.csv', encoding = 'utf-8')
print df.label.value_counts() #balanced dataset (approx 3000 of each)
print df.loc[1]


#MAKING A TOPIC MODEL DATAFRAME
 #defining a working df  - change this when we want to work with all of the texts
tp_df = df
#insert articles into a list
texts_tokenized = []
for text in tp_df['text_clean']:
    tokens = tm1.tokenize(text, length = 1, casefold = False) #casefold equal false because we want uppercase letters to categorize the text using pos_tag
    tagset = pos_tag(tokens, tagset = 'universal', lang = 'eng') #tag tokens with their category
    tokens = [tag[0] for tag in tagset if tag[1] in ['NOUN']] #only retain nouns
    tokens = [token.lower() for token in tokens] #lowercase the tokens
    texts_tokenized.append(tokens)
print type(texts_tokenized[0][0])  #the word in the text
print type(texts_tokenized[0])  #list of words in text
print type(texts_tokenized) #list of texts
#So it is a string within a list within a list (the first list is the text, the second list the nouns in the text and the string is the noun)
    #making a stopwordlist
sw = tm1.gen_ls_stoplist(texts_tokenized, 40)
print sw #this stopword might say some general things about the period of the articles rather than something about the topics
#for now let's just not use it

"""
#applying stopword list to all texts#
nosw_texts = []
for text in texts_tokenized:
    nosw_text = []
    nosw_text =[token for token in text if token not in sw]
    nosw_texts.append(nosw_text)
texts_tokenized = nosw_texts#overwrite with text without the stopwords
dictionary = corpora.Dictionary(texts_tokenized) """


    #bag-of-words representation of the texts
dictionary = corpora.Dictionary(texts_tokenized)
texts_bow = [dictionary.doc2bow(text) for text in texts_tokenized] #bow = bag of words
   
 #training the topic model
k = 10 #number of topics
mdl = models.LdaModel(texts_bow, id2word = dictionary,
                      num_topics = k, random_state = 12345) 
for i in range(k): 
    print 'topic', i
    print [t[0] for t in mdl.show_topic(i, 20)]
    print '----'
    
### ADDING TOPIC TO DATA FRAME AND MAKING A HEAT MAP ### 

def get_theta(doc_bow, mdl):
    tmp = mdl.get_document_topics(doc_bow, minimum_probability=0)
    return [p[1] for p in tmp]

df_topic = pd.DataFrame() #making empty dataframe

for topicnr in range(k):
    topic_name = 'topic %d' %topicnr
    topic_score = []
    print topicnr
    for text in range(len(df)):
        topic_score.append(get_theta(texts_bow[text], mdl)[topicnr])
    topic_name = 'topic %d' %topicnr
    df_topic[topic_name] = topic_score

print mdl[texts_bow[1] #show the topic of the text
print df_topic.head()
print len(df_topic)

#making a heatmap for 10 topics 
df_topic['label'] = df['label']
df_topic = df_topic.sort_values('label')
del df_topic['label']
df_topic.tail()
heat_matrix = df_topic.as_matrix()
ax = sns.heatmap(heat_matrix, yticklabels=False) 





                           
                           
                           
                           
                           
                           
