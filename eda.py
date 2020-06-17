#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 12:33:10 2020

@author: aadi
"""

import re
import pandas as pd
import nltk
import numpy as np
from wordcloud import WordCloud
import cufflinks as cf
cf.go_offline()
cf.set_config_file(offline=False, world_readable=True)
import matplotlib
import matplotlib.pyplot as plt
%matplotlib inline
pd.options.mode.chained_assignment = None
pd.set_option('display.max_colwidth', 100)
matplotlib.rcParams.update(
    {
        'text.usetex': False,
        'font.family': 'stixgeneral',
        'mathtext.fontset': 'stix',
    }
)
import seaborn as sns




#import data of reddit 
data = pd.read_csv('reddit.csv')

# check shape of dataframes
print(data.shape)



#Data Preprocessing
def sub_preprocess(sub):
   
    # run regex to remove certain characters
    sub['Title'] = sub['Title'].map(lambda x: re.sub(r"[@\?\.$%_\[\]()+-:*\"]", ' ', x, flags=re.I))
    sub['Title'] = sub['Title'].map(lambda x: re.sub(r"[,']", '', x, flags=re.I))
    sub['Title'] = sub['Title'].map(lambda x: re.sub("(?<![\w'])\w+?(?=\b|'s)", ' ', x))

    # run regex to remove line breaks and tabs
    sub['Title'] = sub['Title'].map(lambda x: re.sub(r"\s+", ' ', x))
    
sub_preprocess(data)


#storing stopwords
stopwords = nltk.corpus.stopwords.words('english') 



#tokenization ,stopwords removal and Lemmatization
tokenized_tweet = data['Title'].apply(lambda x: x.split())
tokenized_tweet.head()
tokenized_tweet = tokenized_tweet.apply(lambda x: [w for w in x if w not in stopwords])

lemmatizer = nltk.stem.WordNetLemmatizer()
tokenized_tweet = tokenized_tweet.apply(lambda x: [lemmatizer.lemmatize(i) for i in x]) #lemmatizer
tokenized_tweet.head()
for i in range(len(tokenized_tweet)):
    tokenized_tweet[i] = ' '.join(tokenized_tweet[i])
#title_pp - column after stopwords removal and lemmatization
data['title_pp'] = tokenized_tweet
print(data.shape)

def analysis(df):
 
  df['len'] = df['Title'].astype(str).apply(len)
  df['word_count'] = df['Title'].apply(lambda x: len(str(x).split()))
  from textblob import TextBlob
  data['polarity']=data['Title'].apply(lambda x:TextBlob(x).sentiment.polarity)

analysis(data)
print(data.shape)


print("3 Random Title with Highest Polarity:")
for index,Title in enumerate(data.iloc[data['polarity'].sort_values(ascending=False)[:3].index]['Title']):
  print('Title {}:\n'.format(index+1),Title)
  
  
#Polarity Distribution
plt.figure(figsize=(15,10))
plt.margins(0.02)
plt.xlabel('Sentiment', fontsize=50)
plt.xticks(fontsize=40)
plt.ylabel('Frequency', fontsize=50)
plt.yticks(fontsize=40)
plt.hist(data['polarity'], bins=50)
plt.title('Polarity Distribution', fontsize=60)
plt.show()

#Length of Title vs No. of Posts Distribution
plt.figure(figsize=(15,10))
plt.margins(0.02)
plt.xlabel('Length', fontsize=50)
plt.xticks(fontsize=40)
plt.ylabel('Frequency', fontsize=50)
plt.yticks(fontsize=40)
plt.hist(data['len'], bins=50)
plt.title('Length of Title Distribution', fontsize=60)
plt.show()

#Word Count Distribution
plt.figure(figsize=(15,10))
plt.margins(0.02)
plt.xlabel('Length', fontsize=50)
plt.xticks(fontsize=40)
plt.ylabel('Frequency', fontsize=50)
plt.yticks(fontsize=40)
plt.hist(data['word_count'], bins=50)
plt.title('Word Count Distribution', fontsize=60)
plt.show()

#correlation between polarity , length and word_count
correlation = data[['polarity', 'len', 'word_count']].corr()
mask = np.zeros_like(correlation, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
plt.figure(figsize=(10,10))
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)
sns.heatmap(correlation, cmap='coolwarm', annot=True, annot_kws={"size": 24}, linewidths=10, vmin=-1.5, mask=mask)



from sklearn.feature_extraction.text import CountVectorizer

#Most common words in title
co = CountVectorizer()
counts = co.fit_transform(data.Title)
print(counts)
pd.DataFrame(counts.sum(axis=0),columns=co.get_feature_names()).T.sort_values(0,ascending=False).head(50)

#Most rare words in Title
pd.DataFrame(counts.sum(axis=0),columns=co.get_feature_names()).T.sort_values(0,ascending=True).head(50)
  
#Most common words in Title(after stopwords removal and lemmatization)
counts = co.fit_transform(data.title_pp)
pd.DataFrame(counts.sum(axis=0),columns=co.get_feature_names()).T.sort_values(0,ascending=False).head(50)
 #Most rare words in Title(after stopwords removal and lemmatization) 
pd.DataFrame(counts.sum(axis=0),columns=co.get_feature_names()).T.sort_values(0,ascending=True).head(50)
#Top 10 most common bigrams
co = CountVectorizer(ngram_range=(2,2),stop_words=stopwords)
counts = co.fit_transform(data.Title)
pd.DataFrame(counts.sum(axis=0),columns=co.get_feature_names()).T.sort_values(0,ascending=False).head(10)

#Top 10 most common tigrams
co = CountVectorizer(ngram_range=(3,3),stop_words=stopwords)
counts = co.fit_transform(data.Title)
pd.DataFrame(counts.sum(axis=0),columns=co.get_feature_names()).T.sort_values(0,ascending=False).head(10)


# the vectorizer object will be used to transform text to vector form
vectorizer = CountVectorizer(max_df=0.9, min_df=25, token_pattern='\w+|\$[\d\.]+|\S+')

# apply transformation
tf = vectorizer.fit_transform(data['title_pp']).toarray()

# tf_feature_names tells us what word each column in the matric represents
tf_feature_names = vectorizer.get_feature_names()

from sklearn.decomposition import LatentDirichletAllocation

number_of_topics =3

model = LatentDirichletAllocation(n_components=number_of_topics, random_state=0)
model.fit(tf)
pd.set_option('display.expand_frame_repr', False)
def display_topics(model, feature_names, no_top_words):
    topic_dict = {}
    for topic_idx, topic in enumerate(model.components_):
        topic_dict["Topic %d words" % (topic_idx)]= ['{}'.format(feature_names[i])
                        for i in topic.argsort()[:-no_top_words - 1:-1]]
        topic_dict["Topic %d weights" % (topic_idx)]= ['{:.1f}'.format(topic[i])
                        for i in topic.argsort()[:-no_top_words - 1:-1]]
    return pd.DataFrame(topic_dict)

no_top_words = 6
display_topics(model, tf_feature_names, no_top_words)

