#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 12:33:10 2020

@author: aadi
"""

import re
import pandas as pd
import nltk
from wordcloud import WordCloud
import cufflinks as cf
cf.go_offline()
cf.set_config_file(offline=False, world_readable=True)
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib auto
matplotlib.rcParams['figure.figsize'] = (10.0, 6.0)
import plotly.graph_objs as go
import chart_studio.plotly  as py




#import data of reddit/sports 
data = pd.read_csv('reddit.csv')

# check shape of dataframes
print(data.shape)



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


#function to give wordcloud image of words used in text column
def cloud(df):

    t = ' '
# iterate through the csv file 
    for x in df['Title']: 
          
        # typecaste each val to string 
        x = str(x) 
      
        # split the value 
        values = x.split() 
          
        # Converts each token into lowercase 
        for i in range(len(values)): 
            values[i] = values[i].lower() 
              
        for words in values: 
            t = t + words + ' '
      
  
    wc = WordCloud(max_words= 50,
                      width = 744, 
                      height = 544,
                      background_color ='white',
                      stopwords=stopwords, 
                      contour_width=3, 
                      contour_color='steelblue',
                      min_font_size = 10).generate(t) 
  
    # plot the WordCloud image                        
    plt.figure(figsize = (10, 10)) 
    plt.imshow(wc) 
    plt.axis("off")
    #plt.savefig('reddit2.png')

cloud(data)

#tokenization ,stopwords removal and Lemmatization
tokenized_tweet = data['Title'].apply(lambda x: x.split())
tokenized_tweet.head()
tokenized_tweet = tokenized_tweet.apply(lambda x: [w for w in x if w not in stopwords])

lemmatizer = nltk.stem.WordNetLemmatizer()
tokenized_tweet = tokenized_tweet.apply(lambda x: [lemmatizer.lemmatize(i) for i in x]) #lemmatizer
tokenized_tweet.head()
for i in range(len(tokenized_tweet)):
    tokenized_tweet[i] = ' '.join(tokenized_tweet[i])
#Text_pp - column after stopwords removal and stemming
data['title_pp'] = tokenized_tweet
print(data.shape)

def analysis(df):
 # df['polarity'] = df['Title'].map(lambda text: TextBlob(text).sentiment.polarity)
  df['len'] = df['Title'].astype(str).apply(len)
  df['word_count'] = df['Title'].apply(lambda x: len(str(x).split()))
  
analysis(data)
print(data.shape)

fig=data['len'].iplot(
    kind='hist',
    bins=100,
    xTitle='length',
    linecolor='black',
    yTitle='count',
    title='Title Text Length Distribution')
fig.show()

fig, ax = plt.subplots(1,1,figsize=(10,5))
sns.boxplot(x = 'len', y = 'Post', data = data, orient="h")
plt.title('Word Count per Post - Full Data w/Outliers', fontsize = 20)
plt.xlabel("Word Count", fontsize = 15)
plt.ylabel("Subreddit", fontsize = 15)

fig, ax = plt.subplots(1,1,figsize=(10,5))
sns.boxplot(x = 'word_count', y = 'subreddit', data = c_gs_iqr, orient="h")
plt.title('Word Count per Post - Data without Outliers', fontsize = 20)
plt.xlabel("Word Count", fontsize = 15)
plt.ylabel("Subreddit", fontsize = 15
