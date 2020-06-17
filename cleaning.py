#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 13:17:47 2020

@author: aadi
"""


import pandas as pd

data=pd.read_csv('india.csv')
data['category'] =data['Flair'].map({'Politics': 0,})
data=data.dropna(subset=['category'])
data=data.dropna(subset=['Title'])
data['Title'].nunique()
data = data.drop_duplicates(subset=['Title'])
data=data.drop('category',axis=1)
data.to_csv("reddit_politics.csv")