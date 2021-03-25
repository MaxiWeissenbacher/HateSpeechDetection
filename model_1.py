#!/usr/bin/env python
# coding: utf-8

# In[2]:


from flair.data import Sentence
from flair.embeddings import StackedEmbeddings, TransformerDocumentEmbeddings, DocumentPoolEmbeddings, FlairEmbeddings
import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, svm
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder


# In[4]:


#read excel with training data
tweets_training = pd.read_excel("tweets_preprocessed.xlsx")
#change column stacked_embeddings to type object
tweets_training['stacked_embeddings'] = tweets_training['stacked_embeddings'].astype('object')

#read excel with test data
tweets_test = pd.read_excel("tweets_test_preprocessed.xlsx")
#change column stacked_embeddings to type object
tweets_test['stacked_embeddings'] = tweets_test['stacked_embeddings'].astype('object')

#read excel with test data
news_test = pd.read_excel("news_test_preprocessed.xlsx")
#change column stacked_embeddings to type object
news_test['stacked_embeddings'] = news_test['stacked_embeddings'].astype('object')


# In[19]:


#instantiation of the different embeddings
transformer_embedding = TransformerDocumentEmbeddings('xlm-roberta-large')
flair_embeddin_forward = FlairEmbeddings('it-forward')
flair_embedding_backward = FlairEmbeddings('it-backward')

#initialize the document embeddings
document_embeddings = DocumentPoolEmbeddings([flair_embedding_forward, flair_embedding_backward]) 

#create a StackedEmbedding object that combines transformer and forward/backward flair embeddings
stacked_embeddings = StackedEmbeddings([transformer_embedding,document_embeddings,])


# In[20]:


#create features for tweets_training
for i in range(0,len(tweets_training)):

    # create a sentence
    sentence = Sentence(tweets_training['tweets_text'][i])

    # embed the sentence
    stacked_embeddings.embed(sentence)

    embedding = sentence.embedding.cpu()
    
    #save vector as numpy
    embedding = embedding.detach().numpy()
    
    #save vector as pandas dataframe
    embedding = pd.DataFrame(embedding)
    
    #make list out of sentence
    embedding = embedding[0].tolist()

    #add the embedding vector to the column of stacked embeddings
    tweets_training['stacked_embeddings'][i] = embedding
    
    i = i+1


# In[ ]:


#create features for tweets_test
for i in range(0,len(tweets_test)):

    # create a sentence
    sentence = Sentence(tweets_test['tweets_text'][i])

    # embed the sentence
    stacked_embeddings.embed(sentence)

    embedding = sentence.embedding.cpu()
    
    #save vector as numpy
    embedding = embedding.detach().numpy()
    
    #save vector as pandas dataframe
    embedding = pd.DataFrame(embedding)
    
    #make list out of sentence
    embedding = embedding[0].tolist()

    #add the embedding vector to the column of stacked embeddings
    tweets_test['stacked_embeddings'][i] = embedding
    
    i = i+1


# In[ ]:


#create features for news_test
for i in range(0,len(news_test)):

    # create a sentence
    sentence = Sentence(news_test['news_text'][i])

    # embed the sentence
    stacked_embeddings.embed(sentence)

    embedding = sentence.embedding.cpu()
    
    #save vector as numpy
    embedding = embedding.detach().numpy()
    
    #save vector as pandas dataframe
    embedding = pd.DataFrame(embedding)
    
    #make list out of sentence
    embedding = embedding[0].tolist()

    #add the embedding vector to the column of stacked embeddings
    news_test['stacked_embeddings'][i] = embedding
    
    i = i+1


# In[21]:


#change column stacked_embeddings to type object
tweets_training['stacked_embeddings'] = tweets_training['stacked_embeddings'].astype('object')
tweets_test['stacked_embedding'] = tweets_test['stacked_embedding'].astype('object')
news_test['stacked_embeddings'] = news_test['stacked_embeddings'].astype('object')


# In[28]:


#create train and test data sets
Train_X = tweets_training['stacked_embeddings']
Train_Y = tweets_training['label']

Test_X_tweets = tweets_test['stacked_embeddings']
Test_X_news = news_test['stacked_embeddings']


# In[29]:


#fit the training dataset on the classifier
SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
SVM.fit(list(Train_X),Train_Y)

#predict the labels on validation dataset
predictions_tweets = SVM.predict(list(Test_X_tweets))
predictions_news = SVM.predict(list(Test_X_news))


# In[35]:


#create a dataframe for storing the predictions
predictions_news_ = pd.DataFrame()
predictions_news_['predictions_news'] = predictions_news

predictions_tweets_ = pd.DataFrame()
predictions_tweets_['predictions_tweets'] = predictions_tweets

