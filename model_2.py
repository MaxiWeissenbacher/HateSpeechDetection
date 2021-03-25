#!/usr/bin/env python
# coding: utf-8

# In[1]:


from flair.data import Sentence
from flair.embeddings import StackedEmbeddings, TransformerDocumentEmbeddings, DocumentPoolEmbeddings, FlairEmbeddings
import pandas as pd
import numpy as np
import nltk
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


nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')


# In[1]:


#read excel with training data
tweets_training = pd.read_excel("tweets_preprocessed.xlsx")
#change column stacked_embeddings to type object
tweets_training['stacked_embeddings'] = tweet_training['roberta_embeddings'].astype('object')
#change column flair_embeddings to type object
tweets_training['flair_embeddings'] = tweets_training['flair_embeddings'].astype('object')

#read excel with test data
tweets_test = pd.read_excel("tweets_test_preprocessed.xlsx")
#change column stacked_embeddings to type object
tweets_test['stacked_embeddings'] = tweets_test['roberta_embeddings'].astype('object')
#change column flair_embeddings to type object
tweets_test['flair_embeddings'] = tweets_test['flair_embeddings'].astype('object')

#read excel with test data
news_test = pd.read_excel("news_test_preprocessed.xlsx")
#change column stacked_embeddings to type object
news_test['stacked_embeddings'] = news_test['roberta_embeddings'].astype('object')
#change column flair_embeddings to type object
news_test['flair_embeddings'] = news_test['flair_embeddings'].astype('object')


# In[ ]:


#initialize different embeddings

#initialize transformer embeddings
transformer_embedding = TransformerDocumentEmbeddings('xlm-roberta-large')

#initialize flair embeddings
flair_embedding_forward = FlairEmbeddings('it-forward')
flair_embedding_backward = FlairEmbeddings('it-backward')
document_embeddings = DocumentPoolEmbeddings([flair_embedding_forward, flair_embedding_backward]) 


# In[3]:


#create features for tweets_training for Model 2.1
for i in range(0,len(tweets_training)):

    # create a sentence
    sentence = Sentence(tweets_training['tweets_text'][i])

    # embed the sentence
    transformer_embeddings.embed(sentence)

    embedding = sentence.embedding.cpu()
    
    #save vector as numpy
    embedding = embedding.detach().numpy()
    
    #save vector as pandas dataframe
    embedding = pd.DataFrame(embedding)
    
    #make list out of sentence
    embedding = embedding[0].tolist()

    #add the embedding vector to the column of stacked embeddings
    tweets_training['roberta_embeddings'][i] = embedding
    
    i = i+1


# In[3]:


#create features for tweets_training for Model 2.2
for i in range(0,len(tweets_training)):

    # create a sentence
    sentence = Sentence(tweets_training['tweets_text'][i])

    # embed the sentence
    document_embeddings.embed(sentence)

    embedding = sentence.embedding.cpu()
    
    #save vector as numpy
    embedding = embedding.detach().numpy()
    
    #save vector as pandas dataframe
    embedding = pd.DataFrame(embedding)
    
    #make list out of sentence
    embedding = embedding[0].tolist()

    #add the embedding vector to the column of stacked embeddings
    tweets_training['flair_embeddings'][i] = embedding
    
    i = i+1


# In[5]:


#create features for tweets_training for Model 2.3

#tokenization
tweets_training['tweets_text']= [word_tokenize(entry) for entry in tweets_training['tweets_text']]

#tags for removing stop words
tag_map = defaultdict(lambda : wn.NOUN)
tag_map['J'] = wn.ADJ
tag_map['V'] = wn.VERB
tag_map['R'] = wn.ADV

for index,entry in enumerate(tweets_training['tweets_text']):
    #empty list for storing the final words
    Final_words = []
    for word, tag in pos_tag(entry):
        # Below condition is to check for Stop words and consider only alphabets
        if word not in stopwords.words('italian') and word.isalpha():
            Final_words.append(word)
    #storing the final words in column final_text
    tweets_training.loc[index,'final_text'] = str(Final_words)


# In[ ]:


#create features for tweets_test for Model 2.1
for i in range(0,len(tweets_test)):

    # create a sentence
    sentence = Sentence(tweets_test['tweets_text'][i])

    # embed the sentence
    transformer_embeddings.embed(sentence)

    embedding = sentence.embedding.cpu()
    
    #save vector as numpy
    embedding = embedding.detach().numpy()
    
    #save vector as pandas dataframe
    embedding = pd.DataFrame(embedding)
    
    #make list out of sentence
    embedding = embedding[0].tolist()

    #add the embedding vector to the column of stacked embeddings
    tweets_test['roberta_embeddings'][i] = embedding
    
    i = i+1


# In[ ]:


#create features for tweets_test for Model 2.2
for i in range(0,len(tweets_test)):

    # create a sentence
    sentence = Sentence(tweets_test['tweets_text'][i])

    # embed the sentence
    document_embeddings.embed(sentence)

    embedding = sentence.embedding.cpu()
    
    #save vector as numpy
    embedding = embedding.detach().numpy()
    
    #save vector as pandas dataframe
    embedding = pd.DataFrame(embedding)
    
    #make list out of sentence
    embedding = embedding[0].tolist()

    #add the embedding vector to the column of stacked embeddings
    tweets_test['flair_embeddings'][i] = embedding
    
    i = i+1


# In[ ]:


#create features for tweets_test for Model 2.3

#tokenization
tweets_test['tweets_text']= [word_tokenize(entry) for entry in tweets_test['tweets_text']]

#tags for removing stop words
tag_map = defaultdict(lambda : wn.NOUN)
tag_map['J'] = wn.ADJ
tag_map['V'] = wn.VERB
tag_map['R'] = wn.ADV

for index,entry in enumerate(tweets_test['tweets_text']):
    #empty list for storing the final words
    Final_words = []
    for word, tag in pos_tag(entry):
        # Below condition is to check for Stop words and consider only alphabets
        if word not in stopwords.words('italian') and word.isalpha():
            Final_words.append(word)
    #storing the final words in column final_text
    tweets_test.loc[index,'final_text'] = str(Final_words)


# In[ ]:


#create features for news_test for Model 2.1
for i in range(0,len(news_test)):

    # create a sentence
    sentence = Sentence(news_test['news_text'][i])

    # embed the sentence
    transformer_embeddings.embed(sentence)

    embedding = sentence.embedding.cpu()
    
    #save vector as numpy
    embedding = embedding.detach().numpy()
    
    #save vector as pandas dataframe
    embedding = pd.DataFrame(embedding)
    
    #make list out of sentence
    embedding = embedding[0].tolist()

    #add the embedding vector to the column of stacked embeddings
    news_test['roberta_embeddings'][i] = embedding
    
    i = i+1


# In[ ]:


#create features for news_test for Model 2.2
for i in range(0,len(news_test)):

    # create a sentence
    sentence = Sentence(news_test['news_text'][i])

    # embed the sentence
    document_embeddings.embed(sentence)

    embedding = sentence.embedding.cpu()
    
    #save vector as numpy
    embedding = embedding.detach().numpy()
    
    #save vector as pandas dataframe
    embedding = pd.DataFrame(embedding)
    
    #make list out of sentence
    embedding = embedding[0].tolist()

    #add the embedding vector to the column of stacked embeddings
    news_test['flair_embeddings'][i] = embedding
    
    i = i+1


# In[ ]:


#create features for news_test for Model 2.3

#tokenization
news_test['news_text']= [word_tokenize(entry) for entry in news_test['news_text']]

#tags for removing stop words
tag_map = defaultdict(lambda : wn.NOUN)
tag_map['J'] = wn.ADJ
tag_map['V'] = wn.VERB
tag_map['R'] = wn.ADV

for index,entry in enumerate(news_test['news_text']):
    #empty list for storing the final words
    Final_words = []
    for word, tag in pos_tag(entry):
        # Below condition is to check for Stop words and consider only alphabets
        if word not in stopwords.words('italian') and word.isalpha():
            Final_words.append(word)
    #storing the final words in column final_text
    news_test.loc[index,'final_text'] = str(Final_words)


# In[ ]:


#change column stacked_embeddings to type object
tweets_training['stacked_embeddings'] = tweet_training['roberta_embeddings'].astype('object')
#change column flair_embeddings to type object
tweets_training['flair_embeddings'] = tweets_training['flair_embeddings'].astype('object')

#change column stacked_embeddings to type object
tweets_test['stacked_embeddings'] = tweets_test['roberta_embeddings'].astype('object')
#change column flair_embeddings to type object
tweets_test['flair_embeddings'] = tweets_test['flair_embeddings'].astype('object')

#change column stacked_embeddings to type object
news_test['stacked_embeddings'] = news_test['roberta_embeddings'].astype('object')
#change column flair_embeddings to type object
news_test['flair_embeddings'] = news_test['flair_embeddings'].astype('object')


# In[ ]:


#initialize models
model2_1 = svm.SVC(C=1.0, kernel='linear', gamma=1)
model2_2 = svm.SVC(C=1.0, kernel='linear', gamma=1)
model2_3 = svm.SVC(C=1.0, kernel='rbf', gamma=1)

#create train and test data sets for TF-IDF vectorization
Train_X_tf = tweets_train['text_final']
Train_Y_tf = tweets_train['label']

Test_X_tf_tweets = tweets_test['text_final']
Test_X_tf_news = news_test['text_final']

#TF-IDF-vectorization
Tfidf_vect = TfidfVectorizer(max_features=5000)

Tfidf_vect.fit(tweets_train['text_final'])
Train_X_Tfidf = Tfidf_vect.transform(Train_X_tf)

Tfidf_vect.fit(tweets_test['text_final'])
Test_X_Tfidf_tweets = Tfidf_vect.transform(Test_X_tf_tweets)

Tfidf_vect.fit(news_test['text_final'])
Test_X_Tfidf_news = Tfidf_vect.transform(Test_X_tf_news)

#fit the models on the training set
model2_1.fit(list(tweets_train['roberta_embeddings']),tweets_train['label'])
model2_2.fit(list(tweets_train['flair_embeddings']),tweets_train['label'])
model2_3.fit(Train_X_Tfidf, Train_Y_tf)

#make predicitons
model2_1_pred_tweets = model2_1.predict(list(tweets_test['roberta_embeddings'])
model2_1_pred_news = model2_1.predict(list(news_test['roberta_embeddings'])

model2_2_pred_tweets = model2_2.predict(list(tweets_test['flair_embeddings'])
model2_2_pred_news = model2_2.predict(list(news_test['flair_embeddings'])
                                      
model2_3_pred_tweets = model2_3.predict(list(tweets_test['text_final'])
model2_3_pred_news = model2_3.predict(list(news_test['text_final'])


# In[78]:


#store predicitons in data frame
predictions_tweets = pd.DataFrame()
predictions_tweets['model2_1'] = model2_1_pred_tweets
predictions_tweets['model2_2'] = model2_2_pred_tweets
predictions_tweets['model2_3'] = model2_3_pred_tweets

predictions_news = pd.DataFrame()
predictions_news['model2_1'] = model2_1_pred_news
predictions_news['model2_2'] = model2_2_pred_news
predictions_news['model2_3'] = model2_3_pred_news


# In[ ]:


#do majority voting
predictions_tweets_maj['majority'] = predicitons_tweets.mode(axis=1)[0]
predictions_news_maj['majority'] = predicitons_news.mode(axis=1)[0]

