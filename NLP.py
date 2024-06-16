import tensorflow as tf
from tensorflow import keras, train
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler,MinMaxScaler,MaxAbsScaler
from sklearn.model_selection import train_test_split
import IPython
from IPython import display
import os
from pathlib import Path
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix,multilabel_confusion_matrix
import json
from nltk.corpus import twitter_samples
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer
import string

# df = pd.read_csv('twitter_training.csv')

# df.drop(columns=['ID','Name'],inplace=True)

# # df['Target'].replace(['Positive','Netural','Negative','Irrelevant'],[1,2,0,3],inplace=True)

# df1 = (df[(df['Target'] == 'Positive') | (df['Target'] == 'Negative')])

# df1.reset_index(inplace=True,drop=True)

# df1['Target'].replace(['Positive','Negative'],[1,0],inplace=True)

# print(df1)

# sentences = df1['Message'].values

# labels = df1['Target'].values

# sentences = np.array(sentences,dtype=str)
# labels = np.array(labels)

# print(len(sentences))

# print(len(labels))

# token = keras.preprocessing.text.Tokenizer(10000,oov_token='OOV')

# token.fit_on_texts(sentences)

# print(token.word_index)

# feature_sequence = token.fit_on_sequences(sentences)

# # print(feature_sequence)


positive_tweet = twitter_samples.strings('positive_tweets.json')

negative_tweet = twitter_samples.strings('negative_tweets.json')

print(len(negative_tweet))

train_positive = positive_tweet[:4000]
test_positive = positive_tweet[4000:]


train_negative = negative_tweet[:4000]
test_negative = negative_tweet[4000:]

print(train_positive[:2])

train_x = train_positive + train_negative

test_x = test_positive + test_negative

train_y = np.append(np.ones(len(train_positive)),np.zeros(len(train_negative)),axis=0)


test_y = np.append(np.ones(len(test_positive)),np.zeros(len(test_negative)),axis=0)

string_pun = string.punctuation + ':)'

def process_tweet(tweet):
    """Process tweet function.
    Input:
        tweet: a string containing a tweet
    Output:
        tweets_clean: a list of words containing the processed tweet

    """
    stemmer = PorterStemmer()
    stopwords_english = stopwords.words('english')
    # remove stock market tickers like $GE
    tweet = re.sub(r'\$\w*', '', tweet)
    # remove old style retweet text "RT"
    tweet = re.sub(r'^RT[\s]+', '', tweet)
    # remove hyperlinks
    tweet = re.sub(r'https?:\/\/.*[\r\n]*', '', tweet)
    # remove hashtags
    # only removing the hash # sign from the word
    tweet = re.sub(r'#', '', tweet)
    # tokenize tweets
    tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True,
                               reduce_len=True)
    tweet_tokens = tokenizer.tokenize(tweet)
    
    tweets_clean = []
    for word in tweet_tokens:
        if (word not in stopwords_english and word not in string_pun):  # remove punctuation
            # tweets_clean.append(word)
            stem_word = stemmer.stem(word)  # stemming word
            tweets_clean.append(stem_word)

    return tweets_clean


print(process_tweet(train_x[0]))
print(train_x[0])


def build_freqs(tweets, ys):
    yslist = list(ys)
    
    # Start with an empty dictionary and populate it by looping over all tweets
    # and over all processed words in each tweet.
    freqs = {}
    for y, tweet in zip(yslist, tweets):
        for word in process_tweet(tweet):
            pair = (word, y)
            if pair in freqs:
                freqs[pair] += 1
            else:
                freqs[pair] = 1

    return freqs

freqs = build_freqs(train_x,train_y)


def extract_features(tweet,freqs):
    
    newarra = np.zeros((1,3))
    
    newarra[0,0] = 1
    
    new_tweet = process_tweet(tweet)
    
    for word in new_tweet:
        ##pos
        newarra[0,1] = newarra[0,1] + freqs.get((word,1.0),0)
        ##neg
        
        newarra[0,2] = newarra[0,2] + freqs.get((word,0.0),0)
        
    return newarra



def extract_fea_dataset(train_dataset):
    newarr = np.zeros((len(train_dataset),3))
    for i in range(len(train_dataset)):
        newarr[i] = extract_features(train_dataset[i],freqs)

    return newarr

print(extract_fea_dataset(train_x))

print(train_x[:3])

            
model = keras.Sequential([keras.layers.Dense(1,activation='relu'),
                          keras.layers.Dense(10,activation='relu'),
                          keras.layers.Dense(1,activation='sigmoid')
                          ])

model.compile(optimizer='adam',loss=keras.losses.BinaryCrossentropy(),metrics=['accuracy'])

model.fit(extract_fea_dataset(train_x),train_y,batch_size=32,epochs=20,verbose=1,callbacks=keras.callbacks.EarlyStopping(monitor='loss',patience=6))

predict = model.predict(extract_fea_dataset(test_x))   


print(tf.sigmoid(predict))
    
    
    
    
    













