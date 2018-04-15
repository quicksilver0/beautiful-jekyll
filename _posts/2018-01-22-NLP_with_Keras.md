---
title: Prediction using NLP and Keras Neural Net

date: 2018-01-22
tags: machine learning Keras NN deep learning NLP
category: tutorial
layout: post
comments: true
excerpt_separator: <!--more-->
image: /img/Keras/horn_of_plenty_sq.jpg
---

This Notebook focuses on NLP techniques combined with Keras-built Neural Networks. The idea is to complete end-to-end project and to understand best approaches to text processing with Neural Networks by myself on practice. The tutorial provides vivid understanding of how to prepare the data for a Neural Network with Keras and how to actually implement and run it.

**Project description:** predict if the review of the film is positive or negative. The dataset is a set of imdb reviews labeled as positive/negative.

It is inspired by a [DeepLearning with NLP CrashCourse](https://machinelearningmastery.com/crash-course-deep-learning-natural-language-processing/) by Dr. Jason Brownlee.
<!--more-->
### Import libraries


```python
import pandas as pd
import glob
import numpy as np
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from numpy import asarray
from numpy import zeros
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import gc

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers.embeddings import Embedding
from keras.models import Sequential
from keras.layers import Convolution1D, Conv1D, Flatten, Dropout, Dense, MaxPooling1D
from keras.callbacks import TensorBoard
```

### Get Timestamps
Define a function to display timespent


```python
import datetime

def display_time_spent():
    end_time = datetime.datetime.now()
    time_spent = (end_time - start_time)
    
    hours, remainder = divmod(time_spent.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    duration_formatted = '%d:%02d:%02d' % (hours, minutes, seconds)
    print('Wall time: {}'.format(duration_formatted))
```

Put this at the start of the block execution


```python
start_time = datetime.datetime.now()
```

Put this at the end of the block execution


```python
display_time_spent()
```

    Wall time: 0:00:00
​    

### Read in Data
The dataset is collection of 1000 positive and 1000 negative imdb reviews.
Can be [downloaded here.](http://www.cs.cornell.edu/people/pabo/movie-review-data/review_polarity.tar.gz)


```python
# Define a function to read the file and return as a string
def read_file(file):
    f=open(file)
    return f.read()
    f.close()

# Read in positive reviews
positive_files = glob.glob('nlp_keras_embedding/data/txt_sentoken/pos/cv*.txt')
positive_reviews_list = [ read_file(file) for file in positive_files ]
labels = [1]*len(positive_reviews_list)
reviews_positive_df = pd.DataFrame(data={'review': positive_reviews_list, 'label': labels})

# Read in negative reviews
negative_files = glob.glob('nlp_keras_embedding/data/txt_sentoken/neg/cv*.txt')
negative_reviews_list = [ read_file(file) for file in negative_files ]
labels = [0]*len(negative_reviews_list)
reviews_negative_df = pd.DataFrame(data={'review': negative_reviews_list, 'label': labels})

# Concatenate the dataframes into one
reviews_df = pd.concat([reviews_positive_df,reviews_negative_df], ignore_index=True)

reviews_df.head()
```



### Split the dataset


```python
labels = reviews_df['label']
dataset = reviews_df['review']
```


```python
# Split dataset
X_train, X_test, y_train, y_test = train_test_split(dataset, labels, test_size=0.1, random_state=4)
```

### Vectorize and Preprocess text
Make Bag Of Word matrix Representation and fit vectorizer on combined dataset.
I've experimented with different Vectorizers, turns out _binary_ shows best results with NNs.

Notice words preprocessing options passed to CountVectorizer:
- token pattern would be minimum of 3 chars (any letter, a digit or "-","_" signs)
- english stopwords are filtered out
- ngrams would be generated from single to 3 tokens each
- also minimum word occurence should be 3 or more, so min_df=3
  Also CountVectorizer automatically lowercases the text.


```python
# fit vectorizer
vectorizer = CountVectorizer(binary=True, min_df=3, ngram_range=(1,3), token_pattern='(?u)\\b[a-z0-9\-\_]{3,}\\b', stop_words='english')
#vectorizer = CountVectorizer(binary=True, min_df=3, ngram_range=(1,3), token_pattern='(?u)\\b[a-z0-9\-\_][a-z0-9\-\_]+\\b')
#vectorizer = TfidfVectorizer(min_df=3, ngram_range=(1,3))
# tokenize and build vocab
vectorizer.fit(X_train)
# summarize
#print(vectorizer.vocabulary_)
```


    CountVectorizer(analyzer='word', binary=True, decode_error='strict',
            dtype=<class 'numpy.int64'>, encoding='utf-8', input='content',
            lowercase=True, max_df=1.0, max_features=None, min_df=3,
            ngram_range=(1, 3), preprocessor=None, stop_words='english',
            strip_accents=None, token_pattern='(?u)\\b[a-z0-9\\-\\_]{3,}\\b',
            tokenizer=None, vocabulary=None)



Transform train and test datasets to vector sparse matrices, and then to the arrays. (We need it in array forms in order to pass to the Keras NN layer)


```python
X_train_vec = vectorizer.transform(X_train)
X_train_arr = X_train_vec.toarray()
# summarize encoded vector
print(X_train_arr.shape)
```

    (1800, 34402)
Test dataset:    


```python
X_test_vec = vectorizer.transform(X_test)
X_test_arr = X_test_vec.toarray()
# summarize encoded vector
print(X_test_arr.shape)
```

    (200, 34402)
	
So we have quite large arrays to feed into our Neural Network. Train dataset is 1800 rows (samples) of 34k features! Let's see further on how the Net learns from this saprse representation.

Get length of a wordspace. This length would be used as argument input to our NN model.


```python
n_words = X_train_arr.shape[1]
```

### Define a Model
Define Neural Network Architecture and compile.

A cursory exploration ended up with 2 layer architecture of 100 neurons each, with 0.1 dropout regularization showing best results.


```python
# define NN model
model = Sequential()
model.add(Dense(100, input_shape=(n_words,), activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(1, activation='sigmoid'))
# compile NN network
tensorBoardCallback = TensorBoard(log_dir='./logs', write_graph=True)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
```

### Train and evaluate
Fit the model and evaluate.


```python
start_time = datetime.datetime.now()

# fit network
model.fit(X_train_arr, y_train, epochs=10, callbacks=[tensorBoardCallback], verbose=2)

display_time_spent()
```

    Epoch 1/10
     - 3s - loss: 0.4847 - acc: 0.7694
    Epoch 2/10
     - 3s - loss: 0.0281 - acc: 0.9956
    Epoch 3/10
     - 3s - loss: 0.0023 - acc: 1.0000
    Epoch 4/10
     - 3s - loss: 8.0274e-04 - acc: 1.0000
    Epoch 5/10
     - 3s - loss: 4.5116e-04 - acc: 1.0000
    Epoch 6/10
     - 3s - loss: 2.7223e-04 - acc: 1.0000
    Epoch 7/10
     - 3s - loss: 1.6565e-04 - acc: 1.0000
    Epoch 8/10
     - 3s - loss: 1.2852e-04 - acc: 1.0000
    Epoch 9/10
     - 3s - loss: 8.2725e-05 - acc: 1.0000
    Epoch 10/10
     - 3s - loss: 5.5967e-05 - acc: 1.0000


    Wall time: 0:01:26
	
Evaluate: 


```python
start_time = datetime.datetime.now()

# evaluate
loss, acc = model.evaluate(X_test_arr, y_test, verbose=0)
print('Test Accuracy: %f' % (acc*100))

display_time_spent()
```

    Test Accuracy: 91.000000
    Wall time: 0:00:00


### Wrap up

Ok, Nice! A simple Neural Network with just 2 layers, 100 neurons each gives very good results on predicting from relatively large bag of words!

So far, this tutorial explains in full:
- text preprocessing
- test preprocessing and dataset preparation
- NN model implementation with Keras
- prediction

Let me know of your ideas, additions and approaches to this problem. I'd be happy to hear from you and answer the questions.
