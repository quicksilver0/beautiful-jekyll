---
title: Filter Spam with Machine Learning

date: 2017-12-06
tags: machine learning text processing tutorial
category: tutorial
layout: post
comments: true
excerpt_separator: <!--more-->
image: /img/bayes/bayesian-brain1_sq.png
---

This guide contains a very simple, yet powerful Machine Learning technique to filter the spam!

Multinomial Naive Bayes with minimal preprocessing yields incredible results! I was quite amazed by it's efficiency to learn from text.

Please know, this technique is derived from [Applied Text Mining in Python](https://www.coursera.org/learn/python-text-mining/home/welcome) course by University of Michingan, which I highly recommend to anyone in Computer Science and Machine Learning.

Ok, let's dive in.
<!--more--> 
### Read data
First we need data. You can get [spam.csv here](https://github.com/SilverSurfer0/SilverSurfer0.github.io/blob/master/data/spam.csv). The file contains labeled data to train our model on. Usually this is data labeled manually by humans or users.


```python
import pandas as pd
import numpy as np

spam_data = pd.read_csv('spam.csv')

spam_data['target'] = np.where(spam_data['target']=='spam',1,0)
spam_data.head(10)
```

    0	Go until jurong point, crazy.. Available only ...	0
    1	Ok lar... Joking wif u oni...	0
    2	Free entry in 2 a wkly comp to win FA Cup fina...	1
    3	U dun say so early hor... U c already then say...	0
    4	Nah I don't think he goes to usf, he lives aro...	0
    5	FreeMsg Hey there darling it's been 3 week's n...	1
    6	Even my brother is not like to speak with me. ...	0
    7	As per your request 'Melle Melle (Oru Minnamin...	0
    8	WINNER!! As a valued network customer you have...	1
    9	Had your mobile 11 months or more? U R entitle...	1
```	
	
The next step would be splitting the datasets into Train and Test portions, using `train_test_split`.

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(spam_data['text'], 
                                                    spam_data['target'], 
                                                    random_state=0)
```

### Preprocess data

Now we need to preprocess data into so-called term-matrix. It is a two-dimensional matrix where each row is a corresponding sample and each token (word) is presented as a feature - a column. The values are the number of token occurence in each sample.

`sklearn` provides a tool `CountVecotrizer`, which essentially does all the heavy lifting of preprocessing: tokenizes, transforms data to a matrix and also comes with a bunch of other useful parameters, such as presenting features as [ngrams](https://en.wikipedia.org/wiki/N-gram) of words or filtering out the stopwords.

Ok, great, all we need is to run it.


```python
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer() # Instantiate vectorizer
X_vect_matrix = vectorizer.fit_transform(X_train) # Generate term document matrix of tokens
```

### Train the model
We have the matrix! We can now train the Multinomial Bayes Model.


```python
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import roc_auc_score

bayes_model = MultinomialNB(alpha=0.1) # Instantiate Multinomial Naive Bayes with slight smoothing parameter 0.1
bayes_model.fit(X_vect_matrix, y_train) # Train Bayes
```


    MultinomialNB(alpha=0.1, class_prior=None, fit_prior=True)



Few lines of code and our model is trained.

Now we can see result! (Remember that X_test should be similarly transformed into term-matrix)


```python
predictions = bayes_model.predict(vectorizer.transform(X_test)) # Obtain predictions using X_test transformed to a term document matrix.
auc = roc_auc_score(y_test,predictions) # Measure roc_auc

print(auc)
```

    0.972081218274
​    

**97.2%!** Spectacular, isn't it? With just a few lines of code.

Yes, usually text preprocessing requires much more effort and even ingenuity to engineer new features in case of Machine Learning. Yet, Naive Bayes handles such tasks really good.

Good Luck:)
