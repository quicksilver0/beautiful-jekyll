---
title: Word Frequency From a Text

date: 2017-10-04
tags: project pandas mapping basemap
category: project
layout: post
comments: true
excerpt_separator: <!--more-->
image: /img/text_mining/alice03.png
---

Suppose you want to get top frequent words from a text. This task quickly reveals the caveats: there are swarms of words each with dozens of forms, all those n't and 's and and also commas and periods... All these should be accounted for.

Good for us, there are very powerful tools exist for word processing and text mining - libraries that handles these tasks in a best way possible.

Get familiar with **nltk** - a powerful library for NLP (natural language processing)!

<!--more-->

Ok, so the aim is to get word frequencies. The workflow would be:
- imports - get libraries
- Normalize - prepare words
- Tokenize - smart word split
- Lemmatize - smart processing to meaningful 'universal' words.
- Get word frequencies!

Follow these simple, yet powerful text mining steps.

#### imports
First import `nltk`, also you might probably need to do `nltk.download()` to get all resources. That might take some time!


```python
import nltk
nltk.download()
```

    showing info https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/index.xml
    




    True



Loading raw version of Alice in Wonderland as an example for our analysis.


```python
alice = nltk.corpus.gutenberg.raw('carroll-alice.txt')
len(alice)
```




    144395



#### Normalization
Is simply lowercasing all words nicely.


```python
alice_lower = alice.lower()
print(alice_lower[:393])
```

    [alice's adventures in wonderland by lewis carroll 1865]
    
    chapter i. down the rabbit-hole
    
    alice was beginning to get very tired of sitting by her sister on the
    bank, and of having nothing to do: once or twice she had peeped into the
    book her sister was reading, but it had no pictures or conversations in
    it, 'and what is the use of a book,' thought alice 'without pictures or
    conversation?'
    
    

#### Tokenize
Tokenization is a smart splitting of the text to it's constituent parts: words, symbols, endings. `nltk` has a powerful tool `word_tokenize` and takes into consideration many different cases.


```python
alice_tokenized = nltk.word_tokenize(alice_lower)
print(alice_tokenized[:80])
```

    ['[', 'alice', "'s", 'adventures', 'in', 'wonderland', 'by', 'lewis', 'carroll', '1865', ']', 'chapter', 'i.', 'down', 'the', 'rabbit-hole', 'alice', 'was', 'beginning', 'to', 'get', 'very', 'tired', 'of', 'sitting', 'by', 'her', 'sister', 'on', 'the', 'bank', ',', 'and', 'of', 'having', 'nothing', 'to', 'do', ':', 'once', 'or', 'twice', 'she', 'had', 'peeped', 'into', 'the', 'book', 'her', 'sister', 'was', 'reading', ',', 'but', 'it', 'had', 'no', 'pictures', 'or', 'conversations', 'in', 'it', ',', "'and", 'what', 'is', 'the', 'use', 'of', 'a', 'book', ',', "'", 'thought', 'alice', "'without", 'pictures', 'or', 'conversation', '?']
    

#### Filter Stop words
There are a lot of words, such as 'a', 'or', 'the'. By the way 'the' is the most frequent word in english. Since we are interested in meaningful words we shall filter out any stop words.


```python
from nltk.corpus import stopwords
stopWords = set(stopwords.words('english'))
alice_tokenized = [ token for token in alice_tokenized if token not in stopWords ]
```

#### Filter punctuation
Punctuation symbols are top frequent as well for the texts. For our purposes we should filter them out as well.


```python
alice_tokenized = [token for token in alice_tokenized if token.isalpha()]
```

#### Lemmatize
Lemmatization, is a smart-stemming. Stemming finds and transforms a word to it's stem. For example the step of `universal` is `univers` - which is not very appealing for a human. While lemmatization create meaningful words, thus `universal` or `universally` would be transformed to actually `universal`!

See examples of stemming and lemmatization:


```python
# Stemming
porter = nltk.PorterStemmer()
alice_stemmed = [porter.stem(t) for t in alice_tokenized] # Still Lemmatization
print(alice_stemmed[:80])
```

    ['alic', 'adventur', 'wonderland', 'lewi', 'carrol', 'chapter', 'alic', 'begin', 'get', 'tire', 'sit', 'sister', 'bank', 'noth', 'twice', 'peep', 'book', 'sister', 'read', 'pictur', 'convers', 'use', 'book', 'thought', 'alic', 'pictur', 'convers', 'consid', 'mind', 'well', 'could', 'hot', 'day', 'made', 'feel', 'sleepi', 'stupid', 'whether', 'pleasur', 'make', 'would', 'worth', 'troubl', 'get', 'pick', 'daisi', 'suddenli', 'white', 'rabbit', 'pink', 'eye', 'ran', 'close', 'noth', 'remark', 'alic', 'think', 'much', 'way', 'hear', 'rabbit', 'say', 'dear', 'oh', 'dear', 'shall', 'late', 'thought', 'afterward', 'occur', 'ought', 'wonder', 'time', 'seem', 'quit', 'natur', 'rabbit', 'actual', 'took', 'watch']
    


```python
# Lemmatization. Notice the diffetence.
WNlemma = nltk.WordNetLemmatizer()
alice_lemmatized = [WNlemma.lemmatize(t) for t in alice_tokenized]
print(alice_lemmatized[:80])
```

    ['alice', 'adventure', 'wonderland', 'lewis', 'carroll', 'chapter', 'alice', 'beginning', 'get', 'tired', 'sitting', 'sister', 'bank', 'nothing', 'twice', 'peeped', 'book', 'sister', 'reading', 'picture', 'conversation', 'use', 'book', 'thought', 'alice', 'picture', 'conversation', 'considering', 'mind', 'well', 'could', 'hot', 'day', 'made', 'feel', 'sleepy', 'stupid', 'whether', 'pleasure', 'making', 'would', 'worth', 'trouble', 'getting', 'picking', 'daisy', 'suddenly', 'white', 'rabbit', 'pink', 'eye', 'ran', 'close', 'nothing', 'remarkable', 'alice', 'think', 'much', 'way', 'hear', 'rabbit', 'say', 'dear', 'oh', 'dear', 'shall', 'late', 'thought', 'afterwards', 'occurred', 'ought', 'wondered', 'time', 'seemed', 'quite', 'natural', 'rabbit', 'actually', 'took', 'watch']
    

#### Get Distribution
`FreqDist` creates a dictionary of word and it's frequency


```python
word_tokens_freqdist = nltk.FreqDist(alice_lemmatized)
list(list(word_tokens_freqdist.keys())[:10])
```




    ['alice',
     'adventure',
     'wonderland',
     'lewis',
     'carroll',
     'chapter',
     'beginning',
     'get',
     'tired',
     'sitting']




```python
# Get words with frequency >60 and length >5.
freqwords = ['{}:{}'.format(w,dist[w]) for w in word_tokens_freqdist.keys() if len(w) > 5 and word_tokens_freqdist[w] > 60]
freqwords
```




    ['thought:76', 'little:128']



FreqDist class provides us with handy methods to get top frequent words from the dictionary of frequencies in a nice way.

Getting top 10 frequent words from a text!


```python
# Get top 10 frequent words from a text
top_freq_words = word_tokens_freqdist.most_common(10)
print(top_freq_words)
```

    [('said', 462), ('alice', 396), ('little', 128), ('one', 100), ('would', 90), ('know', 90), ('could', 86), ('like', 86), ('went', 83), ('thing', 79)]
    

We can see top frequent words with their occurence in the text presented in a list of tuples.

Cool! 

These are some great basiscs of NLP. We could get get most frequent meaningfult words from a beloved tale Alice in Wonderland. *Said alice little one...*

Hope it was helpful. Your comments and insights are very welcomed.
