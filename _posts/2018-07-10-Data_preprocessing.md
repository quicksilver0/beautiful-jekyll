---
layout: post
title: Cheatsheet: Data Preprocessing
date: 2018-07-10
category: cheatsheet
tags: [cheatsheet preprocessing]
---

This is a cheatsheet article with steps on data preprocessing required for deep learning endeavors.

Read in and slice
First read in dataset and choose relevant columns.


```python
# Importing the dataset
dataset = pd.read_csv('dataset.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values
```

### Handle Nulls
Check if any nulls present and decide on those.


```python
# Checks columns for nulls
X.isnull().sum()
X.fillna() # choose a way to fill nulls
X.dropna() # choose a way to drop nulls
X.drop('colname',axis=1) # possibly drop entirely some columns
```

### Encode categorical data
Encode categorical, also hot encode single valued categoricals.


```python
# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
# fits and transforms on categorical variables of 1st column - makes dummies
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
# Get rid of one of the dummies (the first one) if necessary - to avoid dummy trap.
X = X[:, 1:]
```

### Split the dataset
Splitting the dataset into the Training set and Test set


```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
```

### Feature Scaling


```python
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
```
