---
title: ML Modelling Workflow Tutorial
date: 2017-09-26
tags: tutorial ML Machine MachineLearning Trees GridSearch
category: tutorial

layout: post
comments: true
excerpt_separator: <!--more-->
image: /img/machine_learning/forest.jpg
---
This is a solid and quick Machine Learning tutorial that leads through the steps of building a best prediction model. It allows to get understanding of the process and provides with code examples.

I'll be using Logistic Regressor and Trees (either Gradient Boosted or Ensembles are good), for saving space, although I encourage you to run different classifiers.

Originally built on a project I've completed for Michigan University course [Applied Machine Learning](https://www.coursera.org/learn/python-machine-learning/home/welcome). Which I highly recommend to anyone passionate with Data Science.
<!--more-->



![English Forest](/img/machine_learning/forest.jpg "In Machine Learning you may grow Forests!")

*Picture of English Forest is not accidental. In machine Learning you can grow trees. Thousands of them!*



#### The dataset

The dataset is relatively large - 60k rows, with a dozen of features in a feature space. For discretion reasons, I can not include the dataset, as it is used for student scoring.

##### A typical Machine Learning workflow is as following:
- get acquinted with the data
- concise the datasets - deal with nans, trim for only relevant data. (Classifiers cant deal with NaNs and throw errors!)
- merge datasets
- decide on the features (ensure not to present data leakages), and preprocess the features, such as: convert to nums, minmax transformation, polynomial transformation (for regression tasks)
- ===> Above steps - the data preparation takes considerable input and an essential step! <===
- split Train dataset
- understand the distribution of target labels, and it could be a good idea to train a dummy for comparison.
- choose relevant classifier, tune for best parameters according to the scoring of your choice (auc, precision, recall, accuracy, f1) via Grid Search
- repeat previous test for various classifiers of choice
- ===> Now we have the best model! <===
- feed in (fit) *whole* train dataset to our model without splits!

This workflow ensures to get best possible fitted model. With the best scoring output.

##### A few discoveries while testing different classifiers:
 - **Logistic Regresser** is VERY powerful in his simplicity and rapidness. It outputs very *decent scores* (not the best) within seconds.
 - **Tree Ensembels and Gradient Boosted Trees** are very good, but require resource capacity (powerful cpus) and time to display *best scores*.
 - **Kernelized SVC** turned out quite unusable *for dataset with relatively low number of features*. Require tremendous resource capacity, and training can take hours. For example, tuning kernel=rbf clogged my i5 for 6 hours with *medium output*.

## Prepare dataset and Preprocess features
Assuming dataset is ready, trimmed, some nans are been dealt with, required columns are merged and presented in numberic format.

Here are code snippets for minmax preprocessor and splits:

#### fillna


```python
# Example of converting all nans with negative number
df = df.fillna(value=-1)
```

#### Splitter


```python
# Split dataset. This example assumes the last column to be the target label (represented as binary)
from sklearn.model_selection import train_test_split
def split_dataset(df):
    X = df[df.columns.values[:-1]]
    y = df[df.columns.values[-1]]
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    
    return X_train, X_test, y_train, y_test
X_train, X_test, y_train, y_test = split_dataset(train_df)
```

#### min-max scaler


```python
# Preprocess features via min-max scaler
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

## Finding Best Model
This is an most exciting part! Now you'll see classifiers and your models compete for the best scores. Enjoy and choose the best one :)

### Logistic Regression classifier
Let's complete full workflow using Logistic Regression classifier.

First, tune Logistic Regression classifier for best params via Grid Search. Note that scoring is set to 'roc-auc'.


```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

model = LogisticRegression()
grid_params = {'C':[ 0.1, 1, 10, 20, 40, 60, 80], 'penalty': ['l1', 'l2']}

#Note n_jobs=-1 parameter sets GridSearch to use all cores.
grid_model_best = GridSearchCV(model, param_grid=grid_params, n_jobs=-1, cv=3, scoring='roc_auc') # cv=3 and scoring='accuracy' are default
grid_model_best.fit(X_train_scaled, y_train)
```

#### Model Performance Report
Visualize possible performance metrics for a given model


```python
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report

# This a good function that prints out metrics for the model.
def print_report_grid_search(model):
    print('Accuracy of the classifier on training set: {:.2f}'
         .format(model.score(X_train_scaled, y_train)))
    print('Accuracy of the classifier on test set: {:.2f}'
         .format(model.score(X_test_scaled, y_test)))

    y_predicted = model.predict(X_test_scaled)
    print('\nReport\n',classification_report(y_test, y_predicted, target_names=['non-compliant', 'compliant']))

    # Print out best parameters
    print('\nGrid best parameter (max. AUC): ', model.best_params_)
    print('Grid best Train AUC score: ', model.best_score_)
    
    # Calculate and print out auc
    y_score_model = model.decision_function(X_test_scaled)
    roc_auc_model = roc_auc_score(y_test, y_score_model)
    print('\nModel best Test AUC score: {}'.format(roc_auc_model))
```


```python
print_report_grid_search(grid_model_best)
```

#### Best params
Great! We now have a solid working model, potentially the best one.

Write down best params and any other notes found for a given model.
- Features are min-max scaled
- {'C': 35, 'penalty': 'l1'}
- Model best Test AUC score: 0.7861967572842491

### Gradient Boosted Trees
Trees, especially large forests (>1000) require significant computational power, and takes some time.

Also note, that the model is fitted with non-normalized dataset. Building laarge forests more than 1000 might clog the cpu for a few hours. Obviously the larger forests would give slightly better output.


```python
# Import and instantiate classifier 
from sklearn.ensemble import GradientBoostingClassifier
model = GradientBoostingClassifier()

# Define params
param_grid = [{
    'n_estimators':[100, 200, 300],
    'max_depth':[3, 7, 10, 15],
    'random_state':[0],
    'learning_rate':[0.01, 0.1, 1],
    'max_features': ['auto']
}]

# Find model with best params for the scoring goal.
# Note n_jobs=-1 parameter sets GridSearch to use all cores.
grid_model_best = GridSearchCV(model, param_grid = param_grid, n_jobs=-1, cv=3, scoring='roc_auc')

# Fit the model
grid_model_best.fit(X_train, y_train)
```

#### Model Performance Report
Visualize possible performance metrics for a given model


```python
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report

# The function to print report differs from that of used with Logistic Regression. Particularly predict_proba function is used.
def print_report_grid_search_trees(model):
    
    print('Accuracy of the classifier on a training set: {:.2f}'
         .format(model.score(X_train, y_train)))
    print('Accuracy of the classifier on a test set: {:.2f}'
         .format(model.score(X_test, y_test)))

    y_predicted = model.predict(X_test)
    print('\nReport\n',classification_report(y_test, y_predicted, target_names=['non-compliant', 'compliant']))

    # Print out best parameters
    print('\nGrid best parameter (max. AUC): ', model.best_params_)
    print('Grid best Train AUC score: ', model.best_score_)

    # Calculate test AUC score
    y_probability = model.predict_proba(X_test)
    fpr_model, tpr_model, _ = roc_curve(y_test, y_probability[:,1])
    auc_score = auc(fpr_model, tpr_model)

    print('\nModel best Test AUC score: {}'.format(auc_score))
```


```python
print_report_grid_search_trees(grid_model_best)
```

#### Best params
Even better. Trees output outperforms that of Logistics model.

Latest best params for model with 3 folds (with latlon, with dates):
- no min-max feature scaling
- {'learning_rate': 0.1, 'max_depth': 7, 'max_features': 'auto', 'n_estimators': 200, 'random_state': 0}
- Model best Test AUC score: 0.8537540093641313

## Final Model
So, we have our a model of choice! Gradient Boosted Forest!

The last steps would be:
- fitting the model with full train dataset
- use crossvalidation technique for possibly even better output.

#### Prepare datasets, no split
train dataset is not splitted and is used as a whole

test dataset is used as a whole and obviously has noy target labels.


```python
def dont_split_dataset(train_df, test_df):
    X_train = df[df.columns.values[:-1]]
    y_train = df[df.columns.values[-1]]
    X_test = test_df
    return X_train, y_train, X_test

X_train, y_train, X_test = dont_split_dataset(train_df, test_df)
```

#### Fit the Final Model


```python
#The Model Of Choice! Feed in the results.
model = GradientBoostingClassifier(random_state = 0, n_estimators=200, learning_rate=0.1, max_depth=7)
model.fit(X_train, y_train)
```

**Perfect! The model is ready.**


```python
Display performance during training
```


```python
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report

# See performance on train dataset
def print_train_report_tree_model(model):
    print('Accuracy of Decision Tree classifier on training set: {:.2f}'
         .format(model.score(X_train, y_train)))

    y_probability = model.predict_proba(X_train)

    fpr_model, tpr_model, _ = roc_curve(y_train, y_probability[:,1])
    roc_auc_model = auc(fpr_model, tpr_model)
    print('\nModel Train AUC: {}'.format(roc_auc_model))
    
print_train_report_tree_model(model)
```

#### Predict!
That's what a model is built for.

Give her the Test dataset. She'll make predictions in no time.


```python
target_labels = model.predict(X_test)
```

## Thanks
So far it was a nice introduction (or refresher) of Machine Learning workflow to built a best model. Leaving off initial dataset preparation (as this is something that requires separate approach).

Hope it was helpful! Any inputs and comments are welcomed.

Have great modelling!
