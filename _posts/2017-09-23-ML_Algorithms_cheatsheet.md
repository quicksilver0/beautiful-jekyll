---
title: Machine Learning Algorithms Cheatsheet
date: 2017-09-23
tags: cheatsheet ML MachineLearning
category: cheatsheet

layout: post
comments: true
excerpt_separator: <!--more-->
---

A useful cheatsheet of Machine Learning Algorithms, with brief description on best application along with code examples.

The cheatsheet lists various models as well as few techniques (at the end) to compliment model performance.

<!--more-->


**k-NN Classifier**
- [k-NN Classifier and Regressor](#k-NN-models) -aka- "k-Nearest Points Classifier".

**Linear models for Regression**
- [Linear Model Classifier and Regressor](#linear-regressor) -aka- "Weighted features Classifier"
- [Ridge Regression](#ridge-regressor) - Linear regressor with regularization.
- [Lasso Regression](#lasso-regressor) - regressor with sparse solution.

**Linear models for Classification**
 - [Logistic Regression](#logistic-regressor)
 - [Linear Support Vector Machines](#linear-SVC) or SVC

**Kernelized Vector Machines**
 - [Kernelized SVC](#kernelized-SVC) - rbf and poly kernels

**Decision Trees**
 - [Tree Classifier](#decision-trees) - building, visualization and plotting important features

**Techniques**
 - [Min-Max Scaler](#min-max-scaler) - normalizer
 - [Polynomial Feature Expansion technique](#poly-feature-expansion) - feature magnifier
 - [Cross Validation](#cross-validation) - train model via several data splits (folds)

## Import and initializations
Import libraries, read in data, split data


```python
%matplotlib notebook
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

#Read in some dataframe with feature columns and a column of labels.
fruits = pd.DataFrame()
X = fruits[['height', 'width', 'mass', 'color_score']]
y = fruits['fruit_label']

#Split the data. By default the split ration is 75%:25%
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
```

## k-NN Models <a id="k-NN-models"></a>
k-NN classifier is a simple and popular algorithm, can be used both for classification and regression solutions. Algorithm builds decision boundaries for classes. The prediction accuracy based on the major vote from the k-nearest points. Number of k-nearest points is decided with the parameter n_neighbors.

The higher the `n_neighbors=k`, the simplier the model.

**Best to apply**: predict objects with low number of features.

### k-NN Classifier


```python
# Create classifier object
from sklearn.neighbors import KNeighborsClassifier
# Note the n_neighbors parameter, which is key on how accurate the classifier would be.
knn = KNeighborsClassifier(n_neighbors = 5)

# Train the classifier (fit the estimator) using the training data
knn.fit(X_train, y_train)

# Check the score
knn.score(X_train, y_train)

# Estimate the accuracy of the classifier on future data, using the test data
knn.score(X_test, y_test)

# Use the trained k-NN classifier model to classify new, previously unseen objects
fruit_prediction = knn.predict([[20, 4.3, 5.5]])
lookup_fruit_name[fruit_prediction[0]]
```

### k-NN Regressor


```python
from sklearn.neighbors import KNeighborsRegressor

X_train, X_test, y_train, y_test = train_test_split(X_R1, y_R1, random_state = 0)

knnreg = KNeighborsRegressor(n_neighbors = 5).fit(X_train, y_train)

print(knnreg.predict(X_test))
print('R-squared test score: {:.3f}'
     .format(knnreg.score(X_test, y_test)))
```

## Linear Models for Regression
Linear models use basic and popular algorithms and are good in solving various regression problems. Generalize better than k-NN.

Linear algorithms base their prediction feature weights computed using different techniques. Algorithms can be controlled using regularization: l1 or l2 (linear and squared) to increase generalization level.

Regularization is a penalty applied to large weights.

### Linear Regression <a id="linear-regressor"></a>
No regularization

**Best chosen**: for datasets with medium amount of features.


```python
# Import parameters and datasets
from sklearn.linear_model import LinearRegression
from adspy_shared_utilities import load_crime_dataset
# Communities and Crime dataset
(X_crime, y_crime) = load_crime_dataset()

# Split
X_train, X_test, y_train, y_test = train_test_split(X_crime, y_crime,
                                                   random_state = 0)
# Train the model
linreg = LinearRegression().fit(X_train, y_train)

# Print result prediction of crime rate per capita in the areas based on all features.
print('Crime dataset')
print('linear model intercept: {}'
     .format(linreg.intercept_))
# Prints weights (coefficient) assigned to each feature
print('linear model coeff:\n{}'
     .format(linreg.coef_))
print('R-squared score (training): {:.3f}'
     .format(linreg.score(X_train, y_train)))
print('R-squared score (test): {:.3f}'
     .format(linreg.score(X_test, y_test)))
```

### Ridge Regression <a id="ridge-regressor"></a>
Linear Regression with regularization.

Parameters:
- alpha=1 - defines regularization level. Higher alpha = higher regularization.

Requires feature normalization (min-max transofrmation 0..1) - (!)only on train data, to avoid data leakage.

Can be applied with polynomial feature expansion.

**Best chosen**: works well with medium and smaller sized datasets with large number of features


```python
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

from sklearn.linear_model import Ridge
X_train, X_test, y_train, y_test = train_test_split(X_crime, y_crime,
                                                   random_state = 0)

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

linridge = Ridge(alpha=20.0).fit(X_train_scaled, y_train)

print('Crime dataset')
print('ridge regression linear model intercept: {}'
     .format(linridge.intercept_))
print('ridge regression linear model coeff:\n{}'
     .format(linridge.coef_))
print('R-squared score (training): {:.3f}'
     .format(linridge.score(X_train_scaled, y_train)))
print('R-squared score (test): {:.3f}'
     .format(linridge.score(X_test_scaled, y_test)))
print('Number of non-zero features: {}'
     .format(np.sum(linridge.coef_ != 0)))
```

### Lasso Regression <a id="lasso-regressor"></a>
Similar to Ridge Regression, but L1(linear) regularization applied. So that weighted sum can get equal to 0, unlike in L2 regularization (with weights squared).

So, essentially Lasso Regression applies "Sparse Solution", i.e. chooses features only of highest importance.

Controls:
- alpha=1 defines regularization level, Higher alpha = higher regularization.

**Best chosen**: when dataset contains a few features with medium/large effect.


```python
from sklearn.linear_model import Lasso
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

X_train, X_test, y_train, y_test = train_test_split(X_crime, y_crime,
                                                   random_state = 0)

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

linlasso = Lasso(alpha=2.0, max_iter = 10000).fit(X_train_scaled, y_train)

print('Crime dataset')
print('lasso regression linear model intercept: {}'
     .format(linlasso.intercept_))
print('lasso regression linear model coeff:\n{}'
     .format(linlasso.coef_))
print('Non-zero features: {}'
     .format(np.sum(linlasso.coef_ != 0)))
print('R-squared score (training): {:.3f}'
     .format(linlasso.score(X_train_scaled, y_train)))
print('R-squared score (test): {:.3f}\n'
     .format(linlasso.score(X_test_scaled, y_test)))
print('Features with non-zero weight (sorted by absolute magnitude):')


# Sorts out and presents features by their magnitude
for e in sorted (list(zip(list(X_crime), linlasso.coef_)),
                key = lambda e: -abs(e[1])):
    if e[1] != 0:
        print('\t{}, {:.3f}'.format(e[0], e[1]))
```

## Linear models for Classification

Linear models require less resources for generalization in comparison to Kernel SVC. And thus can be very powerful for larger datasets

### Logistic Regression <a id="logistic-regressor"></a>
It actually uses binary classification, i.e. comparing this class against all others. Virtually linear regression is used for binary classification under the hood.

Controls:
- C parameter, stands for L2 regularization level. Higher C = less regularization.

**Best Chosen**: popular choice for classification even with large datasets


```python
from sklearn.linear_model import LogisticRegression

X_train, X_test, y_train, y_test = train_test_split(X_cancer, y_cancer, random_state = 0)

clf = LogisticRegression().fit(X_train, y_train)
print('Breast cancer dataset')
print('Accuracy of Logistic regression classifier on training set: {:.2f}'
     .format(clf.score(X_train, y_train)))
print('Accuracy of Logistic regression classifier on test set: {:.2f}'
     .format(clf.score(X_test, y_test)))
```

### Linear Support Vector Machines <a id="linear-SVC"></a>
Is ok to solve binary and multiclassification problems. Binary classification happens under the hood.
Controls:
- C parameter, stands for L2 regularization level. Higher C = less regularization.

**Best chosen**: relatively good with large datasets, fast prediction, sparse data


```python
from sklearn.svm import LinearSVC
X_train, X_test, y_train, y_test = train_test_split(X_cancer, y_cancer, random_state = 0)

clf = LinearSVC().fit(X_train, y_train)
print('Breast cancer dataset')
print('Accuracy of Linear SVC classifier on training set: {:.2f}'
     .format(clf.score(X_train, y_train)))
print('Accuracy of Linear SVC classifier on test set: {:.2f}'
     .format(clf.score(X_test, y_test)))
```

## Kernelized Support Vector Machines <a id="kernelized-SVC"></a>

Implements different functions under the hood, called "kernels". The default is RBF kernel - Radio Basis Function.

Kernel examples: rbf, poly

Controls:
- gamma=1, the higher the gamma the less generatlization.
- C, stands for L2 regularization level. Higher C = less regularization.

**Best choosen** Powerful classifiers, especially when supplemented with correct parameter tuning.



```python
from sklearn.svm import SVC
from adspy_shared_utilities import plot_class_regions_for_classifier

X_train, X_test, y_train, y_test = train_test_split(X_D2, y_D2, random_state = 0)

# The default SVC kernel is radial basis function (RBF)
plot_class_regions_for_classifier(SVC().fit(X_train, y_train),
                                 X_train, y_train, None, None,
                                 'Support Vector Classifier: RBF kernel')

# Compare decision boundries with polynomial kernel, degree = 3
plot_class_regions_for_classifier(SVC(kernel = 'poly', degree = 3)
                                 .fit(X_train, y_train), X_train,
                                 y_train, None, None,
                                 'Support Vector Classifier: Polynomial kernel, degree = 3')
```

Example of SVC with min-max features preprocessed.


```python
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

clf = SVC(C=10, gamma=0.1).fit(X_train_scaled, y_train)
print('Breast cancer dataset (normalized with MinMax scaling)')
print('RBF-kernel SVC (with MinMax scaling) training set accuracy: {:.2f}'
     .format(clf.score(X_train_scaled, y_train)))
print('RBF-kernel SVC (with MinMax scaling) test set accuracy: {:.2f}'
     .format(clf.score(X_test_scaled, y_test)))
```

### Decision Trees <a id="decision-trees"></a>
Decision Tree Classifier Builds a structure of features with highest-to-lowest weight features using split-game. Individual decision trees tend to overfit.

Parameters:
 - max_depth - decision tree depth, for generalization purposes and avoid overfitting

**Best chosen**: great for classification, especially when used in ensembles. Good with medium number of features.


```python
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from adspy_shared_utilities import plot_decision_tree
from sklearn.model_selection import train_test_split


iris = load_iris()

X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, random_state = 3)
clf = DecisionTreeClassifier(max_depth = 3).fit(X_train, y_train)

print('Accuracy of Decision Tree classifier on training set: {:.2f}'
     .format(clf.score(X_train, y_train)))
print('Accuracy of Decision Tree classifier on test set: {:.2f}'
     .format(clf.score(X_test, y_test)))
```

#### Visualize Decision Trees


```python
plot_decision_tree(clf, iris.feature_names, iris.target_names)
```

#### Visualize Feature Importances


```python
from adspy_shared_utilities import plot_feature_importances

plt.figure(figsize=(10,4), dpi=80)
plot_feature_importances(clf, iris.feature_names)
plt.show()

print('Feature importances: {}'.format(clf.feature_importances_))
```

## Techniques
Some techniques that complement different models:

- MinMaxScaler - normalizer
- Polynomial Feature Expansion - magnifies features
- Cross Validation - performs several training fold

### MinMax Scaler <a id="min-max-scaler"></a>
Normalizes features.

Best applied along with Regularized Linear Regression models (Ridge) and with Kernelized SVC.


```python
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

from sklearn.linear_model import Ridge
X_train, X_test, y_train, y_test = train_test_split(X_crime, y_crime,
                                                   random_state = 0)

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

### Polynomial feature expansion Technique <a id="poly-feature-expansion"></a>
Allows to magnify features.

Use polynomial features in combination with regression that has a regularization penalty, like **ridge regression**. Applied on initial dataset.


```python
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures


X_train, X_test, y_train, y_test = train_test_split(X_F1, y_F1,
                                                   random_state = 0)
linreg = LinearRegression().fit(X_train, y_train)
print('linear model coeff (w): {}'
     .format(linreg.coef_))
print('linear model intercept (b): {:.3f}'
     .format(linreg.intercept_))
print('R-squared score (training): {:.3f}'
     .format(linreg.score(X_train, y_train)))
print('R-squared score (test): {:.3f}'
     .format(linreg.score(X_test, y_test)))

print('\nNow we transform the original input data to add\n\
polynomial features up to degree 2 (quadratic)\n')
poly = PolynomialFeatures(degree=2)
X_F1_poly = poly.fit_transform(X_F1)
#print('printing original X_F1 of len {}: {}'.format(len(X_F1),X_F1))
#print('printing poly-transformed X_F1 of len {}: {}'.format(len(X_F1_poly)X_F1_poly))
X_train, X_test, y_train, y_test = train_test_split(X_F1_poly, y_F1,
                                                   random_state = 0)
linreg = LinearRegression().fit(X_train, y_train)
print('(poly deg 2) linear model coeff (w):\n{}'
     .format(linreg.coef_))
print('(poly deg 2) linear model intercept (b): {:.3f}'
     .format(linreg.intercept_))
print('(poly deg 2) R-squared score (training): {:.3f}'
     .format(linreg.score(X_train, y_train)))
print('(poly deg 2) R-squared score (test): {:.3f}\n'
     .format(linreg.score(X_test, y_test)))

print('\nAddition of many polynomial features often leads to\n\
overfitting, so we often use polynomial features in combination\n\
with regression that has a regularization penalty, like ridge\n\
regression.\n')

X_train, X_test, y_train, y_test = train_test_split(X_F1_poly, y_F1,
                                                   random_state = 0)
linreg = Ridge().fit(X_train, y_train)

print('(poly deg 2 + ridge) linear model coeff (w):\n{}'
     .format(linreg.coef_))
print('(poly deg 2 + ridge) linear model intercept (b): {:.3f}'
     .format(linreg.intercept_))
print('(poly deg 2 + ridge) R-squared score (training): {:.3f}'
     .format(linreg.score(X_train, y_train)))
print('(poly deg 2 + ridge) R-squared score (test): {:.3f}'
     .format(linreg.score(X_test, y_test)))
```

### Cross Validation Technique <a id="cross-validation"></a>
Allows to reach better scores, by additional splits of the dataset (folds). The scores can be calculated as a mean of scores from each fold.


```python
from sklearn.model_selection import cross_val_score

clf = KNeighborsClassifier(n_neighbors = 5)
X = X_fruits_2d.as_matrix()
y = y_fruits_2d.as_matrix()
cv_scores = cross_val_score(clf, X, y)

print('Cross-validation scores (3-fold):', cv_scores)
print('Mean cross-validation score (3-fold): {:.3f}'
     .format(np.mean(cv_scores)))
```

#### A note on performing cross-validation for more advanced scenarios.

In some cases (e.g. when feature values have very different ranges), we've seen the need to scale or normalize the training and test sets before use with a classifier. The proper way to do cross-validation when you need to scale the data is *not* to scale the entire dataset with a single transform, since this will indirectly leak information into the training data about the whole dataset, including the test data (see the lecture on data leakage later in the course).  Instead, scaling/normalizing must be computed and applied for each cross-validation fold separately.  To do this, the easiest way in scikit-learn is to use *pipelines*.  While these are beyond the scope of this course, further information is available in the scikit-learn documentation here:

http://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html

or the Pipeline section in the recommended textbook: Introduction to Machine Learning with Python by Andreas C. Müller and Sarah Guido (O'Reilly Media).

## Thanks
Great! Hope this cheatsheet was helpful.

Based on handouts from Specialization on coursera [Applied Data Science with Python](https://www.coursera.org/specializations/data-science-python) University of Michigan.

Have a nice day ;)
