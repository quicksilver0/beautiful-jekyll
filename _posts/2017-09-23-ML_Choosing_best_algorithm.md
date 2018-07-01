---
title: ML Model Evaluation cheatsheet
date: 2017-09-23
tags: cheatsheet ML MachineLearning
category: cheatsheet

layout: post
comments: true
excerpt_separator: <!--more-->
---

A handy cheatsheet on tools for model evaluation. Briefly explains key concepts, and ends up with Powerful GridSearch tool, providing code snippets.

<!--more-->

#### The cheatsheet includes:
[Dummy Classifiers](#dummy)

[Confusion Matrices](#confusion) - example of binary confusion matrices

Evaluating Binary-Classification Models
- [**Metrics**](#metrics) - metrics explained
- [AUC](#AUC) - useful extra metric

Evaluating MultiClassification Models
- [MultiClass Confusion Matrices](#multi-confusion)
- [Multiclass Classification Report](#multi-report)

[**Meet the Grid Searh!**](#grid-search) - the essense of this cheatsheet

## Dummy Classifiers <a id='dummy'></a>
Dummy classifiers completely ignore data. Used for performance comparison with real models. The idea is to get better scoring than that of a dummy.

Types of dummy classifiers:
- most_frequent - always predicts most frequent label
- stratified - random predictions based on data distribution of initial dataset
- uniform - generates prediction uniformly
- constant - predicts user-provided label (can be applicable for F1 score evaluation)


```python
from sklearn.dummy import DummyClassifier

# Negative class (0) is most frequent
dummy_majority = DummyClassifier(strategy = 'most_frequent').fit(X_train, y_train)
# Therefore the dummy 'most_frequent' classifier always predicts class 0
y_dummy_predictions = dummy_majority.predict(X_test)

y_dummy_predictions
```


```python
dummy_majority.score(X_test, y_test)
```


```python
svm = SVC(kernel='linear', C=1).fit(X_train, y_train)
svm.score(X_test, y_test)
```

## Confusion Matrices<a id='confusion'></a>
Provide insight via a visual on True Negative, True Positive, False Negative, False Positive distribution.


```python
# Confusion matrix with dummy classifier

from sklearn.metrics import confusion_matrix

# Negative class (0) is most frequent
dummy_majority = DummyClassifier(strategy = 'most_frequent').fit(X_train, y_train)
y_majority_predicted = dummy_majority.predict(X_test)
confusion = confusion_matrix(y_test, y_majority_predicted)

print('Most frequent class (dummy classifier)\n', confusion)
```


```python
# Confusion matrix with SVC

svm = SVC(kernel='linear', C=1).fit(X_train, y_train)
svm_predicted = svm.predict(X_test)
confusion = confusion_matrix(y_test, svm_predicted)

print('Support vector machine classifier (linear kernel, C=1)\n', confusion)
```

## Evaluating Binary-Classification Models
### Metrics <a id='metrics'></a>
Sometimes performance of the model can be evaluated not only accuracy-wise. But various metrics can be taken into consideration or chosen as primary, according to business goals.

**Accuracy** = TP + TN / (TP + TN + FP + FN) - how accurately True values are predicted

**Pecision** = TP / (TP + FP) - how precise the model is. Good as primary for customer-inclined tasks, where False Positive results should be minimized.

**Recall** = TP / (TP + FN)  Also known as sensitivity, or True Positive Rate - good for medical applications, when missed (False Negative) predictions should be minimized.

**F1** = 2 * Precision * Recall / (Precision + Recall)  - evaluates precision and recall equally (harmonic mean)


```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
# Accuracy = TP + TN / (TP + TN + FP + FN)
# Precision = TP / (TP + FP)
# Recall = TP / (TP + FN)  Also known as sensitivity, or True Positive Rate
# F1 = 2 * Precision * Recall / (Precision + Recall) 
print('Accuracy: {:.2f}'.format(accuracy_score(y_test, tree_predicted)))
print('Precision: {:.2f}'.format(precision_score(y_test, tree_predicted)))
print('Recall: {:.2f}'.format(recall_score(y_test, tree_predicted)))
print('F1: {:.2f}'.format(f1_score(y_test, tree_predicted)))
```


```python
# Combined report with all above metrics
from sklearn.metrics import classification_report
print(classification_report(y_test, tree_predicted, target_names=['not 1', '1']))
```

### Area under the Curve (AUC)<a id='AUC'></a>
Shows relation between True Positive and False Positive rates.
The higher the AUC, obviously the better the model (Recall rate is higher).

It is a curve built upon sweeping through the thresholds provided by a decision function.


```python
from sklearn.metrics import roc_curve, auc
from matplotlib import cm

X_train, X_test, y_train, y_test = train_test_split(X, y_binary_imbalanced, random_state=0)

plt.figure()
plt.xlim([-0.01, 1.00])
plt.ylim([-0.01, 1.01])
for g in [0.01, 0.1, 0.20, 1]:
    svm = SVC(gamma=g).fit(X_train, y_train)
    y_score_svm = svm.decision_function(X_test)
    fpr_svm, tpr_svm, _ = roc_curve(y_test, y_score_svm)
    roc_auc_svm = auc(fpr_svm, tpr_svm)
    accuracy_svm = svm.score(X_test, y_test)
    print("gamma = {:.2f}  accuracy = {:.2f}   AUC = {:.2f}".format(g, accuracy_svm, 
                                                                    roc_auc_svm))
    plt.plot(fpr_svm, tpr_svm, lw=3, alpha=0.7, 
             label='SVM (gamma = {:0.2f}, area = {:0.2f})'.format(g, roc_auc_svm))

plt.xlabel('False Positive Rate', fontsize=16)
plt.ylabel('True Positive Rate (Recall)', fontsize=16)
plt.plot([0, 1], [0, 1], color='k', lw=0.5, linestyle='--')
plt.legend(loc="lower right", fontsize=11)
plt.title('ROC curve: (1-of-10 digits classifier)', fontsize=16)
plt.axes().set_aspect('equal')

plt.show()
```

## Evaluating Multiclassification Models

#### Multiclass Confusion Matrix<a id='multi-confusion'></a>


```python
dataset = load_digits()
X, y = dataset.data, dataset.target
X_train_mc, X_test_mc, y_train_mc, y_test_mc = train_test_split(X, y, random_state=0)


svm = SVC(kernel = 'linear').fit(X_train_mc, y_train_mc)
svm_predicted_mc = svm.predict(X_test_mc)
confusion_mc = confusion_matrix(y_test_mc, svm_predicted_mc)
df_cm = pd.DataFrame(confusion_mc, 
                     index = [i for i in range(0,10)], columns = [i for i in range(0,10)])

plt.figure(figsize=(5.5,4))
sns.heatmap(df_cm, annot=True)
plt.title('SVM Linear Kernel \nAccuracy:{0:.3f}'.format(accuracy_score(y_test_mc, 
                                                                       svm_predicted_mc)))
plt.ylabel('True label')
plt.xlabel('Predicted label')


svm = SVC(kernel = 'rbf').fit(X_train_mc, y_train_mc)
svm_predicted_mc = svm.predict(X_test_mc)
confusion_mc = confusion_matrix(y_test_mc, svm_predicted_mc)
df_cm = pd.DataFrame(confusion_mc, index = [i for i in range(0,10)],
                  columns = [i for i in range(0,10)])

plt.figure(figsize = (5.5,4))
sns.heatmap(df_cm, annot=True)
plt.title('SVM RBF Kernel \nAccuracy:{0:.3f}'.format(accuracy_score(y_test_mc, 
                                                                    svm_predicted_mc)))
plt.ylabel('True label')
plt.xlabel('Predicted label');
```

#### Multi-class classification report<a id='multi-report'></a>
Displays combined metrics report for a model


```python
print(classification_report(y_test_mc, svm_predicted_mc))
```

## Meet the Grid Search!<a id='grid-search'></a>
A powerful toolset to evaluate best parameters for maximization of a given score metrics.

So, it simply sweeps through all possible parameter combinations. And thus allows to choose best model!


```python
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score

dataset = load_digits()
X, y = dataset.data, dataset.target == 1
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

clf = SVC(kernel='rbf')
grid_values = {'gamma': [0.001, 0.01, 0.05, 0.1, 1, 10, 100]}

# default metric to optimize over grid parameters: accuracy
grid_clf_acc = GridSearchCV(clf, param_grid = grid_values)
grid_clf_acc.fit(X_train, y_train)

print('Grid best parameter (max. accuracy): ', grid_clf_acc.best_params_)
print('Grid best score (accuracy): ', grid_clf_acc.best_score_)

# alternative metric to optimize over grid parameters: AUC
grid_clf_auc = GridSearchCV(clf, param_grid = grid_values, scoring = 'roc_auc')
grid_clf_auc.fit(X_train, y_train)
y_decision_fn_scores_auc = grid_clf_auc.decision_function(X_test) 

print('Test set AUC: ', roc_auc_score(y_test, y_decision_fn_scores_auc))
print('Grid best parameter (max. AUC): ', grid_clf_auc.best_params_)
print('Grid best score (AUC): ', grid_clf_auc.best_score_)
```

#### Evaluation metrics supported for model selection


```python
from sklearn.metrics.scorer import SCORERS
print(sorted(list(SCORERS.keys())))
```

    ['accuracy', 'adjusted_rand_score', 'average_precision', 'f1', 'f1_macro', 'f1_micro', 'f1_samples', 'f1_weighted', 'log_loss', 'mean_absolute_error', 'mean_squared_error', 'median_absolute_error', 'neg_log_loss', 'neg_mean_absolute_error', 'neg_mean_squared_error', 'neg_median_absolute_error', 'precision', 'precision_macro', 'precision_micro', 'precision_samples', 'precision_weighted', 'r2', 'recall', 'recall_macro', 'recall_micro', 'recall_samples', 'recall_weighted', 'roc_auc']
​    

## Thanks!

Hope the cheatsheet is helpful. Especially the section on the Grid Search.

*The Article is based on handouts from Specialization on coursera [Applied Data Science with Python](https://www.coursera.org/specializations/data-science-python) University of Michigan.*

Have a great time!
