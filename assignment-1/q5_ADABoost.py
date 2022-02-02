"""
The current code given is for the Assignment 2.
> Classification
> Regression
"""

from sklearn.datasets import make_classification
from unicodedata import name
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from metrics import *

from ensemble.ADABoost import AdaBoostClassifier
# from tree.base import DecisionTree
# Or you could import sklearn DecisionTree
from sklearn.tree import DecisionTreeClassifier
# from linearRegression.linearRegression import LinearRegression

np.random.seed(42)

########### AdaBoostClassifier on Real Input and Discrete Output ###################


N = 30
P = 2
NUM_OP_CLASSES = 2
n_estimators = 5
X = pd.DataFrame(np.abs(np.random.randn(N, P)))
y = pd.Series(np.random.randint(NUM_OP_CLASSES, size=N), dtype="category")

criteria = 'entropy'
tree = DecisionTreeClassifier(criterion=criteria, max_depth=1)
Classifier_AB = AdaBoostClassifier(n_estimators=n_estimators)
Classifier_AB.fit(X, y)
y_hat = Classifier_AB.predict(X)
[fig1, fig2] = Classifier_AB.plot(figure=True,figname="Q5A")
print('Criteria :', criteria)
print('Accuracy: ', accuracy(y_hat, y))
for cls in y.unique():
    print('Precision: ', precision(y_hat, y, cls))
    print('Recall: ', recall(y_hat, y, cls))

print('###########################')
print("PART B")


# AdaBoostClassifier on Classification data set using the entire data set

# Dataset used in q2

X, y = make_classification(
    n_features=2, n_redundant=0, n_informative=2, random_state=42, n_clusters_per_class=2, class_sep=0.5)

frac = 0.6
n = len(X)
splitAt = int(frac*n)
data = pd.DataFrame(X)
data['Y'] = pd.Series(y, dtype='category')
# for shuffling
data = data.sample(frac=1).reset_index(drop=True)
X = data.iloc[:, :-1]
y = data.iloc[:, -1]
X_train, y_train = X.iloc[:splitAt, :].reset_index(drop=True), y[:splitAt].reset_index(drop=True)
X_test, y_test = X.iloc[splitAt+1:,
                        :].reset_index(drop=True), y[splitAt+1:].reset_index(drop=True)
classifier = AdaBoostClassifier(n_estimators=3)
classifier.fit(X_train, y_train)
y_hat = classifier.predict(X_test)
classifier.plot(figure = True,figname="Q5B")         
print('Criteria :', criteria)
print('Accuracy: ', accuracy(y_hat, y_test))
for cls in y.unique():
    print('Precision: ', precision(y_hat, y_test, cls))
    print('Recall: ', recall(y_hat, y_test, cls))