import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from tree.base import DecisionTree

from metrics import *

from tree.randomForest import RandomForestClassifier

# Write code here
X, y = make_classification(
    n_features=2, n_redundant=0, n_informative=2, random_state=42, n_clusters_per_class=2, class_sep=0.5)

np.random.seed(42)


n_estimators = 5
frac = 0.6
n = len(X)
split_idx = int(frac*n)
data = pd.DataFrame(X)
data['Y'] = pd.Series(y, dtype='category')
data = data.sample(frac=1).reset_index(drop=True)
X = data.iloc[:, :-1]
y = data.iloc[:, -1]
X_train, y_train = X.iloc[:split_idx, :].reset_index(
    drop=True), y[:split_idx].reset_index(drop=True)

X_test, y_test = X.iloc[split_idx+1:,
                        :].reset_index(drop=True), y[split_idx+1:].reset_index(drop=True)

classifier = RandomForestClassifier(n_estimators=n_estimators, max_depth=10)

classifier.fit(X_train, y_train)

y_hat = classifier.predict(X_test)

classifier.plot(figure=True,figname = "rfc")

print('Accuracy: ', accuracy(y_hat, y_test))
for cls in y.unique():
    print('Precision: ', precision(y_hat, y_test, cls))
    print('Recall: ', recall(y_hat, y_test, cls))