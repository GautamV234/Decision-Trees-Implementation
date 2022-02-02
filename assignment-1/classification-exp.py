import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

from metrics import *
from tree.base import DecisionTree

np.random.seed(42)

# Read dataset
# ...
#
X, y = make_classification(
    n_features=2, n_redundant=0, n_informative=2, random_state=1, n_clusters_per_class=2, class_sep=0.5)

# For plotting
plt.scatter(X[:, 0], X[:, 1], c=y)

dataset = pd.DataFrame(X)
dataset['Y'] = pd.Series(y, dtype='category')
X = dataset.iloc[:, :-1]
y = dataset['Y']
frac = 0.7
split_val = int(frac*len(dataset))

train_data, test_data = dataset.iloc[:split_val, :], dataset.iloc[split_val+1:, :]
X_train = train_data.iloc[:,:-1]
y_train = train_data.iloc[:,-1]
X_test = test_data.iloc[:,:-1]
y_test = test_data.iloc[:,-1]



tree = DecisionTree(criterion="information_gain")  # Split based on Inf. Gain
tree.fit(X_train,y_train)
y_hat = tree.predict(X_test)

tree.plot()
print('Accuracy: ', accuracy(y_hat,  y_test))
for cls in test_data.iloc[:, -1].unique():
    print('Precision: ', precision(y_hat, test_data.iloc[:, -1], cls))
    print('Recall: ', recall(y_hat, test_data.iloc[:, -1], cls))


# Optimizing for depth
def find_best_depth(X, y, folds=5, depths=[1]):
    assert(len(X) == len(y))
    assert(len(X) > 0)

    max_depth = max(depths)
    trees = {}
    accuracies = {}
    # similar to 100/4 = 25, size of data in each fold
    sub_size = int(len(X)//folds)

    for fold in range(folds):
        # Training a seperate model for each fold
        # getting the first set of data
        sub_data_inddexes = range(fold*sub_size, (fold+1)*sub_size)
        c_fold = []
        for i in range(len(X)):
            if(i in sub_data_inddexes):
                c_fold.append(True)
            else:
                c_fold.append(False)
        c_fold = pd.Series(c_fold)
            
        X_train = X[~c_fold].reset_index(
            drop=True)
        y_train = y[~c_fold].reset_index(drop=True)
        X_test = X[c_fold].reset_index(drop=True)
        y_test = y[c_fold].reset_index(drop=True)

        tree = DecisionTree(max_depth=max_depth)
        tree.fit(X_train, y_train)
        trees[fold+1] = tree

        for depth in depths:
            print("Depth is "+str(depth))
            tree = DecisionTree(max_depth=depth)
            tree.fit(X_train, y_train)
            y_hat = tree.predict(X_test)
            if fold+1 in accuracies:
                accuracies[fold+1][depth] = accuracy(y_hat, y_test)
            else:
                accuracies[fold+1] = {depth: accuracy(y_hat, y_test)}

    accuracies = pd.DataFrame(accuracies).transpose()
    accuracies.index.name = "Fold ID"
    accuracies.loc["mean"] = accuracies.mean(axis = 'rows')
    best_mean_acc = accuracies.loc["mean"].max()
    best_depth = accuracies.loc["mean"].idxmax()
    print(accuracies)
    print("Best Mean Accuracy ===>" + str(best_mean_acc))
    print("Optimum Depth ===>"+str(best_depth))


find_best_depth(X, y, folds=5, depths=list(range(1, 11)))