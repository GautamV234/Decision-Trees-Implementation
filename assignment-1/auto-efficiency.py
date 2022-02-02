import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from tree.base import DecisionTree
from metrics import *
from sklearn import preprocessing
from sklearn import tree

np.random.seed(42)

data = pd.read_fwf('auto-mpg.data', names=["mpg", "cylinders", "displacement",
                                                        "horsepower", "weight", "accelaration", "model year", "origin", "car name"])

# CLEANUP
# removing unclear rows
data.drop(data.index[data['horsepower'] == "?"], inplace=True)
# removing unclear columns
data.drop('car name', axis=1, inplace=True)
for i in data.columns:
    data[i] = data[i].astype('float64')
data.reset_index(drop=True, inplace=True)

frac = 0.7
n = int(frac*len(data))
X_train, X_test, y_train, y_test = data.iloc[:n,1:], data.iloc[n+1:,1:], data.iloc[:n,0], data.iloc[n+1:,0]

tree0 = DecisionTree(criterion='information_gain', max_depth=5)  # Split based on Inf. Gain
tree0.fit(X_train, y_train)
y_hat = tree0.predict(X_test)
tree0.plot()
print('RMSE: ', rmse(y_hat, y_test))
print('MAE: ', mae(y_hat, y_test))
print('---------------------')

tree1 = DecisionTreeRegressor(max_depth=5)
tree1.fit(X_train, y_train)
text_representation = tree.export_text(tree1)
print(text_representation)
yy_hat = tree1.predict(X_test)
yy_hat = pd.Series(yy_hat)
print('RMSE: ', rmse(yy_hat, y_test))
print('MAE: ', mae(yy_hat, y_test))