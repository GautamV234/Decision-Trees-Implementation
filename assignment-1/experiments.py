
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from metrics import *

np.random.seed(42)
num_average_time = 100

# Learn DTs 
# ...
# 
# Function to calculate average time (and std) taken by fit() and predict() for different N and P for 4 different cases of DTs
# ...
# Function to plot the results
# ..
# Function to create fake data (take inspiration from usage.py)
# ...
# ..other functions
def checkTimeForDiscreteInputDiscreteOutput():
    timeForBuilding = []
    timeForPredicting = []
    valuesOfM = []
    valuesOfN = []
    # fixing N
    N = 100
    for M in range(1, 15):
        valuesOfM.append(M)

        X = pd.DataFrame({i: pd.Series(np.random.randn(
            2, size=N), dtype="category") for i in range(M)})
        y = pd.Series(np.random.randn(M, size=N), dtype="category")
        begin = time.time()
        tree = DecisionTree(criterion="information_gain")
        tree.fit(X, y)

        end = time.time()
        timeForBuilding.append((end-begin))

        begin = time.time()

        y_hat = tree.predict(X)

        end = time.time()
        timeForPredicting.append((end-begin))

    plt.plot(valuesOfM, timeForBuilding)
    plt.title(
        "Plot of number of features vs time taken to build the tree at a constant N (time for building)")
    plt.xlabel("Number of features")
    plt.ylabel("Time taken")
    plt.show()

    # fixing M

    # M = 5
    # for N in range(1, 30):
    #     valuesOfN.append(N)

    #     X = pd.DataFrame({i: pd.Series(np.random.randint(
    #         2, size=N), dtype="category") for i in range(M)})
    #     y = pd.Series(np.random.randint(M, size=N), dtype="category")

    #     begin = time.time()

    #     tree = DecisionTree(criterion="information_gain")
    #     tree.fit(X, y)

    #     end = time.time()
    #     timeForBuilding.append((end-begin))

    #     begin = time.time()

    #     y_hat = tree.predict(X)

    #     end = time.time()
    #     timeForPredicting.append((end-begin))

    # plt.plot(valuesOfN, timeForBuilding)
    # plt.title(
    #     "Plot of number of data points vs time taken to build the tree at a constant M (timeForBuilding)")
    # plt.xlabel("Number of instances")
    # plt.ylabel("Time taken")
    # plt.show()

if __name__ =='__main__':
    checkTimeForDiscreteInputDiscreteOutput()
    # SAME CAN BE DONE FOR ALL FOUR MODELS