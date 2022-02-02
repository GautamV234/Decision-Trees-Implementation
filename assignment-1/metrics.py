
from itertools import count
# 
import numpy as np
import pandas as pd
import math


def accuracy(y_hat:pd.Series , y:pd.Series):
    """
    Function to calculate the accuracy

    Inputs:
    > y_hat: pd.Series of predictions
    > y: pd.Series of ground truth
    Output:
    > Returns the accuracy as float
    """
    """
    The following assert checks if sizes of y_hat and y are equal.
    Students are required to add appropriate assert checks at places to
    ensure that the function does not fail in corner cases.
    """
    assert(y_hat.size == y.size)
    y_hat.reset_index(drop=True,inplace=True)
    y.reset_index(drop=True,inplace=True)
    # TODO: Write here
    count =0
    for ind,elem in y_hat.iteritems():
        if y_hat[ind]==y[ind]:
            count+=1
    return float((count+0.0)/len(y_hat))

def precision(y_hat, y, cls):
    """
    Function to calculate the precision

    Inputs:
    > y_hat: pd.Series of predictions
    > y: pd.Series of ground truth
    > cls: The class chosen
    Output:
    > Returns the precision as float
    """
    assert(y_hat.size == y.size)
    assert(y_hat.size > 0)
    y_hat.reset_index(drop=True,inplace=True)
    y.reset_index(drop=True,inplace=True)
    pred_pos = y_hat == cls
    if sum(pred_pos) > 0:
        return (y_hat[pred_pos] == y[pred_pos]).sum()/pred_pos.sum()
    else:
        return None


def recall(y_hat, y, cls):
    """
    Function to calculate the recall

    Inputs:
    > y_hat: pd.Series of predictions
    > y: pd.Series of ground truth
    > cls: The class chosen
    Output:
    > Returns the recall as float
    """
    assert(y_hat.size == y.size)
    assert(y_hat.size > 0)
    y_hat.reset_index(drop=True,inplace=True)
    y.reset_index(drop=True,inplace=True)
    act_pos = y == cls
    if sum(act_pos) > 0:
        return (y_hat[act_pos] == y[act_pos]).sum()/act_pos.sum()
    else:
        return None


def rmse(y_hat, y):
    """
    Function to calculate the root-mean-squared-error(rmse)

    Inputs:
    > y_hat: pd.Series of predictions
    > y: pd.Series of ground truth
    Output:
    > Returns the rmse as float
    """
    # assert y_hat.size==y.size
    y_hat.reset_index(drop=True,inplace=True)
    y.reset_index(drop=True,inplace=True)
    rmse = 0.0
    mse = 0.0
    for ind,elem in y_hat.iteritems():
        v = (y_hat[ind]-y[ind]+0.0)**2
        mse+=v
    if(mse==0.0):
        return 0.0
    rmse = ((mse+0.0)/len(y_hat))**0.5

    return float(rmse)

def mae(y_hat, y):
    """
    Function to calculate the mean-absolute-error(mae)

    Inputs:
    > y_hat: pd.Series of predictions
    > y: pd.Series of ground truth
    Output:
    > Returns the mae as float
    """

    mae = 0.0
    for ind,elem in y_hat.iteritems():
       mae += abs(y_hat[ind]-y[ind])
    return float((mae+0.0)/len(y))
