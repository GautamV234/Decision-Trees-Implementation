
import math
from operator import index
from re import L
import pandas as pd
import numpy as np
def entropy(Y):
    """
    Function to calculate the entropy 

    Inputs:
    > Y: pd.Series of Labels
    Outpus:
    > Returns the entropy as a float
    """     
    # find all the different outputs
    total = len(Y)
    diff_outputs_dict = {}
    for i,elem in Y.iteritems():
        # print("entropy test 1")
        # print(elem)
        if str(elem) not in diff_outputs_dict.keys():
            diff_outputs_dict[str(elem)] = 1 
        else:
            diff_outputs_dict[str(elem)] = diff_outputs_dict[str(elem)]+1
    ent = 0.0
    for key in diff_outputs_dict.keys():
        val = diff_outputs_dict[str(key)]
        p_i = (val+0.0)/total
        term = -1*p_i*math.log2(p_i)
        ent+=term
    return float(ent)    


def gini_index(Y):
    """
    Function to calculate the gini index

    Inputs:
    > Y: pd.Series of Labels
    Outpus:
    > Returns the gini index as a float
    """
    assert(Y.shape[0] != 0)
    _, counts = np.unique(Y, return_counts=True)
    p = counts/np.sum(counts)
    return 1-np.sum(p**2)

def information_gain(Y, attr,criterion):
    """
    Function to calculate the information gain
    
    Inputs:
    > Y: pd.Series of Labels
    > attr: pd.Series of attribute at which the gain should be calculated
    Outputs:
    > Return the information gain as a float
    """
    assert criterion in ['information_gain', 'gini_index']
    if(criterion=='information_gain'):
        diff_vals = [] 
        for i, elem in attr.iteritems():
            if(str(elem) not in diff_vals):
                diff_vals.append(str(elem))
        redu_entropy = 0.0
        for val in diff_vals:
            storer = []
            sv_len = 0
            for i, elem in attr.iteritems():
                if(str(elem)==val):
                    # get the corresponding label
                    sv_len+=1
                    y_label = Y[i]
                    storer.append(y_label)
            storer_ = pd.Series(storer)
            ent = entropy(storer_)
            redu_entropy += ((sv_len)/len(attr))*ent
        return float(entropy(Y))-float(redu_entropy)
    else:
        diff_vals = [] 
        for i, elem in attr.iteritems():
            if(str(elem) not in diff_vals):
                diff_vals.append(str(elem))
        redu_gini = 0.0
        for val in diff_vals:
            storer = []
            sv_len = 0
            for i, elem in attr.iteritems():
                if(str(elem)==val):
                    # get the corresponding label
                    sv_len+=1
                    y_label = Y[i]
                    storer.append(y_label)
            storer_ = pd.Series(storer)
            ent = gini_index(storer_)
            redu_gini += ((sv_len)/len(attr))*ent
        return float(gini_index(Y))-float(redu_gini)



def reduction_in_variance(Y,attr):
    total_var = Y.var()
    total_len = len(Y)
    diff_vals = [] # diff vals that attr takes like sunny and shit
    for i, elem in attr.iteritems():
        if(str(elem) not in diff_vals):
            diff_vals.append(str(elem))
    redu_var = 0.0
    for val in diff_vals:
        storer = []
        sv_len = 0
        for i,elem in attr.iteritems():
            if(str(elem)==val):
                sv_len+=1
                y_label = Y[i]
                storer.append(y_label)
        storer_ = pd.Series(storer)
        var = storer_.var()
        redu_var+= (((sv_len+0.0)/total_len)*var)
    return float(total_var-redu_var)

    

            




        


