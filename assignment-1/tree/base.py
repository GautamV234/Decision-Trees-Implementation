"""
The current code given is for the Assignment 1.
You will be expected to use this to make trees for:
> discrete input, discrete output
> real input, real output
> real input, discrete output
> discrete input, real output
"""


from cgitb import reset
from dataclasses import replace
from numbers import Real
import random
from sys import flags
import numpy as np
import pandas as pd
from .utils import entropy, information_gain, gini_index, reduction_in_variance
import math

np.random.seed(42)


class Real_Node():
    def __init__(self, identifier: int, lchild, rchild, split_val, attribute, value, depth, isLeaf: bool):
        self.lchild = lchild
        self.rchild = rchild
        self.split_val = split_val
        self.attribute = attribute
        self.value = value
        self.depth = depth
        self.identifier = identifier
        self.isLeaf = isLeaf

    def add_lchild(self, lchild):
        self.lchild = lchild

    def add_rchild(self, rchild):
        self.rchild = rchild

    def set_attribute(self, attribute):
        self.attribute = attribute

    def set_split_val(self, split_val):
        self.split_val = split_val

    def set_isLeafFalse(self):
        self.isLeaf = False


class Discrete_Node():
    def __init__(self, identifier: int, children: list, attribute, value, depth, isLeaf: bool, answer=None, ):
        # children is the list subnodes
        # answer means it is a leaf node , None means it has children and is not a leaf
        # attribute means it is split over which attribute
        # value means the value of the attribute that this node represents
        self.children = children
        self.answer = answer
        self.attribute = attribute
        self.identifier = identifier
        self.value = value
        self.depth = depth
        self.isLeaf = isLeaf

    def add_node(self, node_to_add):
        self.children.append(node_to_add)

    def set_attr(self, attribute):
        self.attribute = attribute

    def set_answer(self, answer):
        self.answer = answer

    def set_isLeaf(self, isLeaf):
        self.isLeaf = isLeaf


class DecisionTree():
    def __init__(self, criterion='information_gain',max_depth = 10):
        # assert criterion in ['information_gain', 'gini_index']
        self.criterion = criterion
        self.max_depth = max_depth
        self.tree = None
        self.is_discrete_x = True
        self.is_discrete_y = True

    def fit(self, X:pd.DataFrame, y:pd.Series):
        l = X.dtypes.tolist()
        if(np.float64 in l):
            self.is_discrete_x = False
        y_dtype = y.dtype
        if(y_dtype ==np.float64):
            self.is_discrete_y = False
        
        if(self.is_discrete_x==True and self.is_discrete_y==True):
            print("Calling model 1")
            # run dido model
            mode = y.mode(dropna=True)
            mode = y[0]
            root = Discrete_Node(identifier=random.randrange(-1000, 1000),
                             children=[], attribute=-1, value=mode, depth=0, answer=None, isLeaf=True)
            self.DT_di_do(X=X, y=y, tree=root)
            self.tree = root
        
        elif(self.is_discrete_x==True and self.is_discrete_y == False):
            print("Calling model 2")
            # run diro model
            mean  = y.mean()
            root = Discrete_Node(identifier=random.randrange(-1000, 1000),
                             children=[], attribute=-1, value=mean, depth=0, answer=None, isLeaf=True)
            self.DT_di_ro(X=X,y=y,tree=root)
            self.tree = root

        elif (self.is_discrete_x == False and self.is_discrete_y == False):
            print("Calling model 3")
            # run riro model
            mean_val = y.mean(skipna=True)
            mode = y.mode()[0]
            root = Real_Node(identifier=random.randrange(-1000, 1000), lchild=None,
                         rchild=None, split_val=-1, attribute=-1, value=mean_val, depth=0, isLeaf=True)
            self.DT_ri_ro(X, y, root)
            self.tree = root
        elif(self.is_discrete_x== False and self.is_discrete_y==True):
            print("Calling model 4")
            # run rido model
            mode = y.mode(dropna=True)
            mode = y[0]
            root = Real_Node(identifier=random.randrange(-1000, 1000), lchild=None,
                         rchild=None, split_val=-1, attribute=-1, value=mode, depth=0, isLeaf=True)
            self.DT_ri_do(X=X, y=y, tree=root)
            self.tree = root

    def predict(self, X: pd.DataFrame):
        if(self.is_discrete_x):
            predictions = []
            num_instance,num_attributes = X.shape
            for i in range (num_instance):
                pred = self.pred_dido(self.tree,X.iloc[i,:])
                predictions.append(pred)
            return pd.Series(predictions)
        else:
            print("pred_riro called")
            preds = []
            # X.reset_index(inplace=True)
            self.pred_riro(0, self.tree, X, preds)
            return pd.Series(preds)

    def plot(self):
        """
        Function to plot the tree

        Output Example:
        ?(X1 > 4)
            Y: ?(X2 > 7)
                Y: Class A
                N: Class B
            N: Class C
        Where Y => Yes and N => No
        """
        if(self.is_discrete_x == True):
            # Discrete_Node
            self.disc_plot(node=self.tree)

            pass
        elif(self.is_discrete_x== False):
            # Real_Node
            self.real_plot(self.tree)

    def disc_plot(self, node:Discrete_Node,spacing=" "):
        if(node.isLeaf):
            # print("ans ",end="")
            print(node.answer)
            return
        print('?(X'+str(node.attribute)+')')
        for child in node.children:
            print(spacing+ '=='+str(child.value)+':',end="")
            self.disc_plot(child,spacing=spacing+ " ")


    def real_plot(self , node:Real_Node,spacing=" "):
        if(node.isLeaf):
            print(node.value)
            return
        print('?(X'+str(node.attribute)+'> '+str(node.split_val)+')\n'+spacing+ 'Y: ',end="")
        self.real_plot(node.rchild,spacing=spacing+" ")
        print(spacing+ 'N: ',end="")
        self.real_plot(node.lchild,spacing=spacing+" ")
        

    def check_if_pure(self, y: pd.Series):
        if len(np.unique(y)) == 1:  
            return True
        return False

    def create_subtables(self, X: pd.DataFrame, y, attribute, valOfAttribute):
        big_table = X.merge(
            y.rename('Test'), left_index=True, right_index=True)
        sub_X = big_table.loc[X[attribute] == valOfAttribute]
        sub_X.drop(attribute, inplace=True, axis=1)
        sub_y = sub_X['Test']
        sub_X.drop('Test', axis=1, inplace=True)
        return sub_X, sub_y

    def DT_di_do(self, X: pd.DataFrame, y: pd.Series, tree: Discrete_Node):
        """ 
        Function to train and construct the decision tree
        Inputs:
        X: pd.DataFrame with rows as samples and columns as features (shape of X is N X P) where N is the number of samples and P is the number of columns.
        y: pd.Series with rows corresponding to output variable (shape of Y is N)
        """
        assert self.criterion in ['information_gain', 'gini_index']
        if(self.check_if_pure(y) == True):
            tree.set_answer(y.mode()[0])
            tree.set_attr(-2)
            return
        attr_list = X.columns.tolist()
        if(len(attr_list) == 0):
            # all attributes got over
            mode_val = y.mode(dropna=True)
            mode_val = mode_val[0]
            tree.set_answer(mode_val)
            tree.set_attr(-2)
            return
        if(tree.depth == self.max_depth-1):
            # make prediction as the most frequent of the current values in Y
            mode_val = y.mode(dropna=True)
            mode_val = mode_val[0]
            tree.set_answer(mode_val)
            tree.set_attr(-2)
            return
        # NOW IT MEANS IT HAS CHILDREN
        attr_list = X.columns.tolist()
        max_inf_gain = -100
        max_inf_gain_attr = ""
        for attr in attr_list:
            columnOfAttr = X[attr]
            # currently only using information gain for splitting
            inf_gain = information_gain(y, columnOfAttr,self.criterion)
            if(inf_gain > max_inf_gain):
                max_inf_gain = inf_gain
                max_inf_gain_attr = attr
        tree.set_attr(max_inf_gain_attr)
        # Setting leaf nature as false as it has children
        tree.set_isLeaf(False)
        unique_vals = X[max_inf_gain_attr].unique().tolist()
        # adding all subnodes
        for unique_val in unique_vals:
            newNode = Discrete_Node(identifier=random.randrange(-1000, 1000), children=[],
                                    attribute=-1, value=unique_val, depth=tree.depth+1, answer=None, isLeaf=True)
            tree.add_node(newNode)
        i = 0
        while(i < len(unique_vals)):
            sub_X, sub_Y = self.create_subtables(
                X=X, y=y, attribute=max_inf_gain_attr, valOfAttribute=unique_vals[i])
            self.DT_di_do(sub_X, sub_Y, tree=tree.children[i])
            i += 1

    def pred_dido(self, tree: Discrete_Node, X):
        if(tree.isLeaf == True):
            return tree.answer
        else:
            attr = tree.attribute
            example = X
            val = example[attr]
            flag = False
            for child in tree.children:
                if(child.value == val):
                    flag = True
                    return self.pred_dido(child, X)
            if(flag == False):
                return 1


    def DT_di_ro(self, X: pd.DataFrame, y, tree: Discrete_Node):
        if(self.check_if_pure(y) == True):
            tree.set_answer(y.mean())
            # tree.set_answer(y.mode()[0])
            tree.set_attr(-2)
            return
        attr_list = X.columns.tolist()
        if(len(attr_list) == 0):
            tree.set_answer(y.mean())
            tree.set_attr(-2)
            return
        if(tree.depth == self.max_depth-1):
            # make prediction as the most frequent of the current values in Y
            tree.answer(y.mean())
            tree.set_attr(-2)
            return
        # NOW IT MEANS IT HAS CHILDREN
        attr_list = X.columns.tolist()
        max_inf_gain = -100
        max_inf_gain_attr = ""
        for attr in attr_list:
            columnOfAttr = X[attr]
            # currently only using information gain for splitting
            inf_gain = reduction_in_variance(y, columnOfAttr)
            if(inf_gain > max_inf_gain):
                max_inf_gain = inf_gain
                max_inf_gain_attr = attr
        tree.set_attr(max_inf_gain_attr)
        # Setting leaf nature as false as it has children
        tree.set_isLeaf(False)
        unique_vals = X[max_inf_gain_attr].unique().tolist()
        # adding all subnodes
        for unique_val in unique_vals:
            newNode = Discrete_Node(identifier=random.randrange(-1000, 1000), children=[],
                                    attribute=-1, value=unique_val, depth=tree.depth+1, answer=None, isLeaf=True)
            tree.add_node(newNode)
        i = 0
        while(i < len(unique_vals)):
            sub_X, sub_Y = self.create_subtables(
                X=X, y=y, attribute=max_inf_gain_attr, valOfAttribute=unique_vals[i])
            self.DT_di_do(sub_X, sub_Y, tree=tree.children[i])
            i += 1

    """
    Implementing Real input cases below
    """

    def mse(self, y: pd.Series, mean_val):
        a = np.sum(np.square(np.abs(y-mean_val)))
        if(a ==0.0):
            return 0.0
        error = (a)/len(y)
        return error

    # used for real input real output
    def finder(self, X: pd.DataFrame, y: pd.Series):
        # print("entred finder")
        attrs = X.columns.tolist()
        final_left_val = 0.0  # mean of all the y vals in the left
        final_right_val = 0.0  # mean of all the y vals in the right
        final_split_val = 0.0  # x value upon which the splitting is happening
        # value of the attribute/feature from which the val is selected
        final_selected_attr = 'GAMMA'
        least_weighted_mean = math.inf
        for attr in attrs:
            # print("Current attribute -->"+str(attr))
            big_table = X.merge(
                y.rename('Test'), left_index=True, right_index=True)
            sub_X = big_table.sort_values(by=attr)
            sub_y = sub_X['Test']
            sub_X = sub_X[attr]
            sub_X.reset_index(inplace=True, drop=True)
            sub_y.reset_index(inplace=True, drop=True)
            for ind in range(len(sub_X)):
                if(ind == len(sub_X)-1):
                    # print("Caught you")
                    break
                mean = (sub_X[ind]+sub_X[ind+1]+0.0)/2
                left_y = sub_y[0:ind+1]
                right_y = sub_y[ind+1:]
                l_mean = left_y.mean()
                r_mean = right_y.mean()
                left_mse = self.mse(y=left_y, mean_val=l_mean)
                right_mse = self.mse(y=right_y, mean_val=r_mean)
                weighted_total = left_mse*len(left_y) + right_mse*len(right_y)
                if(weighted_total < least_weighted_mean):
                    least_weighted_mean = weighted_total
                    final_split_val = mean
                    final_selected_attr = attr
                    final_left_val = left_y.mean()
                    final_right_val = right_y.mean()
        return final_selected_attr, final_split_val, final_left_val, final_right_val

    def subtables_ri_ro(self, X: pd.DataFrame, y: pd.Series, split_val, attribute, child: str):
        assert child in ['left', 'right']
        columns = X.columns.tolist()
        if(child == 'left'):
            merged = X.merge(y.rename('Test'),
                             left_index=True, right_index=True)
            l_X = merged.loc[X[attribute] <= split_val]
            l_Y = l_X['Test']
            l_X.drop('Test', axis=1, inplace=True)
            return l_X, l_Y
        elif(child == 'right'):
            merged = X.merge(y.rename('Test'),
                             left_index=True, right_index=True)
            r_X = merged.loc[X[attribute] > split_val]
            r_Y = r_X['Test']
            r_X.drop('Test', axis=1, inplace=True)
            return r_X, r_Y

    def DT_ri_ro(self, X: pd.DataFrame, y: pd.Series, tree: Real_Node):
        # if(len(y.unique())==1):
        #     cy = y.copy()
        #     cy.reset_index(replace=True,inplace=True)
        #     tree.set_attribute(-1)
        #     tree.value = cy[0]
        #     return
        # Here we use reduction in rmse as our metric to find
        # the predicted value is the mean of the values of all the points in that section
        final_selected_attr, final_split_val, final_left_val, final_right_val = self.finder(
            X, y)
        # final_selected_attr = feature that is been selected
        # final split_val = value upon which the splitting is occurring
        # final_left_val =  mean of all the y values on the left of the split_val
        # final_right_val = mean of all the y values on the right of the split_val
        tree.set_attribute(final_selected_attr)
        tree.set_split_val(split_val=final_split_val)
        # checking if max depth is reached
        if(self.mse(y, y.mean()) == 0):
            return
        if(tree.depth == self.max_depth):
            return
        # finding subtables for left subtree
        l_X, l_Y = self.subtables_ri_ro(
            X=X, y=y, attribute=final_selected_attr, split_val=final_split_val, child='left',)
        lchild = Real_Node(identifier=random.randrange(-1000, 1000), lchild=None, rchild=None, split_val=-1,
                           attribute=-1, value=final_left_val, depth=tree.depth+1, isLeaf=True)
        # finding subtables for right subtree
        r_X, r_Y = self.subtables_ri_ro(
            X=X, y=y, attribute=final_selected_attr, split_val=final_split_val, child='right',)
        rchild = Real_Node(identifier=random.randrange(-1000, 1000), lchild=None, rchild=None, split_val=-1,
                           attribute=-1, value=final_right_val, depth=tree.depth+1, isLeaf=True)
        # adding child nodes
        tree.add_lchild(lchild=lchild)
        tree.add_rchild(rchild=rchild)
        tree.set_isLeafFalse()
        self.DT_ri_ro(l_X, l_Y, tree.lchild)
        self.DT_ri_ro(r_X, r_Y, tree.rchild)

   # used for real input discrete output
    def finder_rido(self, X: pd.DataFrame, y: pd.Series,crieterion='information_gain'):
        if crieterion =='information_gain':
            total_len = len(y)
            total_entropy = entropy(y)
            max_inf_gain = -100
            attrs = X.columns.tolist()
            final_left_val = 0.0  # mean of all the y vals in the left
            final_right_val = 0.0  # mean of all the y vals in the right
            final_split_val = 0.0  # x value upon which the splitting is happening
            # value of the attribute/feature from which the val is selected
            final_selected_attr = "gamma"
            for attr in attrs:
                big_table = X.merge(
                    y.rename('Test'), left_index=True, right_index=True)
                sub_X = big_table.sort_values(by=attr)
                sub_y = sub_X['Test']
                sub_X = sub_X[attr]
                sub_X.reset_index(inplace=True, drop=True)
                sub_y.reset_index(inplace=True, drop=True)
                for ind in range(len(sub_X)):
                    if(ind == len(sub_X)-1):
                        break
                    split_val = (sub_X[ind]+sub_X[ind+1]+0.0)/2
                    left_y = sub_y[0:ind+1]
                    right_y = sub_y[ind+1:]
                    left_entropy = entropy(Y=left_y)
                    right_entropy = entropy(Y=right_y)   
                    inf_gain = total_entropy - (((len(left_y)/total_len)*left_entropy) +((len(right_y)/total_len)*right_entropy))
                    if(inf_gain > max_inf_gain):
                        max_inf_gain = inf_gain
                        final_selected_attr = attr
                        final_split_val = split_val
                        l_mode = left_y.mode(dropna=True)[0]
                        r_mode = right_y.mode(dropna=True)[0]
                        final_left_val = l_mode
                        final_right_val = r_mode
            return final_selected_attr, final_split_val, final_left_val, final_right_val
        else:
            total_len = len(y)
            total_gini = gini_index(y)
            max_inf_gain = -100
            attrs = X.columns.tolist()
            final_left_val = 0.0  # mean of all the y vals in the left
            final_right_val = 0.0  # mean of all the y vals in the right
            final_split_val = 0.0  # x value upon which the splitting is happening
            # value of the attribute/feature from which the val is selected
            final_selected_attr = "gamma"
            for attr in attrs:
                big_table = X.merge(
                    y.rename('Test'), left_index=True, right_index=True)
                sub_X = big_table.sort_values(by=attr)
                sub_y = sub_X['Test']
                sub_X = sub_X[attr]
                sub_X.reset_index(inplace=True, drop=True)
                sub_y.reset_index(inplace=True, drop=True)
                for ind in range(len(sub_X)):
                    if(ind == len(sub_X)-1):
                        break
                    split_val = (sub_X[ind]+sub_X[ind+1]+0.0)/2
                    left_y = sub_y[0:ind+1]
                    right_y = sub_y[ind+1:]
                    left_gini = gini_index(Y=left_y)
                    right_gini = gini_index(Y=right_y)   
                    inf_gain = total_gini - (((len(left_y)/total_len)*left_gini) +((len(right_y)/total_len)*right_gini))
                    if(inf_gain > max_inf_gain):
                        max_inf_gain = inf_gain
                        final_selected_attr = attr
                        final_split_val = split_val
                        l_mode = left_y.mode(dropna=True)[0]
                        r_mode = right_y.mode(dropna=True)[0]
                        final_left_val = l_mode
                        final_right_val = r_mode
            return final_selected_attr, final_split_val, final_left_val, final_right_val
            

    def DT_ri_do(self, X: pd.DataFrame, y: pd.Series, tree: Real_Node):
        # Here we use reduction in rmse as our metric to find
        # the predicted value is the mean of the values of all the points in that section
        # checking if only data point is left in the dataset
        assert self.criterion in ['information_gain', 'gini_index']
        if(len(X.index) == 1):
            tree.set_attribute(-1)
            tree.set_split_val(-1)
            t = y.mode()[0]
            tree.value = t
            return

        final_selected_attr, final_split_val, final_left_val, final_right_val = self.finder_rido(
            X, y)
        tree.set_attribute(final_selected_attr)
        tree.set_split_val(split_val=final_split_val)
        # checking if max depth is reached
        if(len(y.unique()) == 1):
            return
        if(tree.depth == self.max_depth):
            return
        # finding subtables for left subtree
        l_X, l_Y = self.subtables_ri_ro(
            X=X, y=y, attribute=final_selected_attr, split_val=final_split_val, child='left',)
        lchild = Real_Node(identifier=random.randrange(-1000, 1000), lchild=None, rchild=None, split_val=-1,
                           attribute=-1, value=final_left_val, depth=tree.depth+1, isLeaf=True)
        # finding subtables for right subtree
        r_X, r_Y = self.subtables_ri_ro(
            X=X, y=y, attribute=final_selected_attr, split_val=final_split_val, child='right',)
        rchild = Real_Node(identifier=random.randrange(-1000, 1000), lchild=None, rchild=None, split_val=-1,
                           attribute=-1, value=final_right_val, depth=tree.depth+1, isLeaf=True)
        # adding child nodes
        tree.add_lchild(lchild=lchild)
        tree.add_rchild(rchild=rchild)
        tree.set_isLeafFalse()
        self.DT_ri_do(l_X, l_Y, tree.lchild)
        self.DT_ri_do(r_X, r_Y, tree.rchild)

    def pred_riro(self, current_example_ind: int, tree: Real_Node, X: pd.DataFrame, prediction_arr: list):
        # print("pred began")
        if(tree.rchild is None and tree.lchild is None and tree.isLeaf == False):
            # print("AAA")
            return
        if(current_example_ind == len(X)):
            # print("answer for all examples is found")
            # print("BBB")
            return
        # print("K")
        if(tree.isLeaf):
            # print("CCC")
            prediction_arr.append(tree.value)
            self.pred_riro(current_example_ind+1, self.tree, X, prediction_arr)
            return
        else:
            # print("DDD")
            # check the attribute stored in root
            attr = tree.attribute
            # print("Yo")
            # print(attr)
            # check the value stored in that attribute in the example
            # print(X.columns)
            example = X.iloc[current_example_ind]
            # print(example)
            # print("example is ")
            # print(example)
            # print("attr is "+str(attr))
            val = example[attr]
            # recurse on both children
            if(val <= tree.split_val):
                self.pred_riro(current_example_ind,
                               tree.lchild, X, prediction_arr)
            else:
                self.pred_riro(current_example_ind,
                               tree.rchild, X, prediction_arr)

# if __name__ =='__main__':
#     s = DecisionTree('information_gain',5)
#     # N = 30
#     # P = 5
#     # X = pd.DataFrame(np.random.randn(N, P))
#     # y = pd.Series(np.random.randint(P, size = N), dtype="category")
#     N = 30
#     P = 2
#     X = pd.DataFrame(np.random.randn(N, P))
#     y = pd.Series(np.random.randint(P, size = N), dtype="category")
#     # N = 30
#     # P = 5   
#     # X = pd.DataFrame({i:pd.Series(np.random.randint(P, size = N), dtype="category") for i in range(5)})
#     # y = pd.Series(np.random.randint(P, size = N),  dtype="category")
#     s.fit(X,y)
#     s.plot()
#     y_hat = s.predict(X)
#     print(y_hat)
