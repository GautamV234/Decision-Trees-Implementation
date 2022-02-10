from tqdm import tqdm
import pandas as pd
import random
import numpy as np
from sklearn.utils.extmath import weighted_mode
import os
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn import tree as sktree


class RandomForestClassifier():
    def __init__(self, n_estimators=2, criterion='gini', max_depth=None,max_features=2):
        '''
        :param estimators: DecisionTree
        :param n_estimators: The number of trees in the forest.
        :param criterion: The function to measure the quality of a split.
        :param max_depth: The maximum depth of the tree.
        '''
        self.max_depth = max_depth              # max depth for each tree
        self.criterion = criterion              # criterion for the trees
        self.n_estimators = n_estimators        # number of trees
        self.models = []                         # array stroing the trees
        self.max_features = max_features        # max cols to select together
        self.sampled_Xs = []
        self.sampled_ys = []
        self.column_selected = []
        self.X= None
        self.y = None
        self.base_estimator = DecisionTreeClassifier
        
    

    def fit(self, X:pd.DataFrame, y:pd.Series):
        """
        Function to train and construct the RandomForestClassifier
        Inputs:
        X: pd.DataFrame with rows as samples and columns as features (shape of X is N X P) where N is the number of samples and P is the number of columns.
        y: pd.Series with rows corresponding to output variable (shape of Y is N)
        """
        self.X=X
        self.y = y
        for n in tqdm(range(self.n_estimators)):
            # num_columns = random.randint(1,self.max_features)
            sampled_X =X.sample(n =1 ,axis=1,)
            
            self.column_selected.append(sampled_X.columns.tolist())
            # Decision Tree Fitting
            model = self.base_estimator(criterion='entropy', max_depth=self.max_depth)
            model.fit(sampled_X,y)
            self.models.append(model)
            self.sampled_Xs.append(sampled_X)
            self.sampled_ys.append(y)

    def predict(self, X):
        """
        Funtion to run the RandomForestClassifier on a data point
        Input:
        X: pd.DataFrame with rows as samples and columns as features
        Output:
        y: pd.Series with rows corresponding to output variable. THe output variable in a row is the prediction for sample in corresponding row in X.
        """
        all_y_hats = []
        for model,select_columns in zip(self.models,self.column_selected):
            sub_X = X[select_columns]
            y_hat = model.predict(sub_X)
            y_hat = pd.Series(y_hat)
            all_y_hats.append(y_hat)
        df = pd.DataFrame(all_y_hats)
        return (pd.Series(df.mode(axis = 'rows').iloc[0,:].tolist()))

    def plot(self,figure = False,figname = "test"):
        for i, model in enumerate(self.models):
            print()
            print()
            print("#################")
            print("Current Tree Num = "+str(i+1))
            print()
            print()
            print(sktree.export_text(model))

        if(figure==False):
            return
        
        h = 0.02
        thresh = 0.5

        x_train = self.X.to_numpy()
        y_train = self.y.to_numpy()
        
        x_min, x_max = x_train[:,0].min()- thresh , x_train[:,0].max() + thresh
        y_min,y_max = x_train[:,1].min()- thresh, x_train[:,1].max() + thresh
        # print("1")
        xx,yy = np.meshgrid(np.arange(x_min,x_max,h),np.arange(y_min,y_max,h))
        # print("2")
        fig1,ax =plt.subplots(figsize =(20,5))
        # print("3")
        for i in range(self.n_estimators):
            # print("4")
            ax = plt.subplot(1,self.n_estimators,i+1)
            # print("5")
            plt.title("Estimator -> "+str(i+1))
            # print("6")
            self.plot_one_estimator(ax,self.models[i].predict,xx,yy,self.sampled_Xs[i].to_numpy(),self.sampled_ys[i].to_numpy(),i)
        
        fig2,ax = plt.subplots()
        plt.title("Final estimator")

        self.plot_one_estimator(ax,self.predict,xx,yy,x_train,y_train,-1) 
        fig1.savefig(os.path.join("plots", str(figname)+"_Fig1.png"))
        fig2.savefig(os.path.join("plots", str(figname)+"_Fig2.png"))
        return fig1,fig2
    
    def plot_one_estimator(self,ax,predictor,xx,yy,X,y,i):
        if(i==-1):
            # xx = self.X.to_numpy()
            # yy = self.y.to_numpy()
            X=self.X.to_numpy()
            y =self.y.to_numpy()
            frame = np.c_[xx.ravel(),yy.ravel()]
            df = pd.DataFrame(frame,columns=list(self.X.columns))
            Z = predictor(df)
            if(type(Z)!=np.ndarray):
                Z = Z.to_numpy()
            Z = Z.reshape(xx.shape)
            ax.contourf(xx,yy,Z,cmap = plt.cm.RdBu,alpha = 0.7)
            X_ = self.X.to_numpy()
            ax.scatter(X_[:,0],X_[:,1],c= y,cmap=colors.ListedColormap(["#FF0000","#0000FF"]),edgecolors="k")
            ax.set_xlabel("Feature 0")
            ax.set_ylabel("Feature 1")
            ax.set_xlim(xx.min(),xx.max())
            ax.set_ylim(yy.min(),yy.max())
            ax.set_xticks(())
            ax.set_yticks(())
        else:
            frame = np.c_[xx.ravel(),yy.ravel()]
            df = pd.DataFrame(frame,columns=list(self.X.columns))
            sub_df = df[self.column_selected[i]]
            Z = predictor(sub_df)
            if(type(Z)!=np.ndarray):
                Z = Z.to_numpy()
            Z = Z.reshape(xx.shape)
            ax.contourf(xx,yy,Z,cmap = plt.cm.RdBu,alpha = 0.7)
            X_ = self.X.to_numpy()
            ax.scatter(X_[:,0],X_[:,1],c= y,cmap=colors.ListedColormap(["#FF0000","#0000FF"]),edgecolors="k")
            ax.set_xlabel("Feature 0")
            ax.set_ylabel("Feature 1")
            ax.set_xlim(xx.min(),xx.max())
            ax.set_ylim(yy.min(),yy.max())
            ax.set_xticks(())
            ax.set_yticks(())


class RandomForestRegressor():

    def __init__(self, n_estimators=10, criterion='variance', max_depth=5,max_features=2):
        '''
        :param estimators: DecisionTree
        :param n_estimators: The number of trees in the forest.
        :param criterion: The function to measure the quality of a split.
        :param max_depth: The maximum depth of the tree.
        '''
        self.max_depth = max_depth              # max depth for each tree
        self.criterion = criterion              # criterion for the trees
        self.n_estimators = n_estimators        # number of trees
        self.models = []                         # array stroing the trees
        self.max_features = max_features        # max cols to select together
        self.base_estimator = DecisionTreeRegressor
        self.column_selected = []
        self.sampled_Xs = []
        self.sampled_ys = []

    def fit(self, X:pd.DataFrame, y:pd.Series):
        """
        Function to train and construct the RandomForestClassifier
        Inputs:
        X: pd.DataFrame with rows as samples and columns as features (shape of X is N X P) where N is the number of samples and P is the number of columns.
        y: pd.Series with rows corresponding to output variable (shape of Y is N)
        """
        print("A")
        self.X=X
        self.y = y
        for n in tqdm(range(self.n_estimators)):
            # num_columns = random.randint(1,self.max_features)
            sampled_X =X.sample(n = self.max_features ,axis=1,)
            
            self.column_selected.append(sampled_X.columns.tolist())
            # Decision Tree Fitting
            model = self.base_estimator(max_depth=self.max_depth)
            model.fit(sampled_X,y)
            self.models.append(model)
            self.sampled_Xs.append(sampled_X)
            self.sampled_ys.append(y)

    def predict(self, X):
        """
        Funtion to run the RandomForestClassifier on a data point
        Input:
        X: pd.DataFrame with rows as samples and columns as features
        Output:
        y: pd.Series with rows corresponding to output variable. THe output variable in a row is the prediction for sample in corresponding row in X.
        """
        all_y_hats = []
        for model,select_columns in zip(self.models,self.column_selected):
            sub_X = X[select_columns]
            y_hat = model.predict(sub_X)
            y_hat = pd.Series(y_hat)
            all_y_hats.append(y_hat)
        df = pd.DataFrame(all_y_hats)
        ret = pd.Series(df.mean(axis = 'rows'))
        return ret


    def plot(self):
        """
        Function to plot for the RandomForestClassifier.
        It creates three figures

        1. Creates a figure with 1 row and `n_estimators` columns. Each column plots the learnt tree. If using your sklearn, this could a matplotlib figure.
        If using your own implementation, it could simply invole print functionality of the DecisionTree you implemented in assignment 1 and need not be a figure.

        2. Creates a figure showing the decision surface/estimation for each estimator. Similar to slide 9, lecture 4

        3. Creates a figure showing the combined decision surface/prediction

        """
        for i, model in enumerate(self.models):
            print()
            print()
            print("#################")
            print("Current Tree Num = "+str(i+1))
            print()
            print()
            print(sktree.export_text(model)) 