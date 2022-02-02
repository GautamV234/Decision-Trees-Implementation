from random import sample
# from tree.base import DecisionTree
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn import tree as sktree
import matplotlib.pyplot as plt
from sklearn.utils.extmath import weighted_mode
import os
from sklearn.tree import DecisionTreeClassifier
import matplotlib.colors as colors


class BaggingClassifier():
    def __init__(self, base_estimator = DecisionTreeClassifier , n_estimators=5):

        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.max_depth = 8
        self.models = []
        self.frac = 0.6
        self.criterion = 'entropy'
        self.x_datas = []
        self.y_datas = []
        self.X = None
        self.y = None

        '''
        :param base_estimator: The base estimator model instance from which the bagged ensemble is built (e.g., DecisionTree(), LinearRegression()).
                               You can pass the object of the estimator class
        :param n_estimators: The number of estimators/models in ensemble.
        '''

    def fit(self, X:pd.DataFrame, y:pd.Series):
        """
        Function to train and construct the BaggingClassifier
        Inputs:
        X: pd.DataFrame with rows as samples and columns as features (shape of X is N X P) where N is the number of samples and P is the number of columns.
        y: pd.Series with rows corresponding to output variable (shape of Y is N)
        """
        self.X= X
        self.y = y
        for i in tqdm(range(self.n_estimators)):
            sampled_X = X.sample(frac = self.frac,replace=False)
            sampled_X.reset_index(drop = True,inplace=True)
            x_indices = sampled_X.index
            sampled_y = y[x_indices]
            sampled_y.reset_index(drop = True,inplace=True)
            model = self.base_estimator(
                criterion=self.criterion, max_depth=self.max_depth)
            model.fit(sampled_X,sampled_y)
            self.models.append(model)
            self.x_datas.append(sampled_X)
            self.y_datas.append(sampled_y)

    def predict(self, X):
        """
        Funtion to run the BaggingClassifier on a data point
        Input:
        X: pd.DataFrame with rows as samples and columns as features
        Output:
        y: pd.Series with rows corresponding to output variable. THe output variable in a row is the prediction for sample in corresponding row in X.
        """
        all_y_hats = []
        for model in self.models:
            y_hat = model.predict(X)
            y_hat = pd.Series(y_hat)
            all_y_hats.append(y_hat)
        df = pd.DataFrame(all_y_hats)
        return pd.Series(df.mode(axis = 'rows').iloc[0,:].tolist())

    def plot(self,figure = False,figname = "Q6A"):
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
        xx,yy = np.meshgrid(np.arange(x_min,x_max,h),np.arange(y_min,y_max,h))
        fig1,ax =plt.subplots(figsize =(20,5))
        for i in range(self.n_estimators):
            ax = plt.subplot(1,self.n_estimators,i+1)
            plt.title("Estimator -> "+str(i+1))
            self.plot_one_estimator(ax,self.models[i].predict,xx,yy,self.X.to_numpy(),self.y.to_numpy(),i)
        fig2,ax = plt.subplots()
        plt.title("Final estimator")

        self.plot_one_estimator(ax,self.predict,xx,yy,x_train,y_train,-1) 
        fig1.savefig(os.path.join("plots", str(figname)+"_Fig1.png"))
        fig2.savefig(os.path.join("plots", str(figname)+"_Fig2.png"))
        return fig1,fig2
    
    def plot_one_estimator(self,ax,predictor,xx,yy,X,y,i):
        if(i==-1):
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


