
from sklearn.tree import DecisionTreeClassifier
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import tree as sktree
import os
import matplotlib.colors as colors


class AdaBoostClassifier():
    def __init__(self, base_estimator=DecisionTreeClassifier, n_estimators=5,
                 max_depth=1, criterion="entropy"):
        '''
        :param base_estimator: The base estimator model instance from which the boosted ensemble is built (e.g., DecisionTree, LinearRegression).
                               If None, then the base estimator is DecisionTreeClassifier(max_depth=1).
                               You can pass the object of the estimator class
        :param n_estimators: The maximum number of estimators at which boosting is terminated. In case of perfect fit, the learning procedure may be stopped early.
        '''
        # Base estimator will be the Decision Tree Model which we use from sklearn 
        self.base_estimator = base_estimator
        self.max_depth = max_depth
        self.criterion = criterion
        self.n_estimators = n_estimators
        self.trees = []
        self.alphas = []
        self.X = None
        self.y = None

    def fit(self, X, y):
        """
        Function to train and construct the AdaBoostClassifier
        Inputs:
        X: pd.DataFrame with rows as samples and columns as features (shape of X is N X P) where N is the number of samples and P is the number of columns.
        y: pd.Series with rows corresponding to output variable (shape of Y is N)
        """
        y_len = len(y)
        weight_array = np.ones(y_len)
        weight_array = weight_array/y_len
        self.X = X
        self.y = y
        for _ in tqdm(range(self.n_estimators)):
            # Learning
            # Using the sklearn tree
            tree = self.base_estimator(
                criterion=self.criterion, max_depth=self.max_depth)
            # Fitting the tree with the initialized sample weights 
            tree.fit(X, y, sample_weight=weight_array)
            y_hat = pd.Series(tree.predict(X))
            # Calculating error and alpha
            # Finding the error
            # Storing all the indexes where prediction didnt match
            mismatch_index = y_hat != y
            num_error = np.sum(weight_array[mismatch_index])
            denom_error = np.sum(weight_array)
            error = num_error/denom_error
            alpha = 0.5*np.log((1-error)/error)

            # Updating Weights
            weight_array[mismatch_index] = weight_array[mismatch_index] * np.exp(alpha)
            weight_array[~mismatch_index] = weight_array[~mismatch_index] *np.exp(-alpha)
            # Normalizing weight_array with the new weight array sum
            weight_array /= np.sum(weight_array)

            # Storing the tree and the corresponding alpha
            self.trees.append(tree)
            self.alphas.append(alpha)

    def predict(self, X):
        """
        Input:
        X: pd.DataFrame with rows as samples and columns as features
        Output:
        y: pd.Series with rows corresponding to output variable. THe output variable in a row is the prediction for sample in corresponding row in X.
        """
        ret_val = None
        for i in range (len(self.alphas)):
            alpha = self.alphas[i]
            tree = self.trees[i]
            # iterating over all learnt trees
            if ret_val is None: #not initialized
                ret_val = pd.Series(tree.predict(X))
                ret_val[0] =  ret_val[0]*alpha
            else:
                ret_val += pd.Series(tree.predict(X))*alpha
        return ret_val.apply(np.sign)
    
    def plot(self,figure = False,figname = "test"):
        for i, model in enumerate(self.trees):
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
            self.plot_one_estimator(ax,self.trees[i].predict,xx,yy,x_train,y_train,i)

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
