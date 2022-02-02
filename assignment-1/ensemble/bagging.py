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



    def plot(self, X, y):
        """
        Function to plot the decision surface for BaggingClassifier for each estimator(iteration).
        Creates two figures
        Figure 1 consists of 1 row and `n_estimators` columns and should look similar to slide #16 of lecture
        The title of each of the estimator should be iteration number

        Figure 2 should also create a decision surface by combining the individual estimators and should look similar to slide #16 of lecture

        Reference for decision surface: https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html

        This function should return [fig1, fig2]

        """
        color = ["r", "b", "g"]
        Zs = []
        fig1, ax1 = plt.subplots(
            1, len(self.models), figsize=(5*len(self.models), 4))

        x_min, x_max = X[0].min(), X[0].max()
        y_min, y_max = X[1].min(), X[1].max()
        x_range = x_max-x_min
        y_range = y_max-y_min

        for i, tree in enumerate(self.models):
            X_tree = self.x_datas[i]
            y_tree = self.y_datas[i]
            # X_tree, y_tree = self.datas[i]

            xx, yy = np.meshgrid(np.arange(x_min-0.2, x_max+0.2, (x_range)/50),
                                np.arange(y_min-0.2, y_max+0.2, (y_range)/50))

            ax1[i].set_ylabel("X2")
            ax1[i].set_xlabel("X1")
            Z = tree.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)
            Zs.append(Z)
            cs = ax1[i].contourf(xx, yy, Z, cmap=plt.cm.RdYlBu)
            fig1.colorbar(cs, ax=ax1[i], shrink=0.9)

            for y_label in y.unique():
                idx = y_tree == y_label
                id = list(y_tree.cat.categories).index(y_tree[idx].iloc[0])
                ax1[i].scatter(X_tree.loc[idx, 0], X_tree.loc[idx, 1], c=color[id],
                            cmap=plt.cm.RdYlBu, edgecolor='black', s=30,
                            label="Class: "+str(y_label))
            ax1[i].set_title("Decision Surface Tree: " + str(i+1))
            ax1[i].legend()
        fig1.tight_layout()
        # For Common surface
        fig2, ax2 = plt.subplots(1, 1, figsize=(5, 4))
        Zs = np.array(Zs)
        com_surface, _ = weighted_mode(Zs, np.ones(Zs.shape))
        cs = ax2.contourf(xx, yy, Z, cmap=plt.cm.RdYlBu)
        for y_label in y.unique():
            idx = y == y_label
            id = list(y.cat.categories).index(y[idx].iloc[0])
            ax2.scatter(X.loc[idx, 0], X.loc[idx, 1], c=color[id],
                        cmap=plt.cm.RdYlBu, edgecolor='black', s=30,
                        label="Class: "+str(y_label))
        ax2.set_ylabel("X2")
        ax2.set_xlabel("X1")
        ax2.legend()
        ax2.set_title("Common Decision Surface")
        fig2.colorbar(cs, ax=ax2, shrink=0.9)

        # Saving Figures
        image_path = ''
        fig1.savefig(os.path.join("plots", "Q6_Fig1.png"))
        fig2.savefig(os.path.join("plots", "Q6_Fig2.png"))
        return fig1, fig2

    
