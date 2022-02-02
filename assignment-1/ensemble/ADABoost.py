
from sklearn.tree import DecisionTreeClassifier
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import tree as sktree
import os


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



    def plot(self, X, y, name=""):
        """
        Function to plot the decision surface for AdaBoostClassifier for each estimator(iteration).
        Creates two figures
        Figure 1 consists of 1 row and `n_estimators` columns
        The title of each of the estimator should be associated alpha (similar to slide#38 of course lecture on ensemble learning)
        Further, the scatter plot should have the marker size corresponnding to the weight of each point.

        Figure 2 should also create a decision surface by combining the individual estimators

        Reference for decision surface: https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html

        This function should return [fig1, fig2]
        """

        assert(len(list(X.columns)) == 2)
        color_palette = ["r", "b", "g"]
        Zs = None
        fig1, ax1 = plt.subplots(
            1, len(self.trees), figsize=(5*len(self.trees), 4))

        x_min, x_max = X.iloc[:, 0].min(), X.iloc[:, 0].max()
        y_min, y_max = X.iloc[:, 1].min(), X.iloc[:, 1].max()
        x_range = x_max-x_min
        y_range = y_max-y_min
        
        for i in range(len(self.alphas)):
            tree = self.trees[i]
            alpha_m = self.alphas[i]
            print()
            print()
            print("#################")
            print("Current Tree Num = "+str(i+1))
            print()
            print()
            print(sktree.export_text(tree)) #printing the tree 
            
            xx, yy = np.meshgrid(np.arange(x_min-0.2, x_max+0.2, (x_range)/50),
                                 np.arange(y_min-0.2, y_max+0.2, (y_range)/50))

            ax1[i].set_ylabel("X2")
            ax1[i].set_xlabel("X1")
            # Plot the decision boundary. For that, we will assign a color to each
            # point in the mesh [x_min, x_max]x[y_min, y_max].
            Z = tree.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)
            if Zs is None:
                Zs = alpha_m*Z
            else:
                Zs += alpha_m*Z
            # plotting the subplot
            sp = ax1[i].contourf(xx, yy, Z, cmap=plt.cm.RdYlBu)
            # setting the colors
            fig1.colorbar(sp, ax=ax1[i], shrink=0.7)
            for y_label in y.unique():
                idx = y == y_label
                id = list(y.cat.categories).index(y[idx].iloc[0])
                ax1[i].scatter(X[idx].iloc[:, 0], X[idx].iloc[:, 1], c=color_palette[id],
                               cmap=plt.cm.RdYlBu, edgecolor='black', s=30,
                               label="Class: "+str(y_label))
            ax1[i].set_title("Decision Surface Tree: " + str(i+1))
            ax1[i].legend()
        fig1.tight_layout()

        # For Common surface
        fig2, ax2 = plt.subplots(1, 1, figsize=(5, 4))
        com_surface = np.sign(Zs)
        sp = ax2.contourf(xx, yy, com_surface, cmap=plt.cm.RdYlBu)
        for y_label in y.unique():
            idx = y == y_label
            id = list(y.cat.categories).index(y[idx].iloc[0])
            ax2.scatter(X[idx].iloc[:, 0], X[idx].iloc[:, 1], c=color_palette[id],
                        cmap=plt.cm.RdYlBu, edgecolor='black', s=30,
                        label="Class: "+str(y_label))
        ax2.set_ylabel("X2")
        ax2.set_xlabel("X1")
        ax2.legend(loc="lower right")
        ax2.set_title("Common Decision Surface")
        fig2.colorbar(sp, ax=ax2, shrink=0.9)

        # Saving Figures
        path1 = "Q5_{}Fig1.png".format(name)
        path2 = "Q5_{}Fig2.png".format(name)
        fig1.savefig(os.path.join("plots", path1))
        fig2.savefig(os.path.join("plots",  path2))
        return fig1, fig2


