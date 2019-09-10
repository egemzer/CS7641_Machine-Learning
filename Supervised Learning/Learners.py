# ###CS7641 - Machine Learning Project 1 code: Supervised Learning ###

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import DTGetBestParas
import sklearn
import seaborn as sns

# In this file, there is code that allows for the creation of five different classification learners
# on two different UCI datasets:

# Datasets: Contraceptive Method Choice, Arrhythmia
# Learners: Decision Tree, Neural Network, Boosting, Support Vector Machines (SVM), and k-Nearest Neighbors (KNN)

# Each model will output two plots: learning curve and model complexity based on https://www.dataquest.io/blog/learning-curves-machine-learning/.



# ### Data loading (also in README) ###

# IMPORTANT! To run this code, you need to save the datasets to your local machine, in the same folder as this code.

# STEP 1: Download the Contraceptives Dataset: https://www.openml.org/d/23 as a CSV (or from my Github) into the same folder as this code

# open and create a pandas dataframe
contra_df = pd.read_csv('ContraceptivesMethodsData.csv', na_values = "?")
# contra_df = contra_df.astype('int64').dtypes
contra_df.apply(pd.to_numeric)

# Check the frame (first 5 rows) to ensure all the columns are there
contra_df.head()
contra_df.describe(include='all')
print("Contraceptives Methods dataset size: ",len(contra_df),"rows and", len(contra_df.columns),"columns.")

# STEP 2: Download the Arrhythmia Dataset: https://www.openml.org/d/5 as a CSV (or from my Github) into the same folder as this code

# open and create a pandas dataframe
ekg_df = pd.read_csv('ArrhythmiaData.csv', na_values = "?")
# This dataset uses "?" for missing values, which is problematic for downstream. We will replace this with np.nan
ekg_df = ekg_df.replace("?", np.nan)
ekg_df.apply(pd.to_numeric)


# We need to drop and fill a few columns due to missing data before we can convert the dataframe to numeric
# by doing ekg_df.isna().sum() we can see how many columns have missing values, in this dataset denoted as "?"
# Since "J" column is missing 376 values, we need to delete it since we can't fill it appropriately.
ekg_df = ekg_df.drop(['J'], axis=1)

# "P" column is missing 22 values, perhaps we can take a mean of the remaining values to fill the column since P is an important part of an EKG waveform, we don't want to drop it
ekg_df['P'] = ekg_df['P'].fillna(ekg_df['P'].mean())

# "QRST", "T",  and "heartrate" columns are each missing a single value, let's fill them with a mean
ekg_df['QRST'] = ekg_df['QRST'].fillna(ekg_df['QRST'].mean())
ekg_df['T'] = ekg_df['T'].fillna(ekg_df['T'].mean())
ekg_df['heartrate'] = ekg_df['heartrate'].fillna(ekg_df['heartrate'].mean())

if len(ekg_df.columns[ekg_df.isna().any()].tolist()) > 0:
    print("WARNING! You didn't catch all the missing values!")

# Check the frame (first 5 rows) to ensure all the rows and columns are there
ekg_df.head()
print("Arrhythmia dataset size: ", len(ekg_df), "rows and", len(ekg_df.columns), "columns.")

# ## Now we need to clean up the data for more effective downstream processing / learning
# (1) for any non-ordinal numerical data, we will one hot encode based on the principle researched here:
# https://machinelearningmastery.com/why-one-hot-encode-data-in-machine-learning/
# (2) we will move the results column to be the first column for simplicity

# ## Cleaning up Contraceptives data
# (1) one-hot encode the columns where numerical categories are not ordinal
col_to_one_hot = ['Wifes_religion', 'Wifes_now_working%3F', 'Husbands_occupation', 'Media_exposure']
df_one_hot = contra_df[col_to_one_hot]
df_one_hot = pd.get_dummies(df_one_hot) # converts series to dummy codes
df_remaining_cols = contra_df.drop(col_to_one_hot, axis=1) # the rest of the dataframe is maintained
contra_df = pd.concat([df_one_hot, df_remaining_cols], axis=1)

# (2) move results (output) column to the front for consistency
column_order = list(contra_df)
column_order.insert(0, column_order.pop(column_order.index('Contraceptive_method_used')))
contra_df = contra_df.loc[:, column_order]  # move the target variable to the front for consistency

# (3) Prove that we didn't mess it up
contra_df.describe(include='all') # see the changes

# ## Cleaning up Arrhythmia data (less trivial given >200 attributes and output labels that are not ordinal, though most of the values are linear and ordinal)
# (1) one-hot encode the columns where numerical categories are not ordinal
col_to_one_hot = ['sex']
df_one_hot = ekg_df[col_to_one_hot]
df_one_hot = pd.get_dummies(df_one_hot) # converts series to dummy codes
df_remaining_cols = ekg_df.drop(col_to_one_hot, axis=1) # the rest of the dataframe is maintained
ekg_df = pd.concat([df_one_hot, df_remaining_cols], axis=1)

# (2) move results (output) column to the front for consistency
column_order = list(ekg_df)
column_order.insert(0, column_order.pop(column_order.index('class')))
ekg_df = ekg_df.loc[:, column_order]  # move the target variable to the front for consistency

# (3) Prove that we didn't mess it up
ekg_df.describe(include='all') # see the changes

print("Hi, friend, you did all the data preprocessing with no errors!")

# ### Helper functions that will be leveraged across all the learners ###
from sklearn.model_selection import train_test_split

def import_data(df1, df2):
    X1 = np.array(df1.values[:,1:-1])
    Y1 = np.array(df1.values[:,0])
    X2 = np.array(df2.values[:,1:-1])
    Y2 = np.array(df2.values[:,0])
    return X1, Y1, X2, Y2

def split_data(X1, Y1, testSize):
    """
    Parameters
    ----------
    X1 : training data (inputs)
    Y1 : training data (outputs)
    testSize: float representing the portion of data to hold for testing
        """

    X_train, X_test, Y_train, Y_test = train_test_split(X1, Y1, test_size=testSize)
    return X_train, X_test, Y_train, Y_test


from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.datasets import load_digits
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit

# taken from scikit-learn documentation: https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html
def plot_learning_curve(estimator, learner, X, y , ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate a simple plot of the test and training learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    learner : string
        Type of learner, goes into the title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - :term:`CV splitter`,
          - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : int or None, optional (default=None)
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    train_sizes : array-like, shape (n_ticks,), dtype float or int
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the dtype is float, it is regarded as a
        fraction of the maximum size of the training set (that is determined
        by the selected validation method), i.e. it has to be within (0, 1].
        Otherwise it is interpreted as absolute sizes of the training sets.
        Note that for classification the number of samples usually have to
        be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))
    """
    plt.figure()
    plt.title("Learning Curves for: " + learner)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Model Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")
    plt.legend(loc="best")
    plt.show()
    return train_sizes, train_scores_mean, fit_mean, pred_mean


# ### Decision Tree Learner (with Pruning)
import copy
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import train_test_split
from sklearn.tree._tree import TREE_LEAF

# Prune tree method taken from https://github.com/scikit-learn/scikit-learn/issues/10810
def prune(tree):
    tree = copy.deepcopy(tree)
    dat = tree.tree_
    nodes = range(0, dat.node_count)
    ls = dat.children_left
    rs = dat.children_right
    classes = [[list(e).index(max(e)) for e in v] for v in dat.value]

    leaves = [(ls[i] == rs[i]) for i in nodes]

    LEAF = -1
    for i in reversed(nodes):
        if leaves[i]:
            continue
        if leaves[ls[i]] and leaves[rs[i]] and classes[ls[i]] == classes[rs[i]]:
            ls[i] = rs[i] = LEAF
            leaves[i] = True
    return tree

def decisionTree (X, Y, split_crit, md, mf, mls, mss, testSize):
    X_train, X_test, Y_train, Y_test = split_data(X, Y, testSize)
    clf = tree.DecisionTreeClassifier(criterion=split_crit, max_depth=md,
 max_features=mf,min_samples_leaf=mls,
 min_samples_split=mss, splitter='best')
    clf = clf.fit(X_train, Y_train)

    # Prune it
    clf_pruned = prune(clf)

    # Test it and plot it
    y_pred = clf_pruned.predict(X_test)

import pydotplus
from IPython.display import Image

def plot_tree(self, clf, feature_names):
    '''
    input:
        tree of clf
        feature_importance
    output:

    '''
    dot_data = export_graphviz(clf, out_file=None,
                               feature_names=feature_names,
                               filled=True, rounded=True,
                               special_characters=True)

    graph = pydotplus.graphviz.graph_from_dot_data(dot_data)
    Image(graph.create_png())

# ### Decision Tree building and analysis

# get the data
X_contra, Y_contra, X_ekg, Y_ekg = import_data(contra_df, ekg_df)

# Find the bestsplit_crit for Contraceptives dataset
testSize = 0.25
X_train, X_test, Y_train, Y_test = split_data(X_contra, Y_contra, testSize)
split_criteria = ["gini", "entropy"]
datasets = ["contraceptives", "arrhythmia"]

for criteria in split_criteria:
    decisionTree(X_contra, Y_contra, criteria, md=5, mf=None, mls=1, mss=1.0, testSize=testSize)
    contra_estimator = DecisionTreeClassifier(max_depth=5, min_samples_leaf=1)
    train_sampl_contra, DT_train_score_contra, DT_fit_time_contra, DT_pred_time_contra = plot_learning_curve(contra_estimator, learner=="Decision Tree Contraceptives Methods"
                                                                                                        X_train,
                                                                                                        Y_train)

print("Did you make it past the first tuning?")
