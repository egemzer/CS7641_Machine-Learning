# ###CS7641 - Machine Learning Project 1 code: Supervised Learning ###
# This Python file uses the following encoding: utf-8
# python3
# GT alias: gth659q
import timeit
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import warnings

# This script creates and analyses performance for five different classification learners
# on two different UCI / OpenML datasets:

# Datasets: Contraceptive Method Choice, Arrhythmia
# Learners: Decision Tree, Neural Network, Boosting, Support Vector Machines (SVM), and k-Nearest Neighbors (KNN)

# Each model outputs appropriate plots, at a minimum each model produces these plots:
# model complexity (based on one or two hyperparameters)
# learning curves (time based on F1 model performance based)
# prediction distributions
# confusion matrix
# based on https://www.dataquest.io/blog/learning-curves-machine-learning/.

# =====================
# ### Data loading ###
# =====================

# STEP 1: Download the Contraceptives Dataset from my Github into the same folder as this code.
# You can also get the data here: https://www.openml.org/d/23 (download as CSV)

# open and create a pandas dataframe
contra_df = pd.read_csv('ContraceptivesMethodsData.csv', na_values="?")
contra_df.apply(pd.to_numeric)

# Check the frame (first 5 rows) to ensure all the columns are there
contra_df.head()
contra_df.describe(include='all')
print("Contraceptives Methods dataset size: ",len(contra_df),"instances and", len(contra_df.columns) - 1,"features.")

# STEP 2: Download the Arrhythmia Dataset from my Github into the same folder as this code.
# You can also get the data here: https://www.openml.org/d/5 (download as CSV)

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
print("Arrhythmia dataset size: ", len(ekg_df), "patients and", len(ekg_df.columns) - 1, "features.")

# STEP 3: Now we need to clean up the data for more effective downstream processing / learning
# (1) for any non-ordinal numerical data, we will one hot encode based on the principle researched here:
# https://machinelearningmastery.com/why-one-hot-encode-data-in-machine-learning/
# (2) we will move the results column to be the first column for simplicity

# ## Cleaning up Contraceptives data
# (3A) one-hot encode the columns where numerical categories are not ordinal
col_to_one_hot = ['Wifes_religion', 'Wifes_now_working%3F', 'Husbands_occupation', 'Media_exposure']
df_one_hot = contra_df[col_to_one_hot]
df_one_hot = pd.get_dummies(df_one_hot) # converts series to dummy codes
df_remaining_cols = contra_df.drop(col_to_one_hot, axis=1) # the rest of the dataframe is maintained
contra_df = pd.concat([df_one_hot, df_remaining_cols], axis=1)

# (3B) move results (output) column to the front for consistency
column_order = list(contra_df)
column_order.insert(0, column_order.pop(column_order.index('Contraceptive_method_used')))
contra_df = contra_df.loc[:, column_order]  # move the target variable to the front for consistency

# (3C) Prove that we didn't mess it up
contra_df.describe(include='all') # see the changes

# (3D) Grab the classes from the y-data (which look like numbers but actually have more semantic meaning to be used later.
contra_targets = ['No-use', "Long-term", "Short-term"]

# STEP 4: Cleaning up Arrhythmia data (less trivial given >200 attributes and output labels that are not ordinal, though most of the values are linear and ordinal)
# (4A) one-hot encode the columns where numerical categories are not ordinal
col_to_one_hot = ['sex']
df_one_hot = ekg_df[col_to_one_hot]
df_one_hot = pd.get_dummies(df_one_hot) # converts series to dummy codes
df_remaining_cols = ekg_df.drop(col_to_one_hot, axis=1) # the rest of the dataframe is maintained
ekg_df = pd.concat([df_one_hot, df_remaining_cols], axis=1)

# (4B) move results (output) column to the front for consistency
column_order = list(ekg_df)
column_order.insert(0, column_order.pop(column_order.index('class')))
ekg_df = ekg_df.loc[:, column_order]  # move the target variable to the front for consistency

# (4C) Prove that we didn't mess it up
ekg_df.describe(include='all') # see the changes

# (4D) We have to simplify the number of classes, since we have eight classes with none or fewer than 5 instances.
# We will collapse all the Arrhythmia classes into a single class, keep Normal, and keep the Unclassified class
# For a total of three prediction classes
ekg_df['class'].replace({3:2, 4:2, 5:2, 6:2, 7:2, 8:2, 9:2, 10:2, 11:2, 12:2, 13:2, 14:2, 15:2, 16:3},inplace=True)

# (4E) Grab the classes from the y-data (which look like numbers but actually have more semantic meaning to be used later.
ekg_classes = ['Normal', 'Classified Arrhythmia', 'Unclassified Arrhythmia']

# STEP 5: Shuffle the DF
from sklearn.utils import shuffle
ekg_df = shuffle(ekg_df)

print("Hi, friend, you did all the data preprocessing with no errors!")

#========================================================================
# ### Helper functions that will be leveraged across all the learners ###
#========================================================================
from sklearn.model_selection import GridSearchCV

def import_data(df1, df2):
    X1 = np.array(df1.values[:,1:]) # get all rows, and all columns after the first (y) column
    Y1 = np.array(df1.values[:,0]) # get all rows and just the first (index = 0) column of ys
    X2 = np.array(df2.values[:,1:])
    Y2 = np.array(df2.values[:,0])
    features_1 = df1.columns.values[1:]
    features_2 = df2.columns.values[1:]
    return X1, Y1, X2, Y2, features_1, features_2

from sklearn.model_selection import cross_validate
def plot_learning_curve(estimator, learner, dataset, X_train, y_train, ylim=None, cv=5, n_jobs=-1):
    """
    Generate a simple plot of the test and training learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    learner : string
        Type of learner, goes into the title for the chart.

    dataset : string
        Name of dataset, goes into the title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional. Default: 5

    n_jobs : int or None, optional (default=None)
        Number of jobs to run in parallel.
        ``-1`` means using all processors.

    train_sizes : array-like, shape (n_ticks,), dtype float or int
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. (default: np.linspace(0.1, 1.0, 5))

    Adapted from: https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html
    """
    warnings.filterwarnings('ignore', 'F-score is ill-defined.*')

    #Provision empty arrays to fill
    train_scores_mean = []; train_scores_std = []
    test_scores_mean = []; test_scores_std = []
    train_time_mean = []; train_time_std = []
    test_time_mean =[]; test_time_std = []

    train_sizes = (np.linspace(.1, 1.0, cv) * len(y_train)).astype('int')
    for size in train_sizes:
        index = np.random.randint(X_train.shape[0], size = size)
        y_slice = y_train[index]
        X_slice = X_train[index,:]
        cv_results = cross_validate(estimator, X_slice, y_slice, cv=cv, n_jobs=n_jobs, scoring='f1_weighted', return_train_score=True)

        # Fill out out data arrays with actual results from the cross validator
        train_scores_mean.append(np.mean(cv_results['train_score'])); train_scores_std.append(np.std(cv_results['train_score']))
        test_scores_mean.append(np.mean(cv_results['test_score'])); test_scores_std.append(np.std(cv_results['test_score']))
        train_time_mean.append(np.mean(cv_results['fit_time'] * 1000)); train_time_std.append(np.std(cv_results['fit_time'] * 1000))
        test_time_mean.append(np.mean(cv_results['score_time'] * 1000)); test_time_std.append(np.std(cv_results['score_time'] * 1000))

    # Convert arrays to numpy arrays for later math operations
    train_scores_mean = np.array(train_scores_mean); train_scores_std = np.array(train_scores_std)
    test_scores_mean = np.array(test_scores_mean); test_scores_std = np.array(test_scores_std)
    train_time_mean = np.array(train_time_mean); train_time_std = np.array(train_time_std)
    test_time_mean = np.array(test_time_mean); test_time_std = np.array(test_time_std)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    fig.suptitle(learner + " Learning Curves for: " + dataset)
    if ylim is not None:
        ax1.ylim(*ylim)
    ax1.set(xlabel="Number of Training Examples", ylabel="Model F1 Score")

    # Set up the first learning curve: F1 score (y) vs training size (X):
    ax1.grid()
    ax1.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    ax1.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    ax1.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    ax1.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation testing score")
    ax1.legend(loc="best")

    # Set up the second learning curve: Time (y) vs training size (X):
    if ylim is not None:
        ax2.ylim(*ylim)
    ax2.set(xlabel="Number of Training Examples", ylabel="Model Time, in milliseconds")
    ax2.grid()
    ax2.fill_between(train_sizes, train_time_mean - train_time_std,
                     train_time_mean + train_time_std, alpha=0.1,
                     color="r")
    ax2.fill_between(train_sizes, test_time_mean - test_time_std,
                     test_time_mean + test_time_std, alpha=0.1, color="g")
    ax2.plot(train_sizes, train_time_mean, 'o-', color="r",
             label="Training Time (ms)")
    ax2.plot(train_sizes, test_time_mean, 'o-', color="g",
             label="Prediction Time (ms)")
    ax2.legend(loc="best")

    # show the plots!
    plt.show()

    return train_sizes, train_scores_mean, train_time_mean, test_time_mean

def  get_samples_leaf(n):
    '''
    Advice in ML says that minimum samples per leaf should be between 0.5% and 5% of the dataset.
    Inputs:
        n: number of the samples
        return: number of min_samples_leaf, number of max_samples_leaf

    Citation (leveraged from):
    https://github.com/shenwanxiang/sklearn-post-prune-tree/blob/master/DTGetBestParas.py

    '''
    if n >= 1000:
        return int(0.005 * n), int(0.5 * n)
    else:
        return int(5), int(n/2)


from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import itertools
from sklearn.ensemble import GradientBoostingClassifier


def plot_confusion_matrix(y_true, y_pred, target_names, learner, dataset, cmap=plt.cm.RdPu, normalize=True):
        """
        Plot a confusion matrix using sklearn confusion_matrix
        
        Citations:
        https://stackoverflow.com/questions/19233771/sklearn-plot-confusion-matrix-with-labels
        http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

        Args:
            y_true : the correct values of y
            y_pred : the model's predicted values of y

            target_names: given classification classes such as [0, 1, 2]
            the class names, for example: ['high', 'medium', 'low']

            title: the text to display at the top of the confusion matrix

            cmap: the gradient of the values displayed from matplotlib.pyplot.cm
                      see http://matplotlib.org/examples/color/colormaps_reference.html
                      plt.get_cmap('jet') or plt.cm.Blues

            normalize: If False, plot the raw numbers
                      If True, plot the proportions (as percentage)

        """
        cm = confusion_matrix(y_true, y_pred)
        accuracy = np.trace(cm) / float(np.sum(cm))
        misclass = 1 - accuracy

        if cmap is None:
            cmap = plt.get_cmap('Greens')

        plt.figure(figsize=(12, 10))
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(learner + " Confusion Matrix for: " + dataset)
        plt.colorbar()
        plt.tick_params(labelsize=14)

        if target_names is not None:
            tick_marks = np.arange(len(target_names))
            plt.xticks(tick_marks, target_names)
            plt.yticks(tick_marks, target_names)

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        thresh = 0.55
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            if normalize:
                if i == min(range(cm.shape[0])):
                    plt.text(j, i + 0.25, "{0:0.1%}".format(cm[i, j]),
                             horizontalalignment="center",
                             color="black" if cm[i, j] <= thresh  else "white", fontsize=14)
                elif i == max(range(cm.shape[0])):
                    plt.text(j, i - 0.25, "{0:0.1%}".format(cm[i, j]),
                             horizontalalignment="center",
                             color="black" if cm[i, j] <= thresh else "white", fontsize=14)
                else:
                    plt.text(j, i, "{0:0.1%}".format(cm[i, j]),
                             horizontalalignment="center",
                             color="black" if cm[i, j] <= thresh else "white",  fontsize=14)

            else:
                plt.text(j, i + 0.25, "{:,}".format(cm[i, j]),
                         horizontalalignment="center",
                         color="black" if cm[i, j] <= thresh else "white", fontsize=14)

        plt.tight_layout()
        plt.ylabel('True label', fontsize=14)
        plt.xlabel('Predicted label\naccuracy={:0.3f}; misclass={:0.3f}'.format(accuracy, misclass), fontsize=14)
        plt.show()

# Test plot_confusion_matrix
# y_tests = np.array([0, 0 ,0, 0, 1, 1, 1, 1, 1])
# y_preds = np.array([0, 0 ,0, 1, 0, 1, 1, 1, 1])
# learn = "Decision Tree"
# data = "TestData"
# classes = [False, True]
# plot_confusion_matrix(y_tests, y_preds, learner=learn, dataset=data, target_names=classes)
# print("test of confusion matrix complete!")

def evaluate_classifier(classifier, learner, dataset, X_train, X_test, y_train, y_test, class_names, feature_names):
    # Training time
    start_time = timeit.default_timer()
    classifier.fit(X_train, y_train)
    end_time = timeit.default_timer()
    training_time = (end_time - start_time) * 1000  # milliseconds

    # Prediction Time
    start_time = timeit.default_timer()
    y_pred = classifier.predict(X_test)
    end_time = timeit.default_timer()
    pred_time = (end_time - start_time) * 1000  # milliseconds

    # Distribution of Predictions
    unique, counts = np.unique(y_pred, return_counts=True)
    res_classes = dict(zip(unique, counts))
    plt.bar(*zip(*res_classes.items()))
    plt.xticks((1, 2, 3), labels=class_names)
    plt.title("Distribution of Predictions, over all Classes")
    plt.show()

    # See the Tree
    if isinstance(classifier, DecisionTreeClassifier):
        plot_tree(classifier, featureNames=feature_names, fileName='Final Decision Tree for %s' %(dataset))
    elif isinstance(classifier, GradientBoostingClassifier):
        plot_random_tree(classifier, featureNames=feature_names, fileName='Random DT in the Boosted Classifier for %s' %(dataset), X_train=X_train, y_train=y_train)

    # Standard Metrics
    f1 = f1_score(y_test, y_pred, average='weighted', labels=np.unique(y_pred))  # Use average = 'micro', 'macro' or 'weighted' since we have non-binary classes. Don't count classes that are never predicted.
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', labels=np.unique(y_pred))
    recall = recall_score(y_test, y_pred, average='weighted', labels=np.unique(y_pred))
    plot_confusion_matrix(y_test, y_pred, learner=learner, dataset=dataset, target_names=class_names)

    print("Metrics for the Candidate Classifier on the %s Dataset" %dataset)
    print("Model Training Time (ms):   " + "{:.1f}".format(training_time))
    print("Model Prediction Time (ms): " + "{:.1f}\n".format(pred_time))
    print("F1 Score:  " + "{:.2f}".format(f1))
    print("Check the confusion plot!")
    print("Accuracy:  " + "{:.2f}".format(accuracy))
    print("Precision: " + "{:.2f}".format(precision))
    print("Recall:    " + "{:.2f}".format(recall))
    return "{0:.2f}".format(f1), "{0:.2f}".format(accuracy), "{0:.2f}".format(precision), "{0:.2f}".format(recall), "{:.1f}".format(training_time), "{:.1f}".format(pred_time)

#======================#
# ===== Data Prep =====#
#======================#
from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# get the data and segment into training and testing
testSize = 0.15  # default is 0.25, manually tune if interesting

# Pull the data from the dataframe
X_contra, y_contra, X_ekg, y_ekg, contra_features, ekg_features = import_data(contra_df, ekg_df)
X_train_contra, X_test_contra, y_train_contra, y_test_contra = train_test_split(np.array(X_contra), np.array(y_contra), test_size=testSize, random_state=17)

# Split into training and test data
X_train_ekg, X_test_ekg, y_train_ekg, y_test_ekg = train_test_split(np.array(X_ekg), np.array(y_ekg), test_size=testSize, random_state=17) # Split the data into test and train BEFORE resampling

# We need to resample the Arrhythmia dataset because the majority class is more than 50% of the dataset AND all other classes represent <10% AND our total dataset size (rows) is <1000, we need to oversample the minority classes. Random usually works better than picking.
ros1 = RandomOverSampler(sampling_strategy='not majority')
X_train_res_ekg, y_train_res_ekg = ros1.fit_resample(X_train_ekg, y_train_ekg)

# # Prove that you've correctly resampled
# print('Resampled training dataset shape %s' % len(y_train_res_ekg))
# unique, counts = np.unique(y_train_res_ekg, return_counts=True)
# res_classes = dict(zip(unique, counts))
# plt.bar(*zip(*res_classes.items()))
# plt.xticks((1,2,3), labels=ekg_classes)
# plt.title("Resampling (Oversampling) of Training Data")
# plt.show()
#
# unique, counts = np.unique(y_test_ekg, return_counts=True)
# res_classes = dict(zip(unique, counts))
# plt.bar(*zip(*res_classes.items()))
# plt.xticks((1,2,3), labels=ekg_classes)
# plt.title("Distribution of Test Data")
# plt.show()

# We need to scale both datasets for Neural Networks and SVMs to perform well.
# Note that the training will be done on the rebalanced/resampled training dataset
# But evaluation will be done on the original holdout test set with NO resampling.
contra_scaler = StandardScaler()
ekg_scaler = StandardScaler()
contra_scaler.fit(X_train_contra)
ekg_scaler.fit(X_train_res_ekg)
X_train_contra = contra_scaler.transform(X_train_contra)
X_test_contra = contra_scaler.transform(X_test_contra)
X_train_res_ekg = ekg_scaler.transform(X_train_res_ekg)
X_test_ekg = ekg_scaler.transform(X_test_ekg)

# =========================================
# ### Decision Tree Learner (with Pruning)
# =========================================
from sklearn.tree import DecisionTreeClassifier

# ====== Pruning Algorithm and methods =====
# Leveraged from: https://stackoverflow.com/questions/51397109/prune-unnecessary-leaves-in-sklearn-decisiontreeclassifier
from sklearn.tree._tree import TREE_LEAF
import copy

def prune_inner(inner_tree, index=0, threshold=5):
    '''
    removes all children of the nodes with minimum instance per class count less than a threshold
    :param inner_tree: your classifier's tree, ie: classifier.tree_
    :param index: the node to start. You have to start at the root (index=0)
    :param threshold: set to 5 by default but modifiable
    :return: pruned copy of the input classifier
    '''

    if inner_tree.value[index].size >= 6:
        if np.count_nonzero(inner_tree.value[index]) / inner_tree.value[index].size < 1/2:
            # turn node into a leaf by "unlinking" its children
            inner_tree.children_left[index] = TREE_LEAF
            inner_tree.children_right[index] = TREE_LEAF
        # if there are children, visit them as well
        if inner_tree.children_left[index] != TREE_LEAF:
            prune_inner(inner_tree, inner_tree.children_left[index])
            prune_inner(inner_tree, inner_tree.children_right[index])

    else:
        if inner_tree.value[index].min() < threshold:
            # turn node into a leaf by "unlinking" its children
            inner_tree.children_left[index] = TREE_LEAF
            inner_tree.children_right[index] = TREE_LEAF
        # if there are children, visit them as well
        if inner_tree.children_left[index] != TREE_LEAF:
            prune_inner(inner_tree, inner_tree.children_left[index], threshold=threshold)
            prune_inner(inner_tree, inner_tree.children_right[index], threshold=threshold)

def prune(classifier, threshold):
    classifier2 = copy.deepcopy(classifier)
    inner_tree = classifier2.tree_
    prune_inner(inner_tree, threshold=threshold)
    return classifier2

from sklearn.tree import export_graphviz
import pydotplus

def plot_tree(clf, featureNames, fileName):
    dot_data = export_graphviz(clf, feature_names=featureNames, filled=True, rounded=True, special_characters=True)
    graph = pydotplus.graph_from_dot_data(dot_data)
    graph.write_png(fileName)

def plot_random_tree(clf, featureNames, fileName, X_train, y_train):
    clf = clf.fit(X_train, y_train)

    # Get tree #17
    sub_tree = clf.estimators_[17, 0]

    dot_data = export_graphviz(sub_tree, feature_names=featureNames, filled=True, rounded=True, special_characters=True)
    graph = pydotplus.graph_from_dot_data(dot_data)
    graph.write_png(fileName)

#=========================================
# Decision Tree Learner (with Pruning)
#=========================================
# manual tuning approach leveraged from: https://www.kaggle.com/drgilermo/playing-with-the-knobs-of-sklearn-decision-tree
# Decision tree builder that also does pruning and calculates F1 score (due to non-binary, multi-classification task, see: https://medium.com/usf-msds/choosing-the-right-metric-for-evaluating-machine-learning-models-part-2-86d5649a5428)

def dt_initial_tuning(X_train, y_train, max_features):
    warnings.filterwarnings('ignore', 'F-score is ill-defined.*')

    max_depth_absolute = 17 # should be small as a means of pre-pruning

    # Parameters to optimize
    params = {}
    params['criterion'] = ["gini", "entropy"]
    params['max_depth'] = np.arange(2, max_depth_absolute, 2)

    bestTree = GridSearchCV(
        estimator=DecisionTreeClassifier(random_state=17, max_features=max_features),
        param_grid=params, cv=3, n_jobs=-1)
    bestTree.fit(X_train, y_train)

    print("Using GridSearchCV for hyperparameter tuning, we find that the best initial DT parameters are:")
    print(bestTree.best_params_)

    # Get Test Scores Mean and std for each grid search
    scores_mean = bestTree.cv_results_['mean_test_score']
    scores_mean = np.array(scores_mean).reshape(len(params['max_depth']), len(params['criterion']))

    scores_sd = bestTree.cv_results_['std_test_score']
    scores_sd = np.array(scores_sd).reshape(len(params['max_depth']), len(params['criterion']))

    # Plot Validation Curves / Model Complexity Curves from GS results
    _, ax = plt.subplots(1, 1)

    # Param1 is the X-axis, Param 2 is represented as a different curve (color line)
    for idx, val in enumerate(params['max_depth']):
        ax.plot(params['criterion'], scores_mean[idx, :], '-o', label='max_depth' + ': ' + str(val))

    ax.set_title("DT Validation Curves / Model Complexity")
    ax.set_xlabel('criterion')
    ax.set_ylabel('3-Fold Cross-validation Mean F1 Score')
    ax.legend(loc="best")
    plt.show()

    return bestTree.best_params_['max_depth'], bestTree.best_params_['criterion']

def finalDT(X_train, y_train, max_features, max_depth, criterion):
    warnings.filterwarnings('ignore', 'F-score is ill-defined.*')

    # Parameters to optimize
    params = {}
    params['max_leaf_nodes'] = np.arange(10, int(len(y_train) / max_features), 10)
    params['min_samples_split'] = (0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0)


    bestTree = GridSearchCV(estimator=DecisionTreeClassifier(criterion=criterion, max_depth=max_depth, random_state=17, max_features=max_features), param_grid=params, cv=3, n_jobs=-1)
    bestTree.fit(X_train, y_train)
    print("Using GridSearchCV for hyperparameter tuning, we find that the best DT parameters are:")
    print(bestTree.best_params_)

    # Get Test Scores Mean and std for each grid search
    scores_mean = bestTree.cv_results_['mean_test_score']
    scores_mean = np.array(scores_mean).reshape(len(params['min_samples_split']),len(params['max_leaf_nodes']))

    scores_sd = bestTree.cv_results_['std_test_score']
    scores_sd = np.array(scores_sd).reshape(len(params['min_samples_split']),len(params['max_leaf_nodes']))

    # Plot Validation Curves / Model Complexity Curves from GS results
    _, ax = plt.subplots(1,1)

    # Param1 is the X-axis, Param 2 is represented as a different curve (color line)
    for idx, val in enumerate(params['min_samples_split']):
        ax.plot(params['max_leaf_nodes'], scores_mean[idx,:], '-o', label= 'min_samples_split' + ': ' + str(val))

    ax.set_title("DT Validation Curves / Model Complexity")
    ax.set_xlabel('max_leaf_nodes')
    ax.set_ylabel('3-Fold Cross-validation Mean F1 Score')
    ax.legend(loc="best")
    plt.show()

    return bestTree.best_params_['min_samples_split'], bestTree.best_params_['max_leaf_nodes']

# ======= Decision Tree on Contraceptive Methods Dataset ======= #
# # Try some manual hyperparameter tuning, if you please
# dt_manual_tuning(X_train_contra, y_train_contra, X_test_contra, y_test_contra, "Contraceptives Methods")

# Get the hyperparameters for Contraceptive Methods via GridSearchCV and build the classifier / estimator
max_depth_contra, criterion_contra = dt_initial_tuning(X_train_res_ekg, y_train_res_ekg, max_features=9)
msl_contra, max_leaf_nodes_contra = finalDT(X_train_contra, y_train_contra, max_features=9, max_depth=max_depth_contra, criterion=criterion_contra)
contra_estimator = DecisionTreeClassifier(criterion=criterion_contra, max_features=9, min_samples_leaf=msl_contra, max_depth=max_depth_contra, max_leaf_nodes=max_leaf_nodes_contra, random_state=17)
contra_train_sizes, contra_train_scores_mean, contra_train_time_mean, contra_test_time_mean = plot_learning_curve(estimator=contra_estimator, learner='Decision Tree', dataset="Contraceptive Methods", X_train=X_train_contra, y_train=y_train_contra)
contra_dt_test_f1, contra_dt_test_acc, contra_dt_test_precision, contra_dt_test_recall, contra_dt_train_time, contra_dt_test_time =  evaluate_classifier(classifier=contra_estimator, X_train=X_train_contra, X_test=X_test_contra, y_train=y_train_contra, y_test=y_test_contra, learner='Decision Tree', dataset="Contraceptive Methods", class_names=contra_targets, feature_names=contra_features)
# print('Leaves count before pruning: %d' % contra_estimator.get_n_leaves())
plot_tree(clf=contra_estimator, featureNames=contra_features, fileName="Pre-pruned Contraceptives DT.png")

# Prune the classifier, plot learning curves, and evaluate the final classifier
pruned_contra_clf = prune(contra_estimator, threshold=5)
# print('Leaves count after pruning: %d' % pruned_contra_clf.get_n_leaves())
plot_tree(clf=pruned_contra_clf, featureNames=contra_features, fileName="Post-pruned Contraceptives DT.png")
contra_train_sizes, contra_train_scores_mean, contra_train_time_mean, contra_test_time_mean = plot_learning_curve(estimator=pruned_contra_clf, learner='Decision Tree', dataset="Contraceptive Methods", X_train=X_train_contra, y_train=y_train_contra)
contra_dt_test_f1, contra_dt_test_acc, contra_dt_test_precision, contra_dt_test_recall, contra_dt_train_time, contra_dt_test_time =  evaluate_classifier(classifier=pruned_contra_clf, X_train=X_train_contra, X_test=X_test_contra, y_train=y_train_contra, y_test=y_test_contra, learner='Decision Tree', dataset="Contraceptive Methods", class_names=contra_targets, feature_names=contra_features)
contra_dt_results = ['contra_dt_results', contra_dt_test_f1, contra_dt_test_acc, contra_dt_test_precision, contra_dt_test_recall, contra_dt_train_time, contra_dt_test_time]

print("Done with Contraceptives Decision Tree.")

# ======== Decision Tree on Arrhythmia Dataset ======= #

# Get the hyperparameters for Contraceptive Methods via GridSearchCV and build the classifier / estimator
max_depth_ekg, criterion_ekg = dt_initial_tuning(X_train_res_ekg, y_train_res_ekg, max_features=24)
msl_ekg, max_leaf_nodes_ekg = finalDT(X_train_res_ekg, y_train_res_ekg, max_features=20, max_depth=max_depth_ekg, criterion=criterion_ekg)
ekg_estimator = DecisionTreeClassifier(criterion=criterion_ekg, max_features=20, min_samples_leaf=msl_ekg, max_depth=max_depth_ekg, max_leaf_nodes=max_leaf_nodes_ekg, random_state=17)
ekg_train_sizes, ekg_train_scores_mean, ekg_train_time_mean, ekg_test_time_mean = plot_learning_curve(estimator=ekg_estimator, learner='Decision Tree', dataset="Arrhythmia (EKG)", X_train=X_ekg, y_train=y_ekg, cv=4)
ekg_dt_test_f1, ekg_dt_test_acc, ekg_dt_test_precision, ekg_dt_test_recall, ekg_dt_train_time, ekg_dt_test_time  = evaluate_classifier(classifier=ekg_estimator, X_train=X_train_res_ekg, X_test=X_test_ekg, y_train=y_train_res_ekg, y_test=y_test_ekg, learner='Decision Tree', dataset="Arrhythmia (EKG)", class_names=ekg_classes, feature_names=ekg_features)
# print('Leaves count before pruning: %d' % ekg_estimator.get_n_leaves())
plot_tree(clf=ekg_estimator, featureNames=ekg_features, fileName="Pre-pruned Arrhythmia (EKG) DT.png")

# Prune the classifier, plot learning curves, and evaluate the final classifier
pruned_ekg_clf = prune(ekg_estimator, threshold=5)
# print('Leaves count after pruning: %d' % pruned_ekg_clf.get_n_leaves())
plot_tree(clf=pruned_ekg_clf, featureNames=ekg_features, fileName="Post-pruned Arrhythmia DT.png")
ekg_train_sizes, ekg_train_scores_mean, ekg_train_time_mean, ekg_test_time_mean = plot_learning_curve(estimator=pruned_ekg_clf, learner='Decision Tree', dataset="Arrhythmia (EKG)", X_train=X_ekg, y_train=y_ekg, cv=4)
ekg_dt_test_f1, ekg_dt_test_acc, ekg_dt_test_precision, ekg_dt_test_recall, ekg_dt_train_time, ekg_dt_test_time = evaluate_classifier(classifier=pruned_ekg_clf, X_train=X_train_res_ekg, X_test=X_test_ekg, y_train=y_train_res_ekg, y_test=y_test_ekg, learner='Decision Tree', dataset="Arrhythmia (EKG)", class_names=ekg_classes, feature_names=ekg_features)
ekg_dt_results = ['ekg_dt_results', ekg_dt_test_f1, ekg_dt_test_acc, ekg_dt_test_precision, ekg_dt_test_recall, ekg_dt_train_time, ekg_dt_test_time]

print("Done with Arrhythmia Decision Tree")

# =================================
#   Boosted Decision Tree Learner
# =================================
def boostedDT(X_train, y_train, X_test, y_test, max_depth, datasetName):
    f1_test = []
    f1_train = []
    # n_estimators = np.linspace(10, 200, 20).astype('int') # gradient boosting is pretty robust to overfitting, so large numbers usually improve performance
    # for num in n_estimators:
    # learning_rate = np.arange(0.01,1, 0.01)
    # for lr in learning_rate:
    # min_samples_leaf = np.arange(1,50, 1)
    # for msl in min_samples_leaf:
    clf = GradientBoostingClassifier(max_depth=max_depth, n_jobs=-1, random_state=17)
    clf.fit(X_train, y_train)
    y_pred_train = clf.predict(X_train)
    y_pred_test = clf.predict(X_test)
    f1_train.append(f1_score(y_train, y_pred_train, average='weighted'))
    f1_test.append(f1_score(y_test, y_pred_test, average='weighted'))

    # plt.plot(n_estimators, f1_train, 'o-', color='b', label='Train F1 Score')
    # plt.plot(n_estimators, f1_test, 'o-', color='r', label='Test F1 Score')
    #
    # plt.ylabel('Model F1 Score')
    # plt.xlabel('No. Estimators')
    #
    # plt.title("Analysis of n_estimators Effect on Boosted DT for: " + datasetName)
    # plt.legend(loc='best')
    # plt.tight_layout()
    # plt.show()

def final_boosted_DT(X_train, y_train, max_depth, max_features, param1, param2):
    warnings.filterwarnings('ignore', 'Solver terminated early.*')
    warnings.filterwarnings('ignore', 'F-score is ill-defined.*')

    # Parameters to optimize
    params = {}
    params['n_estimators'] = np.linspace(10, 300, 10).astype('int')
    params['learning_rate'] = [0.1, 0.3, 0.6]

    boostedTree = GridSearchCV(estimator=GradientBoostingClassifier(max_depth=max_depth, random_state=17, max_features=9), param_grid=params, cv=3, n_jobs=-1, verbose=True)
    boostedTree.fit(X_train, y_train)
    print("Using GridSearchCV for hyperparameter tuning on the boosted DT, we find that the best parameters are:")
    print(boostedTree.best_params_)

    # Get Test Scores Mean and std for each grid search
    scores_mean = boostedTree.cv_results_['mean_test_score']
    scores_mean = np.array(scores_mean).reshape(len(params['learning_rate']),len(params['n_estimators']))

    scores_sd = boostedTree.cv_results_['std_test_score']
    scores_sd = np.array(scores_sd).reshape(len(params['learning_rate']),len(params['n_estimators']))

    # Plot Validation Curves / Model Complexity Curves from GS results
    _, ax = plt.subplots(1,1)

    # Param1 is the X-axis, Param 2 is represented as a different curve (color line)
    for idx, val in enumerate(params['learning_rate']):
        ax.plot(params['n_estimators'], scores_mean[idx,:], '-o', label= param2 + ': ' + str(val))

    ax.set_title("Boosted DT Validation Curves / Model Complexity")
    ax.set_xlabel(param1)
    ax.set_ylabel('3-Fold Cross-validation Mean F1 Score')
    ax.legend(loc="best")
    plt.show()
    return boostedTree.best_params_['n_estimators'], boostedTree.best_params_['learning_rate']

# ====== Boosted Tree on Contraceptive Methods Dataset ====== #
# Try some manual hyperparameter tuning, if you please
# boostedDT(X_train_contra, y_train_contra, X_test_contra, y_test_contra, max_depth=max_depth_contra, datasetName="Contraceptives Methods")
# boostedDT(X_train=X_train_res_ekg, y_train=y_train_res_ekg, X_test=X_test_ekg, y_test=y_test_ekg, max_depth=max_depth_ekg, datasetName="Arrhythmia")

# Get the hyperparameters via GridSearchCV and build the classifier / estimator
max_depth_contra_boosted = 1  # many weak classifiers refine themselves sequentially
max_features_contra = 9  # 9 is the number of features, we don't wait to constrain this tree

n_estimators_contra, learning_rate_contra = final_boosted_DT(X_train_contra, y_train_contra, max_depth_contra_boosted, max_features=max_features_contra, param1='n_estimators', param2='learning_rate')
contra_boosted_estimator = GradientBoostingClassifier(random_state=17, max_features=max_features_contra, n_estimators=n_estimators_contra, learning_rate=learning_rate_contra, min_samples_leaf=4, max_depth=max_depth_contra_boosted)
contra_train_sizes, contra_train_scores_mean, contra_train_time_mean, contra_test_time_mean = plot_learning_curve(estimator=contra_boosted_estimator, learner='Boosted Decision Tree', dataset="Contraceptive Methods", X_train=X_train_contra, y_train=y_train_contra)
contra_boost_test_f1, contra_boost_test_acc, contra_boost_test_precision, contra_boost_test_recall, contra_boost_train_time, contra_boost_test_time = evaluate_classifier(classifier=contra_boosted_estimator, X_train=X_train_contra, X_test=X_test_contra, y_train=y_train_contra, y_test=y_test_contra, learner='Boosted Decision Tree', dataset="Contraceptive Methods", class_names=contra_targets, feature_names=contra_features)
contra_boost_results = ['contra_boost_results', contra_boost_test_f1, contra_boost_test_acc, contra_boost_test_precision, contra_boost_test_recall, contra_boost_train_time, contra_boost_test_time]
print('Leaves count on random tree: %d' % contra_boosted_estimator.estimators_[17, 0].get_n_leaves())
plot_random_tree(clf=contra_boosted_estimator, featureNames=contra_features, fileName="Boosted Contraceptives DT.png", X_train=X_train_contra, y_train=y_train_contra)
print("Done with Boosted Decision Tree for Contraceptives.")

# ====== Boosted Tree on Arrhythmia Dataset ====== #
max_depth_ekg_boosted = 4  # many weak classifiers refine themselves sequentially
max_features_ekg = 14  # too much dimensionality, we can't converge without this

# Get the hyperparameters via GridSearchCV and build the classifier / estimator
n_estimators_ekg, learning_rate_ekg = final_boosted_DT(X_train_res_ekg, y_train_res_ekg, max_depth_ekg_boosted, max_features=max_features_ekg, param1='n_estimators', param2='learning_rate')
ekg_boosted_estimator = GradientBoostingClassifier(random_state=17, max_features=max_features_ekg, n_estimators=n_estimators_ekg, learning_rate=learning_rate_ekg, max_depth=max_depth_ekg_boosted)
ekg_train_sizes, ekg_train_scores_mean, ekg_train_time_mean, ekg_test_time_mean = plot_learning_curve(estimator=ekg_boosted_estimator, learner='Boosted Decision Tree', dataset="Arrhythmia (EKG)", X_train=X_ekg, y_train=y_ekg, cv=4)
ekg_boost_test_f1, ekg_boost_test_acc, ekg_boost_test_precision, ekg_boost_test_recall, ekg_boost_train_time, ekg_boost_test_time = evaluate_classifier(classifier=ekg_boosted_estimator, X_train=X_train_res_ekg, X_test=X_test_ekg, y_train=y_train_res_ekg, y_test=y_test_ekg, learner='Boosted Decision Tree', dataset="Arrhythmia (EKG)", class_names=ekg_classes, feature_names=ekg_features)
ekg_boost_results = ['ekg_boost_results', ekg_boost_test_f1, ekg_boost_test_acc, ekg_boost_test_precision, ekg_boost_test_recall, ekg_boost_train_time, ekg_boost_test_time]
print('Leaves count on random tree' % ekg_boosted_estimator.estimators_[17, 0].get_n_leaves())
plot_random_tree(clf=ekg_boosted_estimator, featureNames=ekg_features, fileName="Boosted Arrhythmia DT.png", X_train=X_train_res_ekg, y_train=y_train_res_ekg)
print("Done with Boosted Decision Tree for Arrhythmia.")

# ===============================
# ### k-Nearest Neighbor Learner
# ===============================
from sklearn.neighbors import KNeighborsClassifier

def knn_manual_tuning(X_train, y_train, X_test, y_test, datasetName):
    f1_test = []
    f1_train = []
    n_neighbors = np.arange(1, 50, 1)
    for n in n_neighbors:  # set param n_neighbors=n
    # distance_power = [1, 2, 3] # power parameter in Minkowski metric. 1: Manhattan distance, 2: euclidean
    # for d in distance_power:  # set param p=d
    # weights = ['uniform', 'distance']
    # for w in weights: # set param weights=w
        clf = KNeighborsClassifier(n_jobs=-1, p=1)
        clf.fit(X_train, y_train)
        y_pred_train = clf.predict(X_train)
        y_pred_test = clf.predict(X_test)
        f1_train.append(f1_score(y_train, y_pred_train, average='weighted'))
        f1_test.append(f1_score(y_test, y_pred_test, average='weighted'))

    plt.plot(n_neighbors, f1_train, 'o-', color='b', label='Train F1 Score')
    plt.plot(n_neighbors, f1_test, 'o-', color='r', label='Test F1 Score')

    plt.ylabel('Model F1 Score')
    plt.xlabel('n_neighbors')

    plt.title("Analysis of n_neighbors Effect on KNN for: " + datasetName)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()

def final_knn(X_train, y_train, param1, param2):
    warnings.filterwarnings('ignore', 'Solver terminated early.*')
    warnings.filterwarnings('ignore', 'F-score is ill-defined.*')

    # Parameters to optimize
    params = {}
    params['n_neighbors'] = np.arange(2, 50, 2)
    params['p'] = [1,2,3] # power parameter in Minkowski metric. 1: Manhattan distance, 2: euclidean

    bestKNN = GridSearchCV(KNeighborsClassifier(), scoring='f1_weighted', param_grid=params, cv=3, n_jobs=-1)
    bestKNN.fit(X_train, y_train)
    print("Using GridSearchCV for hyperparameter tuning on the kNN, we find that the best parameters are:")
    print(bestKNN.best_params_)

    # Get Test Scores Mean and std for each grid search
    scores_mean = bestKNN.cv_results_['mean_test_score']
    scores_mean = np.array(scores_mean).reshape(len(params['p']),len(params['n_neighbors']))

    scores_sd = bestKNN.cv_results_['std_test_score']
    scores_sd = np.array(scores_sd).reshape(len(params['p']),len(params['n_neighbors']))

    # Plot Validation Curves / Model Complexity Curves from GS results
    _, ax = plt.subplots(1,1)

    # Param1 is the X-axis, Param 2 is represented as a different curve (color line)
    for idx, val in enumerate(params['p']):
        ax.plot(params['n_neighbors'], scores_mean[idx,:], '-o', label= param2 + ': ' + str(val))

    ax.set_title("kNN Validation Curves / Model Complexity")
    ax.set_xlabel(param1)
    ax.set_ylabel('3-Fold Cross-validation Mean F1 Score')
    ax.legend(loc="best")
    plt.show()

    return bestKNN.best_params_['p'], bestKNN.best_params_['n_neighbors']

# ======== kNN for Both Datasets ========== #
# Try some manual hyperparameter tuning, if you please
# KNN(X_train_contra, y_train_contra, X_test_contra, y_test_contra, datasetName="Contraceptives Methods")
# KNN(X_train=X_train_res_ekg, y_train=y_train_res_ekg, X_test=X_test_ekg, y_test=y_test_ekg, datasetName="Arrhythmia")

# Contraceptives
contra_knn_p, contra_knn_nneighbors = final_knn(X_train=X_train_contra, y_train=y_train_contra, param1='n_neighbors', param2='p')
contra_knn_estimator = KNeighborsClassifier(n_jobs=-1, p=contra_knn_p, n_neighbors=contra_knn_nneighbors)
contra_train_sizes, contra_train_scores_mean, contra_train_time_mean, contra_test_time_mean = plot_learning_curve(estimator=contra_knn_estimator, learner='k-Nearest Neighbor', dataset="Contraceptive Methods", X_train=X_train_contra, y_train=y_train_contra)
contra_knn_test_f1, contra_knn_test_acc, contra_knn_test_precision, contra_knn_test_recall, contra_knn_train_time, contra_knn_test_time = evaluate_classifier(classifier=contra_knn_estimator, X_train=X_train_contra, X_test=X_test_contra, y_train=y_train_contra, y_test=y_test_contra, learner='k-Nearest Neighbor', dataset="Contraceptive Methods", class_names=contra_targets, feature_names=contra_features)
contra_knn_results = ['contra_knn_results', contra_knn_test_f1, contra_knn_test_acc, contra_knn_test_precision, contra_knn_test_recall, contra_knn_train_time, contra_knn_test_time]

# Arrhythmia
ekg_knn_p, ekg_knn_nneighbors = final_knn(X_train=X_train_res_ekg, y_train=y_train_res_ekg, param1='n_neighbors', param2='p')
ekg_knn_estimator = KNeighborsClassifier(n_jobs=-1, p=ekg_knn_p, n_neighbors=ekg_knn_nneighbors)
ekg_train_sizes, ekg_train_scores_mean, ekg_train_time_mean, ekg_test_time_mean = plot_learning_curve(estimator=ekg_knn_estimator, learner='k-Nearest Neighbor', dataset="Arrhythmia (EKG)", X_train=X_ekg, y_train=y_ekg, cv=4)
ekg_knn_test_f1, ekg_knn_test_acc, ekg_knn_test_precision, ekg_knn_test_recall, ekg_knn_train_time, ekg_knn_test_time = evaluate_classifier(classifier=ekg_knn_estimator, X_train=X_train_res_ekg, X_test=X_test_ekg, y_train=y_train_res_ekg, y_test=y_test_ekg, learner='k-Nearest Neighbor', dataset="Arrhythmia (EKG)", class_names=ekg_classes, feature_names=ekg_features)
ekg_knn_results = ['ekg_knn_results', ekg_knn_test_f1, ekg_knn_test_acc, ekg_knn_test_precision, ekg_knn_test_recall, ekg_knn_train_time, ekg_knn_test_time]

print("Done with KNN for both Contraceptives and Arrhythmia.")

# =======================
# ### Neural Networks!!!
# =======================
from sklearn.neural_network import MLPClassifier

def nn_manual_tuning(X_train, y_train, X_test, y_test, datasetName):
    f1_test = []
    f1_train = []
    # hidden_layers = np.arange(10, 50)  # The ith element represents the number of neurons in the ith hidden layer. Default(100,)
    # for hls in hidden_layers:  # set param hidden_layer_sizes=(hls,)
    # activation = ['identity', 'logistic', 'tanh', 'relu'] # activation function for hidden layer
    # for a in activation:  # set param activation=a
    solver = ['lbfgs', 'sgd', 'adam'] # The solver for weight optimization.
    for s in solver:  # set solver=s
    # alpha = np.linspace(0.0001, .4, 30) # L2 penalty (regularization term) parameter.
    # for a in alpha:  # set alpha=a
    # learning_rate = ['constant', 'invscaling', 'adaptive'] # Only used when solver = 'sgd' Learning rate schedule for weight updates.
    # for lr in learning_rate:  # set learning_rate=lr
    # learning_rate_init = np.linspace(0.0001, .4, 30) # only used on solver = 'sgd' or 'adam'
    # for lri in learning_rate_init:  # set learning_rate_init=lri
    # max_iter = np.linspace(10, 1000, 50).astype('int') # Maximum number of iterations. Default: 200
    # for max in max_iter:  # set max_iter=max
            clf = MLPClassifier(solver=s, max_iter=2000, hidden_layer_sizes=(150,))
            clf.fit(X_train, y_train)
            y_pred_train = clf.predict(X_train)
            y_pred_test = clf.predict(X_test)
            f1_train.append(f1_score(y_train, y_pred_train, average='weighted'))
            f1_test.append(f1_score(y_test, y_pred_test, average='weighted'))

    plt.plot(solver, f1_train, 'o-', color='b', label='Train F1 Score')
    plt.plot(solver, f1_test, 'o-', color='r', label='Test F1 Score')

    plt.ylabel('Model F1 Score')
    plt.xlabel('solver')

    plt.title("solver Effect on Neural Network for " + datasetName)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()

def nn_first_tuning(X_train, y_train, param1, param2):
    warnings.filterwarnings('ignore', 'Solver terminated early.*')
    warnings.filterwarnings('ignore', 'F-score is ill-defined.*')

    # Parameters to optimize
    params = {}
    params['solver'] = ['lbfgs', 'sgd', 'adam']
    params['learning_rate'] = ['constant', 'invscaling', 'adaptive']

    bestNN = GridSearchCV(MLPClassifier(random_state=17, max_iter=1000), scoring='f1_weighted', param_grid=params, cv=3, n_jobs=-1)
    bestNN.fit(X_train, y_train)
    print("Using GridSearchCV for hyperparameter tuning on the Neural Network, we find that the best parameters are:")
    print(bestNN.best_params_)

    # Get Test Scores Mean and std for each grid search
    scores_mean = bestNN.cv_results_['mean_test_score']
    scores_mean = np.array(scores_mean).reshape(len(params['learning_rate']),len(params['solver']))

    scores_sd = bestNN.cv_results_['std_test_score']
    scores_sd = np.array(scores_sd).reshape(len(params['learning_rate']),len(params['solver']))

    # Plot Validation Curves / Model Complexity Curves from GS results
    _, ax = plt.subplots(1,1)

    # Param1 is the X-axis, Param 2 is represented as a different curve (color line)
    for idx, val in enumerate(params['learning_rate']):
        ax.plot(params['solver'], scores_mean[idx, :], '-o', label=param2 + ': ' + str(val))

    ax.set_title("Validation Curves / Model Complexity")
    ax.set_xlabel(param1)
    ax.set_ylabel('3-Fold Cross-validation Mean F1 Score')
    ax.legend(loc="best")
    plt.show()

    return bestNN.best_params_['learning_rate'], bestNN.best_params_['solver']

def final_nn(X_train, y_train, param1, param2, solver, max_iter):
    warnings.filterwarnings('ignore', 'Solver terminated early.*')
    warnings.filterwarnings('ignore', 'F-score is ill-defined.*')

    # Parameters to optimize
    params = {}
    params['hidden_layer_sizes'] = [(10, 10, 10), (30, 30, 30), (20, 50), (10,), (30,), (70,), (100,), (150,), (500,)]  # np.linspace(5, 150, 10).astype('int')  # The ith element represents the number of neurons in the ith hidden layer. Default(100,)
    params['activation'] = ['identity', 'logistic', 'tanh', 'relu']

    bestNN = GridSearchCV(MLPClassifier(random_state=17, solver=solver, max_iter=max_iter), scoring='f1_weighted', param_grid=params, cv=3, n_jobs=-1)
    bestNN.fit(X_train, y_train)
    print("Using GridSearchCV for hyperparameter tuning on the Neural Network, we find that the best parameters are:")
    print(bestNN.best_params_)

    # Get Test Scores Mean and std for each grid search
    scores_mean = bestNN.cv_results_['mean_test_score']
    scores_mean = np.array(scores_mean).reshape(len(params['activation']),len(params['hidden_layer_sizes']))

    scores_sd = bestNN.cv_results_['std_test_score']
    scores_sd = np.array(scores_sd).reshape(len(params['activation']),len(params['hidden_layer_sizes']))

    # Plot Validation Curves / Model Complexity Curves from GS results
    _, ax = plt.subplots(1,1)

    # Param1 is the X-axis, Param 2 is represented as a different curve (color line)
    for idx, val in enumerate(params['activation']):
        ax.plot(scores_mean[idx, :], '-o', label=param2 + ': ' + str(val))

    ax.set_title("Neural Network Validation Curves / Model Complexity")
    ax.set_xlabel(param1)
    newTicks = ['-'] + params['hidden_layer_sizes']
    ax.set_xticklabels(newTicks, rotation=30)
    ax.set_ylabel('3-Fold Cross-validation Mean F1 Score')
    ax.legend(loc="best")
    plt.show()

    return bestNN.best_params_['activation'], bestNN.best_params_['hidden_layer_sizes']

def plot_iterations_nn(X_train, y_train, estimator):
    warnings.filterwarnings('ignore', 'Solver terminated early.*')
    warnings.filterwarnings('ignore', 'F-score is ill-defined.*')

    params = {}
    params['max_iter'] = np.linspace(10, 2500, 30).astype('int')

    bestNN = GridSearchCV(estimator=estimator, param_grid=params, cv=3, scoring='f1_weighted', n_jobs=-1, return_train_score=True)
    bestNN.fit(X_train, y_train)

    train_scores_mean = np.array(bestNN.cv_results_['mean_train_score'])
    train_scores_std = np.array(bestNN.cv_results_['std_train_score'])
    test_scores_mean = np.array(bestNN.cv_results_['mean_test_score'])
    test_scores_std = np.array(bestNN.cv_results_['std_test_score'])

    plt.figure
    plt.title('Impact of Iterations on Model Score')
    plt.xlabel("Number of Iterations")
    plt.ylabel("3-Fold Cross Validated Model F1 Score")
    plt.fill_between(params['max_iter'], train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,color="r")
    plt.fill_between(params['max_iter'], test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(params['max_iter'], train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(params['max_iter'], test_scores_mean, 'o-', color="g", label="Cross-validation testing score")
    plt.legend(loc="best")

    # show the plot!
    plt.show()

# # ==== Neural Network for Both Datasets ==== #
# nn_manual_tuning(X_train_contra, y_train_contra, X_test_contra, y_test_contra, datasetName="Contraceptives Methods")
# nn_manual_tuning(X_train=X_train_res_ekg, y_train=y_train_res_ekg, X_test=X_test_ekg, y_test=y_test_ekg, datasetName="Arrhythmia")

# Contraceptives hyperparameter tuning (via x-validation) and final classifier evalulation
contra_nn_activation, contra_nn_hls = final_nn(X_train=X_train_contra, y_train=y_train_contra, solver='lbfgs', param2='activation', param1='hidden_layer_size', max_iter=200)
contra_nn_estimator = MLPClassifier(activation=contra_nn_activation, max_iter=200, solver='lbfgs', hidden_layer_sizes=contra_nn_hls, random_state=17)
plot_iterations_nn(X_train=X_train_contra, y_train=y_train_contra, estimator=contra_nn_estimator)
contra_train_sizes, contra_train_scores_mean, contra_train_time_mean, contra_test_time_mean = plot_learning_curve(estimator=contra_nn_estimator, learner='Neural Network', dataset="Contraceptive Methods", X_train=X_train_contra, y_train=y_train_contra)
contra_nn_test_f1, contra_nn_test_acc, contra_nn_test_precision, contra_nn_test_recall, contra_nn_train_time, contra_nn_test_time = evaluate_classifier(classifier=contra_nn_estimator, X_train=X_train_contra, X_test=X_test_contra, y_train=y_train_contra, y_test=y_test_contra, learner='Neural Network', dataset="Contraceptive Methods", class_names=contra_targets, feature_names=contra_features)
contra_nn_results = ['contra_nn_results', contra_nn_test_f1, contra_nn_test_acc, contra_nn_test_precision, contra_nn_test_recall, contra_nn_train_time, contra_nn_test_time]

# Arrhythmia x-validation and final classifier evalulation
ekg_nn_activation, ekg_nn_hls = final_nn(X_train=X_train_res_ekg, y_train=y_train_res_ekg, solver='lbfgs', param2='activation', param1='hidden_layer_size', max_iter=1000)
ekg_nn_estimator = MLPClassifier(activation=ekg_nn_activation, max_iter=400, solver='lbfgs', hidden_layer_sizes=ekg_nn_hls)
plot_iterations_nn(X_train=X_train_res_ekg, y_train=y_train_res_ekg, estimator=ekg_nn_estimator)
ekg_train_sizes, ekg_train_scores_mean, ekg_train_time_mean, ekg_test_time_mean = plot_learning_curve(estimator=ekg_nn_estimator, learner='Neural Network', dataset="Arrhythmia (EKG)", X_train=X_ekg, y_train=y_ekg, cv=4)
ekg_nn_test_f1, ekg_nn_test_acc, ekg_nn_test_precision, ekg_nn_test_recall, ekg_nn_train_time, ekg_nn_test_time = evaluate_classifier(classifier=ekg_nn_estimator, X_train=X_train_res_ekg, X_test=X_test_ekg, y_train=y_train_res_ekg, y_test=y_test_ekg, learner='Neural Network', dataset="Arrhythmia (EKG)", class_names=ekg_classes, feature_names=ekg_features)
ekg_nn_results = ['ekg_nn_results', ekg_nn_test_f1, ekg_nn_test_acc, ekg_nn_test_precision, ekg_nn_test_recall, ekg_nn_train_time, ekg_nn_test_time]

print("Done with Neural Networks for both Contraceptives and Arrhythmia.")

# ============================
# ### Support Vector Machines
# ============================
from sklearn.svm import SVC

def svm_manual_tuning(X_train, y_train, X_test, y_test, datasetName):
    f1_test = []
    f1_train = []
    degree = [2] # Degree of the polynomial kernel function (‘poly’).
    kernel = ['linear', 'poly', 'rbf', 'sigmoid']  # default is 'rbf'
    for k in kernel:  # set param kernel=k
        if k in 'poly':
            for d in degree:  # set degree=d, only used for kernal=='poly' but ignored by others. Default degree is 3.
                clf = SVC(kernel=k, degree=d, random_state=17, max_iter=350)
                clf.fit(X_train, y_train)
                y_pred_train = clf.predict(X_train)
                y_pred_test = clf.predict(X_test)
                f1_train.append(f1_score(y_train, y_pred_train, average='weighted'))
                f1_test.append(f1_score(y_test, y_pred_test, average='weighted'))
        else:
            clf = SVC(kernel=k, random_state=17, max_iter=350)
            clf.fit(X_train, y_train)
            y_pred_train = clf.predict(X_train)
            y_pred_test = clf.predict(X_test)
            f1_train.append(f1_score(y_train, y_pred_train, average='weighted'))
            f1_test.append(f1_score(y_test, y_pred_test, average='weighted'))

    x_labels = ['linear', '2nd-deg-poly', 'rbf', 'sigmoid']
    plt.plot(x_labels, f1_train, 'o-', color='b', label='Train F1 Score')
    plt.plot(x_labels, f1_test, 'o-', color='r', label='Test F1 Score')

    plt.ylabel('Model F1 Score')
    plt.xlabel('Kernel')

    plt.title("Kernel Effect on SVM for " + datasetName)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()

def final_svm(X_train, y_train, param1, param2, kernel):
    warnings.filterwarnings('ignore', 'Solver terminated early.*')
    warnings.filterwarnings('ignore', 'F-score is ill-defined.*')

    # Parameters to optimize
    params = {}
    params['C'] = np.linspace(0.0001, 1.0, 10).astype('float')  # The penalty parameter of the error term, Default = 1.0
    params['gamma'] = [0.1, 1, 10, 100] # Kernel coefficient for ‘rbf’, ‘poly’ and ‘sigmoid’.

    bestSVC = GridSearchCV(estimator=SVC(kernel=kernel, random_state=17, max_iter=10000), scoring='f1_weighted', param_grid=params, cv=3, n_jobs=-1)
    bestSVC.fit(X_train, y_train)
    print("Using GridSearchCV for hyperparameter tuning on the SVM, we find that the best parameters are:")
    print(bestSVC.best_params_)

    # Get Test Scores Mean and std for each grid search
    scores_mean = bestSVC.cv_results_['mean_test_score']
    scores_mean = np.array(scores_mean).reshape(len(params['gamma']),len(params['C']))

    scores_sd = bestSVC.cv_results_['std_test_score']
    scores_sd = np.array(scores_sd).reshape(len(params['gamma']),len(params['C']))

    # Plot Validation Curves / Model Complexity Curves from GS results
    _, ax = plt.subplots(1,1)

    # Param1 is the X-axis, Param 2 is represented as a different curve (color line)
    for idx, val in enumerate(params['gamma']):
        ax.plot(params['C'], scores_mean[idx,:], '-o', label= param2 + ': ' + str(val))

    ax.set_title("SVM Validation Curves / Model Complexity")
    ax.set_xlabel(param1)
    ax.set_ylabel('3-Fold Cross-validation Mean F1 Score')
    ax.legend(loc="best")
    ax.grid('on')
    plt.show()

    return bestSVC.best_params_['C'], bestSVC.best_params_['gamma']

def plot_iterations_svm(X_train, y_train, estimator):
    warnings.filterwarnings('ignore', 'Solver terminated early.*')
    warnings.filterwarnings('ignore', 'F-score is ill-defined.*')

    # Parameters to optimize
    params = {}
    params['max_iter'] = np.linspace(10, 7500, 30).astype('int')

    bestSVM = GridSearchCV(estimator=estimator, param_grid=params, cv=3, scoring='f1_weighted', n_jobs=-1, return_train_score=True)
    bestSVM.fit(X_train, y_train)

    train_scores_mean = np.array(bestSVM.cv_results_['mean_train_score'])
    train_scores_std = np.array(bestSVM.cv_results_['std_train_score'])
    test_scores_mean = np.array(bestSVM.cv_results_['mean_test_score'])
    test_scores_std = np.array(bestSVM.cv_results_['std_test_score'])

    plt.title('Impact of Iterations on Model Score')
    plt.xlabel("Number of Iterations")
    plt.ylabel("3-Fold Cross Validated Model F1 Score")
    plt.fill_between(params['max_iter'], train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,color="r")
    plt.fill_between(params['max_iter'], test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(params['max_iter'], train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(params['max_iter'], test_scores_mean, 'o-', color="g", label="Cross-validation testing score")
    plt.legend(loc="best")

    plt.show()

# # ==== SVMs for Both Datasets ==== #
# svm_manual_tuning(X_train_contra, y_train_contra, X_test_contra, y_test_contra, datasetName="Contraceptive Methods")
# svm_manual_tuning(X_train=X_train_res_ekg, y_train=y_train_res_ekg, X_test=X_test_ekg, y_test=y_test_ekg, datasetName="Arrhythmia")

# From manual hyperparameter tuning, we found the following optimized hyperparameter values
kernel1 = 'rbf'
kernel2 = 'linear'

# Contraceptives x-validation and final classifier evaluation

# Using Kernel = 'rbf'
contra_svc_c, contra_svc_gamma = final_svm(X_train=X_train_contra, y_train=y_train_contra, param1='C', param2='gamma', kernel=kernel1)
contra_svc_estimator_1 = SVC(kernel=kernel1, C=contra_svc_c, gamma=contra_svc_gamma, random_state=17, max_iter=10000)
plot_iterations_svm(X_train=X_train_contra, y_train=y_train_contra, estimator=contra_svc_estimator_1)
contra_train_sizes, contra_train_scores_mean, contra_train_time_mean, contra_test_time_mean = plot_learning_curve(estimator=contra_svc_estimator_1 , learner='SVM', dataset="Contraceptive Methods", X_train=X_train_contra, y_train=y_train_contra)
contra_svm_k1_test_f1, contra_svm_k1_test_acc, contra_svm_k1_test_precision, contra_svm_k1_test_recall, contra_svm_k1_train_time, contra_svm_k1_test_time = evaluate_classifier(classifier=contra_svc_estimator_1, X_train=X_train_contra, X_test=X_test_contra, y_train=y_train_contra, y_test=y_test_contra, learner='SVM', dataset="Contraceptive Methods", class_names=contra_targets, feature_names=contra_features)
contra_svm_k_rbf_results = ['contra_svm_k_rbf_results', contra_svm_k1_test_f1, contra_svm_k1_test_acc, contra_svm_k1_test_precision, contra_svm_k1_test_recall, contra_svm_k1_train_time, contra_svm_k1_test_time]

# Using Kernel = 'linear'
contra_svc_c, contra_svc_gamma = final_svm(X_train=X_train_contra, y_train=y_train_contra, param1='C', param2='gamma', kernel=kernel2)
contra_svc_estimator_2 = SVC(kernel=kernel2, degree=3, random_state=17, max_iter=10000)
plot_iterations_svm(X_train=X_train_contra, y_train=y_train_contra, estimator=contra_svc_estimator_2)
contra_train_sizes, contra_train_scores_mean, contra_train_time_mean, contra_test_time_mean = plot_learning_curve(estimator=contra_svc_estimator_1 , learner='SVM', dataset="Contraceptive Methods", X_train=X_train_contra, y_train=y_train_contra)
contra_svm_k2_test_f1, contra_svm_k2_test_acc, contra_svm_k2_test_precision, contra_svm_k2_test_recall, contra_svm_k2_train_time, contra_svm_k2_test_time = evaluate_classifier(classifier=contra_svc_estimator_2, X_train=X_train_contra, X_test=X_test_contra, y_train=y_train_contra, y_test=y_test_contra, learner='SVM', dataset="Contraceptive Methods", class_names=contra_targets, feature_names=contra_features)
contra_svm_k_linear_results = ['contra_svm_k_linear_results', contra_svm_k2_test_f1, contra_svm_k2_test_acc, contra_svm_k2_test_precision, contra_svm_k2_test_recall, contra_svm_k2_train_time, contra_svm_k2_test_time]

# Arrhythmia x-validation and final classifier evaluation

# Using Kernel = 'rbf'
ekg_svc_c, ekg_svc_gamma = final_svm(X_train=X_train_res_ekg, y_train=y_train_res_ekg, param1='C', param2='gamma', kernel=kernel1)
ekg_svc_estimator_1 = SVC(kernel=kernel1, C=ekg_svc_c, gamma=ekg_svc_gamma, random_state=17, max_iter=10000)
plot_iterations_svm(X_train=X_train_res_ekg, y_train=y_train_res_ekg, estimator=ekg_svc_estimator_1)
ekg_train_sizes, ekg_train_scores_mean, ekg_train_time_mean, ekg_test_time_mean = plot_learning_curve(estimator=ekg_svc_estimator_1, learner='SVM', dataset="Arrhythmia (EKG)", X_train=X_ekg, y_train=y_ekg, cv=4)
ekg_svm_k1_test_f1, ekg_svm_k1_test_acc, ekg_svm_k1_test_precision, ekg_svm_k1_test_recall, ekg_svm_k1_train_time, ekg_svm_k1_test_time = evaluate_classifier(classifier=ekg_svc_estimator_1, X_train=X_train_res_ekg, X_test=X_test_ekg, y_train=y_train_res_ekg, y_test=y_test_ekg, learner='SVM', dataset="Arrhythmia (EKG)", class_names=ekg_classes, feature_names=ekg_features)
ekg_svm_k_rbf_results = ['ekg_svm_k_rbf_results', ekg_svm_k1_test_f1, ekg_svm_k1_test_acc, ekg_svm_k1_test_precision, ekg_svm_k1_test_recall, ekg_svm_k1_train_time, ekg_svm_k1_test_time]

# Using Kernel = 'linear'
ekg_svc_c, ekg_svc_gamma = final_svm(X_train=X_train_res_ekg, y_train=y_train_res_ekg, param1='C', param2='gamma', kernel=kernel2)
ekg_svc_estimator_2 = SVC(kernel=kernel2, C=ekg_svc_c, gamma=ekg_svc_gamma, random_state=17, max_iter=10000)
plot_iterations_svm(X_train=X_train_res_ekg, y_train=y_train_res_ekg, estimator=ekg_svc_estimator_2)
ekg_train_sizes, ekg_train_scores_mean, ekg_train_time_mean, ekg_test_time_mean = plot_learning_curve(estimator=ekg_svc_estimator_2, learner='SVM', dataset="Arrhythmia (EKG)", X_train=X_ekg, y_train=y_ekg, cv=4)
ekg_svm_k2_test_f1, ekg_svm_k2_test_acc, ekg_svm_k2_test_precision, ekg_svm_k2_test_recall, ekg_svm_k2_train_time, ekg_svm_k2_test_time = evaluate_classifier(classifier=ekg_svc_estimator_2, X_train=X_train_res_ekg, X_test=X_test_ekg, y_train=y_train_res_ekg, y_test=y_test_ekg, learner='SVM', dataset="Arrhythmia (EKG)", class_names=ekg_classes, feature_names=ekg_features)
ekg_svm_k_linear_results = ['ekg_svm_k_linear_results', ekg_svm_k2_test_f1, ekg_svm_k2_test_acc, ekg_svm_k2_test_precision, ekg_svm_k2_test_recall, ekg_svm_k2_train_time, ekg_svm_k2_test_time]

print("Done with SVMs for both Contraceptives and Arrhythmia.")

#========= Tabulate the final data ========#
from tabulate import tabulate

print(tabulate([contra_dt_results, contra_boost_results, contra_knn_results, contra_nn_results, contra_svm_k_rbf_results, contra_svm_k_linear_results], headers=['Learner', 'F1_Weighted', 'Accuracy', 'Precision', 'Recall', 'Training time (ms)', 'Prediction time (ms)']))
print(tabulate([ekg_dt_results, ekg_boost_results, ekg_knn_results, ekg_nn_results, ekg_svm_k_rbf_results, ekg_svm_k_linear_results], headers=['Learner', 'F1_Weighted', 'Accuracy', 'Precision', 'Recall', 'Training time (ms)', 'Prediction time (ms)']))
