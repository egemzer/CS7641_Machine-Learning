# ###CS7641 - Machine Learning Project 1 code: Supervised Learning ###
# python3
# GT alias: gth659q
import timeit
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# In this file, there is code that allows for the creation of five different classification learners
# on two different UCI datasets:

# Datasets: Contraceptive Method Choice, Arrhythmia
# Learners: Decision Tree, Neural Network, Boosting, Support Vector Machines (SVM), and k-Nearest Neighbors (KNN)

# Each model will output two plots: learning curve and model complexity based on https://www.dataquest.io/blog/learning-curves-machine-learning/.


# =====================================
# ### Data loading (also in README) ###
# =====================================
# IMPORTANT! To run this code, you need to save the datasets to your local machine, in the same folder as this code.

# STEP 1: Download the Contraceptives Dataset: https://www.openml.org/d/23 as a CSV (or from my Github) into the same folder as this code

# open and create a pandas dataframe
contra_df = pd.read_csv('ContraceptivesMethodsData.csv', na_values = "?")
# contra_df = contra_df.astype('int64').dtypes
contra_df.apply(pd.to_numeric)

# Check the frame (first 5 rows) to ensure all the columns are there
contra_df.head()
contra_df.describe(include='all')
print("Contraceptives Methods dataset size: ",len(contra_df),"instances and", len(contra_df.columns) - 1,"features.")

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
print("Arrhythmia dataset size: ", len(ekg_df), "patients and", len(ekg_df.columns) - 1, "features.")

# STEP 3: Now we need to clean up the data for more effective downstream processing / learning
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

# (4) Grab the classes from the y-data (which look like numbers but actually have more semantic meaning to be used later.
contra_targets = ['No-use', "Long-term", "Short-term"]

# STEP 4: Cleaning up Arrhythmia data (less trivial given >200 attributes and output labels that are not ordinal, though most of the values are linear and ordinal)
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

# (4) Grab the classes from the y-data (which look like numbers but actually have more semantic meaning to be used later.
ekg_classes = ['Normal', 'Arrhythmia - Class 2', 'Arrhythmia - Class 3', 'Arrhythmia - Class 4', 'Arrhythmia - Class 5', 'Arrhythmia - Class 6',
               'Arrhythmia - Class 7', 'Arrhythmia - Class 8', 'Arrhythmia - Class 9', 'Arrhythmia - Class 10', 'Arrhythmia - Class 14', 'Arrhythmia - Class 15', 'Arrhythmia - Class 16 (Unclassified)']

print("Hi, friend, you did all the data preprocessing with no errors!")

#========================================================================
# ### Helper functions that will be leveraged across all the learners ###
#========================================================================
from sklearn.model_selection import train_test_split, GridSearchCV

def import_data(df1, df2):
    X1 = np.array(df1.values[:,1:]) # get all rows, and all columns after the first (y) column
    Y1 = np.array(df1.values[:,0]) # get all rows and just the first (index = 0) column of ys
    X2 = np.array(df2.values[:,1:])
    Y2 = np.array(df2.values[:,0])
    features_1 = df1.columns.values[1:]
    features_2 = df2.columns.values[1:]
    return X1, Y1, X2, Y2, features_1, features_2

from sklearn.model_selection import learning_curve, cross_validate
def plot_learning_curve(estimator, learner, dataset, X, y , ylim=None, cv=3,
                        n_jobs=-1):
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
        ``-1`` means using all processors.

    train_sizes : array-like, shape (n_ticks,), dtype float or int
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. (default: np.linspace(0.1, 1.0, 5))

    Adapted from: https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html
    """

    #Provision empty arrays to fill
    train_scores_mean = []; train_scores_std = []
    test_scores_mean = []; test_scores_std = []
    train_time_mean = []; train_time_std = []
    test_time_mean =[]; test_time_std = []

    train_sizes = (np.linspace(.1, 1.0, 5) * len(y)).astype('int')
    for size in train_sizes:
        index = np.random.randint(X.shape[0], size = size)
        y_slice = y[index]
        X_slice = X[index,:]
        cv_results = cross_validate(estimator, X_slice, y_slice, cv=cv, n_jobs=n_jobs, return_train_score=True)

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

    fig, (ax1, ax2) = plt.subplots(1, 2)
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


from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import itertools

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

        plt.figure(figsize=(8, 6))
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(learner + " Confusion Matrix for: " + dataset)
        plt.colorbar()

        if target_names is not None:
            tick_marks = np.arange(len(target_names))
            plt.xticks(tick_marks, target_names, rotation=45)
            plt.yticks(tick_marks, target_names)

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        thresh = cm.max() / 1.5 if normalize else cm.max() / 2
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            if normalize:
                if i == min(range(cm.shape[0])):
                    plt.text(j, i + 0.25, "{0:0.1%}".format(cm[i, j]),
                             horizontalalignment="center",
                             color="white" if cm[i, j] > thresh else "black")
                elif i == max(range(cm.shape[0])):
                    plt.text(j, i - 0.25, "{0:0.1%}".format(cm[i, j]),
                             horizontalalignment="center",
                             color="white" if cm[i, j] > thresh else "black")
                else:
                    plt.text(j, i, "{0:0.1%}".format(cm[i, j]),
                             horizontalalignment="center",
                             color="white" if cm[i, j] > thresh else "black")

            else:
                plt.text(j, i + 0.25, "{:,}".format(cm[i, j]),
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label\naccuracy={:0.3f}; misclass={:0.3f}'.format(accuracy, misclass))
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
    training_time = end_time - start_time

    # Prediction Time
    start_time = timeit.default_timer()
    y_pred = classifier.predict(X_test)
    end_time = timeit.default_timer()
    pred_time = end_time - start_time

    # Count predictions
    unique, counts = np.unique(y_pred, return_counts=True)
    res_classes = dict(zip(unique, counts))
    plt.bar(*zip(*res_classes.items()))
    plt.title("Distribution of Predictions, over all Classes")
    plt.show()

    # See the Tree
    plot_tree(classifier, featureNames=feature_names, fileName='FinalTree.png')

    # Standard Metrics
    f1 = f1_score(y_test, y_pred, average='macro') #  Use average = 'macro' or 'weighted' since we have non-binary classes. Macro will work best on the unbalanced EKG data by penalizing the model if it doesn't predict minority classes well
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    plot_confusion_matrix(y_test, y_pred, learner=learner, dataset=dataset, target_names=class_names)
    # auc = roc_auc_score(y_test, y_pred, average='macro') # may not use, it's not particularly well suited to non-binary classification

    print("Metrics: Candidate Classifier on the Test Dataset")
    print("Model Training Time (s):   " + "{:.5f}".format(training_time))
    print("Model Prediction Time (s): " + "{:.5f}\n".format(pred_time))
    print("F1 Score:  " + "{:.2f}".format(f1))
    print("Check the confusion plot!")
    print("Accuracy:  " + "{:.2f}".format(accuracy))
    print("Precision: " + "{:.2f}".format(precision))
    print ("Recall:    " + "{:.2f}".format(recall))


# =========================================
# ### Decision Tree Learner (with Pruning)
# =========================================
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

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

def plot_tree(clf, featureNames, fileName):
    dot_data = export_graphviz(clf, feature_names=featureNames, filled=True, rounded=True, special_characters=True)
    graph = pydotplus.graph_from_dot_data(dot_data)
    graph.write_png(fileName)

from sklearn.tree import export_graphviz
import pydotplus

# manual tuning approach leveraged from: https://www.kaggle.com/drgilermo/playing-with-the-knobs-of-sklearn-decision-tree
# Decision tree builder that also does pruning and calculates F1 score (due to non-binary, multi-classification task, see: https://medium.com/usf-msds/choosing-the-right-metric-for-evaluating-machine-learning-models-part-2-86d5649a5428)
def decisionTree (X_train, y_train, X_test, y_test, datasetName):

    f1_score_test = []
    f1_score_train = []
    # commented out code below as used for manually tuning hyperparameters for each dataset
    # split_criteria = ["gini", "entropy"]
    # for criteria in split_criteria:
    # max_depth = list(range(1, 30))
    # for d in max_depth:
    # min_samples_leaf = list(range(1,201))
    # for msl in min_samples_leaf:
    # max_leaf_nodes = list(range(50, int(len(y_train) / 2), 10))
    # for mln in max_leaf_nodes:
    # min_samples_split = list(np.arange(0.05, 1, .01))
    # for mss in min_samples_split:
    clf = tree.DecisionTreeClassifier(criterion='gini', min_samples_split= mss)
    clf = clf.fit(X_train, y_train)

    # Prune the tree
    # clf_pruned = prune(clf, threshold=2)
    # print('Leaves count before pruning: %d' % clf.get_n_leaves())
    # print('Leaves count after pruning: %d' % clf_pruned.get_n_leaves())

    # Test it, plot it, and store the F1 score
    y_pred_test = clf.predict(X_test)
    y_pred_train = clf.predict(X_train)
    f1_score_test.append(f1_score(y_test, y_pred_test,average='macro'))
    f1_score_train.append(f1_score(y_train, y_pred_train, average='macro'))

    # plt.plot(min_samples_split, f1_score_test, 'o-', color='r', label='Test F1 Score')
    # plt.plot(min_samples_split, f1_score_train, 'o-', color='b', label='Train F1 Score')
    # plt.ylabel('Model F1 Score')
    # plt.xlabel('min_samples_split')
    #
    # plt.title("Analysis of min_samples_split Effect on DT for: " + datasetName)
    # plt.legend(loc='best')
    # plt.show()
    # print("you got here!")

def tuningGridSearch(X_train, y_train):
    min_samples_leaf, max_samples_leaf = get_samples_leaf(len(y_train))
    max_depth_absolute = 16 # should be small as a means of pre-pruning
    num_classes = len(np.unique(y_train, return_counts=True)[0])

    # Parameters to optimize
    params = {}
    params['max_depth'] = np.arange(9, max_depth_absolute, 1)
    params['max_leaf_nodes'] = np.arange(10, int(len(y_train)/num_classes), 10)

    tree = GridSearchCV(estimator = DecisionTreeClassifier(criterion='gini'), param_grid=params, cv=3, verbose=1, n_jobs=-1)
    tree.fit(X_train, y_train)
    print("Using GridSearchCV for hyperparameter tuning, we find that the best parameters are:")
    print(tree.best_params_)
    return tree.best_params_['max_depth'], tree.best_params_['max_leaf_nodes']

# =======================================
# ### Decision Tree building and analysis

# get the data and segment into training and testing
testSize = 0.25 # can manually tune if interesting
X_contra, y_contra, X_ekg, y_ekg, contra_features, ekg_features = import_data(contra_df, ekg_df)
X_train_contra, X_test_contra, y_train_contra, y_test_contra = train_test_split(np.array(X_contra), np.array(y_contra), test_size=testSize)

# ====================== Contraceptive Methods Dataset ========================= #
# # Try some manual hyperparameter tuning, if you please
# decisionTree(X_train_contra, y_train_contra, X_test_contra, y_test_contra, "Contraceptives Methods")

# Get the hyperparameters for Contraceptive Methods via GridSearchCV and build the classifier / estimator
max_depth_contra, max_leaf_nodes_contra = tuningGridSearch(X_train_contra, y_train_contra)
contra_estimator = DecisionTreeClassifier(criterion='gini', max_depth=max_depth_contra, min_samples_leaf=9, max_leaf_nodes=max_leaf_nodes_contra)
contra_train_sizes, contra_train_scores_mean, contra_train_time_mean, contra_test_time_mean = plot_learning_curve(estimator=contra_estimator, learner='Decision Tree', dataset="Contraceptive Methods", X=X_contra, y=y_contra)
evaluate_classifier(classifier=contra_estimator, X_train=X_train_contra, X_test=X_test_contra, y_train=y_train_contra, y_test=y_test_contra, learner='Decision Tree', dataset="Contraceptive Methods", class_names=contra_targets, feature_names=contra_features)
print('Leaves count before pruning: %d' % contra_estimator.get_n_leaves())
plot_tree(clf=contra_estimator, featureNames=contra_features, fileName="Pre-pruned Contraceptives DT.png")

# Prune the classifier, plot learning curves, and evaluate the final classifier
pruned_contra_clf = prune(contra_estimator, threshold=5)
print('Leaves count after pruning: %d' % pruned_contra_clf.get_n_leaves())
plot_tree(clf=pruned_contra_clf, featureNames=contra_features, fileName="Post-pruned Contraceptives DT.png")
contra_train_sizes, contra_train_scores_mean, contra_train_time_mean, contra_test_time_mean = plot_learning_curve(estimator=pruned_contra_clf, learner='Decision Tree', dataset="Contraceptive Methods", X=X_contra, y=y_contra)
evaluate_classifier(classifier=pruned_contra_clf, X_train=X_train_contra, X_test=X_test_contra, y_train=y_train_contra, y_test=y_test_contra, learner='Decision Tree', dataset="Contraceptive Methods", class_names=contra_targets, feature_names=contra_features)

print("Done with Contraceptives Decision Tree, starting Arrhythmia Decision Tree.")
# ====================== Arrhythmia Dataset ========================= #
# First, we need to resample. Since the majority class is more than 50% of the dataset AND all other classes represent <10% AND our total dataset size (rows) is <1000, we need to oversample the minority classes. Random usually works better than picking.
from imblearn.over_sampling import RandomOverSampler
X_train_ekg, X_test_ekg, y_train_ekg, y_test_ekg = train_test_split(np.array(X_ekg), np.array(y_ekg), test_size=testSize)
ros1 = RandomOverSampler(sampling_strategy='not majority')
X_train_res_ekg, y_train_res_ekg = ros1.fit_resample(X_train_ekg, y_train_ekg) # Split the data into test and train BEFORE resampling

ros2 = RandomOverSampler(sampling_strategy='not majority')
X_test_res_ekg, y_test_res_ekg = ros2.fit_resample(X_test_ekg, y_test_ekg)
ros3 = RandomOverSampler(sampling_strategy='not majority')
X_res_ekg, y_res_ekg = ros3.fit_resample(X_ekg, y_ekg)

# # Prove that you've correctly resampled
# print('Resampled training dataset shape %s' % len(y_train_res_ekg))
# unique, counts = np.unique(y_train_res_ekg, return_counts=True)
# res_classes = dict(zip(unique, counts))
# plt.bar(*zip(*res_classes.items()))
# plt.title("Re (Over) Sampling of Training Data")
# plt.show()
# print('Resampled test dataset shape %s' % len(y_test_res_ekg))
# unique, counts = np.unique(y_test_res_ekg, return_counts=True)
# res_classes = dict(zip(unique, counts))
# plt.bar(*zip(*res_classes.items()))
# plt.title("Re (Over) Sampling of Test Data")
# plt.show()

# # Try some manual hyperparameter tuning, if you please
# decisionTree(X_train=X_train_ekg, y_train=y_train_ekg, X_test=X_test_ekg, y_test=y_test_ekg, datasetName="Arrhythmia (EKG) Classification")

# Get the hyperparameters for Contraceptive Methods via GridSearchCV and build the classifier / estimator
max_depth_ekg, max_leaf_nodes_ekg = tuningGridSearch(X_train_res_ekg, y_train_res_ekg)
ekg_estimator = DecisionTreeClassifier(criterion='gini', max_depth=max_depth_ekg, max_leaf_nodes=max_leaf_nodes_ekg)
ekg_train_sizes, ekg_train_scores_mean, ekg_train_time_mean, ekg_test_time_mean = plot_learning_curve(estimator=ekg_estimator, learner='Decision Tree', dataset="Arrhythmia (EKG)", X=X_res_ekg, y=y_res_ekg)
evaluate_classifier(classifier=ekg_estimator, X_train=X_train_res_ekg, X_test=X_test_res_ekg, y_train=y_train_res_ekg, y_test=y_test_res_ekg, learner='Decision Tree', dataset="Arrhythmia (EKG)", class_names=ekg_classes, feature_names=ekg_features)
print('Leaves count before pruning: %d' % ekg_estimator.get_n_leaves())
plot_tree(clf=ekg_estimator, featureNames=ekg_features, fileName="Pre-pruned Arrhythmia (EKG) DT.png")

# Prune the classifier, plot learning curves, and evaluate the final classifier
pruned_ekg_clf = prune(ekg_estimator, threshold=1)
print('Leaves count after pruning: %d' % pruned_ekg_clf.get_n_leaves())
plot_tree(clf=pruned_ekg_clf, featureNames=ekg_features, fileName="Post-pruned Arrhythmia DT.png")
ekg_train_sizes, ekg_train_scores_mean, ekg_train_time_mean, ekg_test_time_mean = plot_learning_curve(estimator=pruned_ekg_clf, learner='Decision Tree', dataset="Arrhythmia (EKG)", X=X_res_ekg, y=y_res_ekg)
evaluate_classifier(classifier=pruned_ekg_clf, X_train=X_train_res_ekg, X_test=X_test_res_ekg, y_train=y_train_res_ekg, y_test=y_test_res_ekg, learner='Decision Tree', dataset="Arrhythmia (EKG)", class_names=ekg_classes, feature_names=ekg_features)
print("Done with Arrhythmia Decision Tree")

# =================================================
# ### Boosted Decision Tree Learner (with Pruning)
# =================================================