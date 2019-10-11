import mlrose
import timeit
import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import datetime
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')

hidden_nodes = [10, 10, 10]
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

# STEP 2: Now we need to clean up the data for more effective downstream processing / learning
# (1) for any non-ordinal numerical data, we will one hot encode based on the principle researched here:
# https://machinelearningmastery.com/why-one-hot-encode-data-in-machine-learning/
# (2) we will move the results column to be the first column for simplicity

# ## Cleaning up Contraceptives data
# (2A) one-hot encode the columns where numerical categories are not ordinal
col_to_one_hot = ['Wifes_religion', 'Wifes_now_working%3F', 'Husbands_occupation', 'Media_exposure']
df_one_hot = contra_df[col_to_one_hot]
df_one_hot = pd.get_dummies(df_one_hot) # converts series to dummy codes
df_remaining_cols = contra_df.drop(col_to_one_hot, axis=1) # the rest of the dataframe is maintained
contra_df = pd.concat([df_one_hot, df_remaining_cols], axis=1)

# (2B) move results (output) column to the front for consistency
column_order = list(contra_df)
column_order.insert(0, column_order.pop(column_order.index('Contraceptive_method_used')))
contra_df = contra_df.loc[:, column_order]  # move the target variable to the front for consistency

# (2C) Prove that we didn't mess it up
contra_df.describe(include='all') # see the changes

# (2D) Grab the classes from the y-data (which look like numbers but actually have more semantic meaning to be used later.
contra_targets = ['No-use', "Long-term", "Short-term"]


#================================#
# ======= Helper Functions ======#
#================================#
def import_data(df1):
    X1 = np.array(df1.values[:,1:]) # get all rows, and all columns after the first (y) column
    Y1 = np.array(df1.values[:,0]) # get all rows and just the first (index = 0) column of ys
    features = df1.columns.values[1:]
    return X1, Y1, features

# ===== Data Pre-Processing =====#
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# get the data and segment into training and testing
testSize = 0.25  # default is 0.25, manually tune if interesting

# Pull the data from the dataframe
X_contra, y_contra, contra_features = import_data(contra_df)

# One hot encode target values
from sklearn.preprocessing import OneHotEncoder
one_hot = OneHotEncoder(categories='auto')
y_hot_contra = one_hot.fit_transform(y_contra.reshape(-1, 1)).todense()

# split the data into train and test sets
X_train_contra, X_test_contra, y_hot_train_contra, y_hot_test_contra = train_test_split(np.array(X_contra), np.array(y_hot_contra), test_size=testSize, random_state=17)

# We need to scale both datasets for Neural Network to perform well.
# Note that the training will be done on the rebalanced/resampled training dataset
# But evaluation will be done on the original holdout test set with NO resampling.
contra_scaler = StandardScaler()
X_train_contra = contra_scaler.fit_transform(X_train_contra)
X_test_contra = contra_scaler.transform(X_test_contra)

def build_model(algorithm, hidden_nodes, activation, schedule=mlrose.GeomDecay(init_temp=5000), restarts=75,
                population=300, mutation=0.2, max_iters=20000, learning_rate=0.01):
    nn_model = mlrose.neural.NeuralNetwork(hidden_nodes=hidden_nodes, activation=activation,
                                        algorithm=algorithm, max_iters=max_iters,
                                        bias=True, is_classifier=True, learning_rate=learning_rate,
                                        early_stopping=False, restarts = restarts, clip_max=1,
                                        schedule=schedule,
                                        max_attempts=1000, pop_size=population, mutation_prob=mutation,
                                        random_state=17, curve=True)
    return nn_model

from sklearn.model_selection import cross_validate
import warnings
def plot_learning_curve(estimator, search_algo, dataset, X_train, y_train, cv=5):
    """
    Generate a simple plot of the test and training learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    search_algo : string
        Type of search algorithm, goes into the title for the chart.

    dataset : string
        Name of dataset, goes into the title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;

    cv : int, cross-validation generator or an iterable, optional. Default: 5

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

    train_sizes = (np.linspace(.25, 1.0, cv) * len(y_train)).astype('int')
    for size in train_sizes:
        index = np.random.randint(X_train.shape[0], size = size)
        y_slice = y_train[index]
        X_slice = X_train[index,:]
        cv_results = cross_validate(estimator, X_slice, y_slice, cv=cv, scoring='f1_weighted', return_train_score=True, n_jobs=-1)

        # Fill out out data arrays with actual results from the cross validator
        train_scores_mean.append(np.mean(cv_results['train_score'])); train_scores_std.append(np.std(cv_results['train_score']))
        test_scores_mean.append(np.mean(cv_results['test_score'])); test_scores_std.append(np.std(cv_results['test_score']))
        train_time_mean.append(np.mean(cv_results['fit_time'])); train_time_std.append(np.std(cv_results['fit_time']))
        test_time_mean.append(np.mean(cv_results['score_time'])); test_time_std.append(np.std(cv_results['score_time']))

    # Convert arrays to numpy arrays for later math operations
    train_scores_mean = np.array(train_scores_mean); train_scores_std = np.array(train_scores_std)
    test_scores_mean = np.array(test_scores_mean); test_scores_std = np.array(test_scores_std)
    train_time_mean = np.array(train_time_mean); train_time_std = np.array(train_time_std)
    test_time_mean = np.array(test_time_mean); test_time_std = np.array(test_time_std)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    fig.suptitle(search_algo + " Learning Curves for: " + dataset)
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
    ax2.set(xlabel="Number of Training Examples", ylabel="Model Time, in seconds")
    ax2.grid()
    ax2.fill_between(train_sizes, train_time_mean - train_time_std,
                     train_time_mean + train_time_std, alpha=0.1,
                     color="r")
    ax2.fill_between(train_sizes, test_time_mean - test_time_std,
                     test_time_mean + test_time_std, alpha=0.1, color="g")
    ax2.plot(train_sizes, train_time_mean, 'o-', color="r",
             label="Training Time (s)")
    ax2.plot(train_sizes, test_time_mean, 'o-', color="g",
             label="Prediction Time (s)")
    ax2.legend(loc="best")

    # show the plots!
    plt.show()

    return train_sizes, train_scores_mean, train_time_mean, test_time_mean


def evaluate_model(nn_model, search_algo, X_train, X_test, y_train, y_test, class_names, feature_names):
    # Training time
    start_time = timeit.default_timer()
    nn_model.fit(X=X_train, y=y_train)
    end_time = timeit.default_timer()
    training_time = (end_time - start_time) # seconds
    nn_fitness_curve = nn_model.fitness_curve
    df1 = pd.DataFrame(y_test)
    df1.to_csv('df_y_test.csv', index=False)

    # Prediction Time
    start_time = timeit.default_timer()
    y_pred = nn_model.predict(X_test)
    end_time = timeit.default_timer()
    pred_time = (end_time - start_time)  # seconds
    df2 = pd.DataFrame(y_pred)
    df2.to_csv('df_y_pred.csv', index=False)

    # Distribution of Predictions
    decoded = []
    for i in range(0, len(y_pred)):
        if tuple(y_pred[i]) == (1,0,0):
            decoded.insert(i, class_names[0])
        elif tuple(y_pred[i]) == (0,1,0):
            decoded.insert(i, class_names[1])
        elif tuple(y_pred[i]) == (0,0,1):
            decoded.insert(i, class_names[2])

    unique, counts = np.unique(decoded, return_counts=True)
    res_classes = dict(zip(unique, counts))
    fig, ax = plt.subplots()
    plt.bar(*zip(*res_classes.items()))
    plt.xticks((0, 1, 2), labels=class_names)
    ax.title.set_text("Distribution of Predictions, over all Classes")
    plt.show()

# Standard Metrics
    f1 = f1_score(y_test, y_pred, average='weighted', labels=np.unique(y_test))  # Use average = 'micro', 'macro' or 'weighted' since we have non-binary classes. Don't count classes that are never predicted.
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', labels=np.unique(y_test))
    recall = recall_score(y_test, y_pred, average='weighted', labels=np.unique(y_test))

    print("Metrics for the Candidate Neural Network using the %s Search Algorithm" %(search_algo))
    print("Model Training Time (ms):   " + "{:.3f}".format(training_time))
    print("Model Prediction Time (ms): " + "{:.3f}\n".format(pred_time))
    print("F1 Score:  " + "{:.2f}".format(f1))
    print("Accuracy:  " + "{:.2f}".format(accuracy))
    print("Precision: " + "{:.2f}".format(precision))
    print("Recall:    " + "{:.2f}".format(recall))
    return "{0:.2f}".format(f1), "{0:.2f}".format(accuracy), "{0:.2f}".format(precision), "{0:.2f}".format(recall), "{:.1f}".format(training_time), "{:.1f}".format(pred_time), nn_fitness_curve


# Gradient Descent
time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
print("Starting Gradient Descent Learning Curves at: " + time)
nn_gd_model = build_model(algorithm='gradient_descent', hidden_nodes=hidden_nodes, activation='relu',
                          max_iters=5000, learning_rate=0.001)

gd_contra_train_sizes, gd_contra_train_scores_mean, gd_contra_train_time_mean, gd_contra_test_time_mean = \
     plot_learning_curve(estimator=nn_gd_model, search_algo='Gradient Descent', dataset="Contraceptive Methods", X_train=X_train_contra, y_train=y_hot_train_contra, cv=3)

time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
print("Starting Gradient Descent Model Evaluation at: " + time)
gd_nn_test_f1, gd_nn_test_acc, gd_nn_test_precision, gd_nn_test_recall, gd_nn_train_time, rhc_nn_test_time, nn_gd_fitness = \
    evaluate_model(nn_model=nn_gd_model, search_algo='Gradient Descent', X_train=X_train_contra, X_test=X_test_contra, y_train=y_hot_train_contra,
                   y_test=y_hot_test_contra, class_names=contra_targets, feature_names=contra_features)


# Random Hill Climbing
time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
print("Starting Random Hill Climbing Learning Curves at: " + time)
nn_rhc_model = build_model(algorithm='random_hill_climb', hidden_nodes=hidden_nodes, activation='relu',
                           restarts=75, max_iters=20000, learning_rate=0.01)

rhc_contra_train_sizes, rhc_contra_train_scores_mean, rhc_contra_train_time_mean, rhc_contra_test_time_mean = \
    plot_learning_curve(estimator=nn_rhc_model, search_algo='RHC', dataset="Contraceptive Methods", X_train=X_train_contra, y_train=y_hot_train_contra, cv=3)

time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
print("Starting Random Hill Climbing Model Evaluation at: " + time)
rhc_nn_test_f1, rhc_nn_test_acc, rhc_nn_test_precision, rhc_nn_test_recall, rhc_nn_train_time, rhc_nn_test_time, nn_rhc_fitness = \
    evaluate_model(nn_model=nn_rhc_model, search_algo='RHC', X_train=X_train_contra, X_test=X_test_contra, y_train=y_hot_train_contra,
                   y_test=y_hot_test_contra, class_names=contra_targets, feature_names=contra_features)

# Genetic Algorithms
time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
print("Starting Genetic Algorithms Learning Curves at: " + time)
nn_ga_model = build_model(algorithm='genetic_alg', hidden_nodes=hidden_nodes, activation='relu',
                          population=300, mutation=0.2, max_iters=5000, learning_rate=0.01)

ga_contra_train_sizes, ga_contra_train_scores_mean, ga_contra_train_time_mean, ga_contra_test_time_mean = \
    plot_learning_curve(estimator=nn_ga_model, search_algo='GA', dataset="Contraceptive Methods", X_train=X_train_contra, y_train=y_hot_train_contra, cv=3)

time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
print("Starting Genetic Algorithms Model Evaluation at: " + time)
ga_nn_test_f1, ga_nn_test_acc, ga_nn_test_precision, ga_nn_test_recall, ga_nn_train_time, ga_nn_test_time, nn_ga_fitness = \
    evaluate_model(nn_model=nn_ga_model, search_algo='GA', X_train=X_train_contra, X_test=X_test_contra, y_train=y_hot_train_contra,
                   y_test=y_hot_test_contra, class_names=contra_targets, feature_names=contra_features)

# Simulated Annealing
time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
print("Starting Simulated Annealing Learning Curves at: " + time)
nn_sa_model = build_model(algorithm='simulated_annealing', hidden_nodes=hidden_nodes, activation='relu',
                          schedule=mlrose.algorithms.decay.ArithDecay(init_temp=5000, decay=0.001, min_temp=0.001),
                          max_iters=20000, learning_rate=0.01)

sa_contra_train_sizes, sa_contra_train_scores_mean, sa_contra_train_time_mean, sa_contra_test_time_mean = \
    plot_learning_curve(estimator=nn_sa_model, search_algo='SA', dataset="Contraceptive Methods", X_train=X_train_contra, y_train=y_hot_train_contra, cv=3)

time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
print("Starting Simulated Annealing Model Evaluation at: " + time)
sa_nn_test_f1, sa_nn_test_acc, sa_nn_test_precision, sa_nn_test_recall, sa_nn_train_time, sa_nn_test_time, nn_sa_fitness = \
    evaluate_model(nn_model=nn_sa_model, search_algo='SA', X_train=X_train_contra, X_test=X_test_contra, y_train=y_hot_train_contra,
                   y_test=y_hot_test_contra, class_names=contra_targets, feature_names=contra_features)

#======= Comparison of all four search algorithms ==========#
fig5, (ax9, ax10) = plt.subplots(1, 2, figsize=(15, 5))
fig5.suptitle('Comparing Random Search Optimizers on Neural Network Weight Optimization for Contraceptive Methods Dataset: F1-Score and Convergence Time')

ax9.set(xlabel="Number of Iterations", ylabel="F1-Score")
ax9.grid()
ax9.plot(iterations, nn_gd_fitness, 'o-', color="y",label="Gradient Descent Training Fitness")
ax9.plot(iterations, nn_rhc_fitness, 'o-', color="r",label="RHC Training Fitness")
ax9.plot(iterations, nn_ga_fitness, 'o-', color="b",label="GA Training Fitness")
ax9.plot(iterations, nn_sa_fitness, 'o-', color="m",label="SA Training Fitness")

ax9.legend(loc="best")

ax10.set(xlabel="Number of Iterations", ylabel="Training Time (in seconds)")
ax10.grid()
ax10.plot(iterations, gd_nn_train_time, 'o-', color="y", label='Gradient Descent Training Time')
ax10.plot(iterations, rhc_nn_train_time, 'o-', color="r", label='RHC Training Time')
ax10.plot(iterations, ga_nn_train_time, 'o-', color="b", label='GA Training Time')
ax10.plot(iterations, sa_nn_train_time, 'o-', color="m", label='SA Training Time')
ax10.legend(loc="best")

plt.show()
print("You got here!")

