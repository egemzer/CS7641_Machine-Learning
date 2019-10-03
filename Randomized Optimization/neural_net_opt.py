import mlrose
import timeit
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
matplotlib.use('TkAgg')
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix


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

#======================#
# ===== Data Prep =====#
#======================#
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# get the data and segment into training and testing
testSize = 0.15  # default is 0.25, manually tune if interesting

# Pull the data from the dataframe
X_contra, y_contra, contra_features = import_data(contra_df)
X_train_contra, X_test_contra, y_train_contra, y_test_contra = train_test_split(np.array(X_contra), np.array(y_contra), test_size=testSize, random_state=17)

# We need to scale both datasets for Neural Network to perform well.
# Note that the training will be done on the rebalanced/resampled training dataset
# But evaluation will be done on the original holdout test set with NO resampling.
contra_scaler = StandardScaler()
X_train_contra = contra_scaler.fit_transform(X_train_contra)
X_test_contra = contra_scaler.transform(X_test_contra)


def build_model(algorithm, hidden_nodes, activation):
    nn_model = mlrose.NeuralNetwork(hidden_nodes=hidden_nodes, activation=activation, \
                                        algorithm=algorithm, max_iters=1000, \
                                        bias=True, is_classifier=True, learning_rate=0.0001, \
                                        early_stopping=True, clip_max=5, max_attempts=1000, \
                                        random_state=17)
    return nn_model

def evaluate_model(nn_model, search_algo, X_train, X_test, y_train, y_test, class_names, feature_names):
    # Training time
    start_time = timeit.default_timer()
    nn_model.fit(X=X_train, y=y_train)
    end_time = timeit.default_timer()
    training_time = (end_time - start_time) * 1000  # milliseconds

    # Prediction Time
    start_time = timeit.default_timer()
    y_pred = nn_model.predict(X_test)
    end_time = timeit.default_timer()
    pred_time = (end_time - start_time) * 1000  # milliseconds

    # Predict labels for train set and assess accuracy
    y_train_pred = nn_model.predict(X_train)
    y_train_accuracy = accuracy_score(y_train, y_train_pred)

    # Predict labels for test set and assess accuracy
    y_test_pred = nn_model.predict(X_test)
    y_test_accuracy = accuracy_score(y_test, y_test_pred)

    # Distribution of Predictions
    unique, counts = np.unique(y_pred, return_counts=True)
    res_classes = dict(zip(unique, counts))
    plt.bar(*zip(*res_classes.items()))
    plt.xticks((1, 2, 3), labels=class_names)
    plt.title("Distribution of Predictions, over all Classes")
    plt.show()

# Standard Metrics
    f1 = f1_score(y_test, y_pred, average='weighted', labels=class_names)  # Use average = 'micro', 'macro' or 'weighted' since we have non-binary classes. Don't count classes that are never predicted.
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', labels=np.unique(y_pred))
    recall = recall_score(y_test, y_pred, average='weighted', labels=np.unique(y_pred))

    print("Metrics for the Candidate Neural Network using the %s Search Algorithm" %search_algo)
    print("Model Training Time (ms):   " + "{:.1f}".format(training_time))
    print("Model Prediction Time (ms): " + "{:.1f}\n".format(pred_time))
    print("F1 Score:  " + "{:.2f}".format(f1))
    print("Accuracy:  " + "{:.2f}".format(accuracy))
    print("Precision: " + "{:.2f}".format(precision))
    print("Recall:    " + "{:.2f}".format(recall))
    return "{0:.2f}".format(f1), "{0:.2f}".format(accuracy), "{0:.2f}".format(precision), "{0:.2f}".format(recall), "{:.1f}".format(training_time), "{:.1f}".format(pred_time)


# Random Hill Climbing
nn_rhc_model = build_model(algorithm='random_hill_climb', hidden_nodes=[150], activation='relu')
rhc_nn_test_f1, rhc_nn_test_acc, rhc_nn_test_precision, rhc_nn_test_recall, rhc_nn_train_time, rhc_nn_test_time = evaluate_model(nn_rhc_model, 'RHC', X_train_contra, X_test_contra, y_train_contra, y_test_contra, contra_targets, contra_features)

print("You got here!")