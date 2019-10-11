import numpy as np
import pandas as pd

'''NOT COMPLETE!!

For preprocessing data prior to running ML algorithms. This method opens a CSV, creates a dataframe,
one hot encodes both the features and labels, splits the data into test and train sets, and scales the data.

Returns: X_train, X_test, y_train, y_test numpy nd arrays'''

def load_data(csv_filename, dataset_name):
    # open and create a pandas dataframe
    df = pd.read_csv(csv_filename, na_values="?")
    df.apply(pd.to_numeric)

    # Check the frame (first 5 rows) to ensure all the columns are there
    df.head()
    df.describe(include='all')
    print("%s dataset size: ", len(df), "instances and", len(df.columns) - 1, "features." %(dataset_name))

def arrange_features(): #NOT COMPLETE!
    # (2A) one-hot encode the columns where numerical categories are not ordinal
    col_to_one_hot = ['Wifes_religion', 'Wifes_now_working%3F', 'Husbands_occupation', 'Media_exposure']
    df_one_hot = contra_df[col_to_one_hot]
    df_one_hot = pd.get_dummies(df_one_hot)  # converts series to dummy codes
    df_remaining_cols = contra_df.drop(col_to_one_hot, axis=1)  # the rest of the dataframe is maintained
    contra_df = pd.concat([df_one_hot, df_remaining_cols], axis=1)

    # (2B) move results (output) column to the front for consistency
    column_order = list(contra_df)
    column_order.insert(0, column_order.pop(column_order.index('Contraceptive_method_used')))
    contra_df = contra_df.loc[:, column_order]  # move the target variable to the front for consistency

    # (2C) Prove that we didn't mess it up
    contra_df.describe(include='all')  # see the changes

    # (2D) Grab the classes from the y-data (which look like numbers but actually have more semantic meaning to be used later.
    contra_targets = ['No-use', "Long-term", "Short-term"]

# =====================
# ### Data loading ###
# =====================

load_data(csv_filename='PimaDiabetesDataset.csv', dataset_name="Pima Diabetes")

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