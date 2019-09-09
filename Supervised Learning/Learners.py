# ###CS7641 - Machine Learning Project 1 code: Supervised Learning ###

import matplotlib.pyplot as plt
import pandas as pd
import sklearn
import seaborn as sns

from sklearn import tree
from sklearn.tree import DecisionTreeClassifier, export_graphviz

# In this file, there is code that allows for the creation of five different classification learners
# on two different UCI datasets:

# Datasets: Contraceptive Method Choice, Arrhythmia
# Learners: Decision Tree, Neural Network, Boosting, Support Vector Machines (SVM), and k-Nearest Neighbors (KNN)

# Each model will output two plots: learning curve and model complexity based on https://www.dataquest.io/blog/learning-curves-machine-learning/.



# ### Data loading (also in README) ###

# IMPORTANT! To run this code, you need to save the datasets to your local machine, in a folder you like.
# Then change the current directory to the location / folder where you saved the datasets.

# STEP 1: Download the Contraceptives Dataset: https://www.openml.org/d/23 as a CSV (or from my Github)

# open and create a pandas dataframe
contra_df = pd.read_csv('ContraceptivesMethodsData.csv').astype('category')

# Check the frame (first 5 rows) to ensure all the columns are there
contra_df.head()
contra_df.describe(include='all')
print("Data size: ",len(contra_df),"rows and", len(contra_df.columns),"columns.")

# Check for missing data
if contra_df.isnull().sum() > 0:
    print("Warning: Missing data in the contraceptives dataset!")

# STEP 2: Download the Arrhythmia Dataset: https://www.openml.org/d/5 as a CSV (or from my Github)

# open and create a pandas dataframe
ekg_df = pd.read_csv('ArrhythmiaData.csv').astype('category')

# Check the frame (first 5 rows) to ensure all the columns are there
ekg_df.head()
ekg_df.describe(include='all')
print("Data size: ", len(ekg_df), "rows and", len(ekg_df.columns), "columns.")

# Check for missing data
if ekg_df.isnull().sum() > 0:
    print("Warning: Missing Data in the arrhythmia dataset!")


# ### Helper functions that will be leveraged across all the learners ###