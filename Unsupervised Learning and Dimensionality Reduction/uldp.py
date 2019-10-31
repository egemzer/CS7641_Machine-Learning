# ###CS7641 - Machine Learning Project 1 code: Supervised Learning ###
# This Python file uses the following encoding: utf-8
# python3
# GT alias: gth659q
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import warnings
import itertools
from sklearn.neural_network import MLPClassifier
import timeit
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix


# This script creates and analyzes performance for six different unsupervised learning clustering and dimensionality reduction algorithms
# on two different UCI / OpenML datasets:
# Datasets: Contraceptive Method Choice, Arrhythmia
# Algorithms: k-means clustering, expectation maximization, PCA, ICA, Randomized Projections

#================================#
# ======= Helper Functions ======#
#================================#
def import_data(df1):
    X1 = np.array(df1.values[:,1:]) # get all rows, and all columns after the first (y) column
    Y1 = np.array(df1.values[:,0]) # get all rows and just the first (index = 0) column of ys
    features = df1.columns.values[1:]
    return X1, Y1, features

def plot_elbows(distortions_ekg, distortions_contra):
    fig1, ax1 = plt.subplots()
    ax1.title.set_text("Elbow Method for Determining Optimal k")
    ax2 = ax1.twinx()
    lns1 = ax1.plot(range(1, maxk + 1), distortions_ekg, marker='o', color="r", label='Arrhythmia')
    lns2 = ax2.plot(range(1, maxk + 1), distortions_contra, marker='o', color="b", label='Contraceptive Methods')
    ax1.set_xlabel('Number of Clusters')
    ax1.set_ylabel('Arrhythmia Distortions: Sum of Squared Distance', color='r')
    ax2.set_ylabel('Contraceptive Methods: Sum of Squared Distance', color='b')
    lns = lns1 + lns2
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc='best')
    return plt

def plot_homogeneity(hom_contra, hom_ekg):
    fig3, ax5 = plt.subplots()
    ax5.title.set_text("Homogeneity Score for Determining Optimal K Value")
    ax6 = ax5.twinx()
    lns1 = ax5.plot(range(2, maxk + 1), hom_ekg, marker='o', color="r", label='Arrhythmia')
    lns2 = ax6.plot(range(2, maxk + 1), hom_contra, marker='o', color="b", label='Contraceptive Methods')
    ax5.set_xlabel('Number of Clusters')
    ax5.set_ylabel('Arrhythmia: Homogeneity Score', color='r')
    ax6.set_ylabel('Contraceptive Methods: Homogeneity Score', color='b')
    lns = lns1 + lns2
    labs = [l.get_label() for l in lns]
    ax5.legend(lns, labs, loc='best')
    return plt

def plot_completeness(compl_contra, compl_ekg):
    fig4, ax7 = plt.subplots()
    ax7.title.set_text("Completeness Score for Determining Optimal K Value")
    ax8 = ax7.twinx()
    lns1 = ax7.plot(range(2, maxk + 1), compl_ekg, marker='o', color="r", label='Arrhythmia')
    lns2 = ax8.plot(range(2, maxk + 1), compl_contra, marker='o', color="b", label='Contraceptive Methods')
    ax7.set_xlabel('Number of Clusters')
    ax7.set_ylabel('Arrhythmia: Completeness Score', color='r')
    ax8.set_ylabel('Contraceptive Methods: Completeness Score', color='b')
    lns = lns1 + lns2
    labs = [l.get_label() for l in lns]
    ax7.legend(lns, labs, loc='best')
    return plt

def plot_silhouette(sil_ekg, sil_contra):
    fig2, ax3 = plt.subplots()
    ax3.title.set_text("Silhouette Score for Determining Global Optimum K Value")
    ax4 = ax3.twinx()
    lns1 = ax3.plot(range(2, maxk + 1), sil_ekg, marker='o', color="r", label='Arrhythmia')
    lns2 = ax4.plot(range(2, maxk + 1), sil_contra, marker='o', color="b", label='Contraceptive Methods')
    ax3.set_xlabel('Number of Clusters')
    ax3.set_ylabel('Arrhythmia: Silhouette Score', color='r')
    ax4.set_ylabel('Contraceptive Methods: Silhouette Score', color='b')
    lns = lns1 + lns2
    labs = [l.get_label() for l in lns]
    ax3.legend(lns, labs, loc='best')
    return plt

def plot_predictions(y_pred, k, dataset_name, estimator):
    plt.figure()
    unique, counts = np.unique(y_pred, return_counts=True)
    res_classes = dict(zip(unique, counts))
    plt.bar(*zip(*res_classes.items()))
    plt.ylabel('Number of Samples')
    plt.xlabel('Cluster ID')
    plt.xticks(range(0,k))
    plt.title("Distribution of Cluster Predictions for %s on %s" %(estimator, dataset_name))
    return plt

def plot_pca_variance(pca):
    fig, ax = plt.subplots()
    ax.title.set_text("Variance Explained by PCA Components")
    ax2 = ax.twinx()
    features = range(pca.n_components_)
    var_cum = np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4))
    ax.bar(features, pca.explained_variance_ratio_, color='black')
    ax2.plot(var_cum, marker='o')
    ax.set_xlabel('PCA features')
    ax.set_xticks(features)
    ax.set_ylabel('By Component Variance %')
    ax2.set_ylabel('Cumulative Variance %')
    return plt

def plot_pca_components(PCA_components, dataset):
    plt.figure()
    plt.title("PCA1 and PCA2 for %s" %(dataset))
    plt.scatter(PCA_components[0], PCA_components[1], alpha=.1)
    plt.xlabel('PCA 1')
    plt.ylabel('PCA 2')
    return plt

def plot_clusters(X_clustered, X_data, dataset_name, estimator_name):
    plt.figure(figsize=(7, 7))
    LABEL_COLOR_MAP = {0: 'b', 1: 'g', 2: 'r', 3: 'c', 4: 'm', 5: 'y', 6: 'k', 7: 'b', 8: 'g', 9: 'r', 10: 'c', 11: 'm', 12: 'y', 13: 'k', 14: 'b', 15: 'g', 16: 'r', 17: 'c', 18: 'm', 19: 'y', 20: 'k', 21: 'b', 22: 'g', 23: 'r', 24: 'c', 25: 'm', 26: 'y', 27: 'k', 28: 'b', 29: 'g', 30: 'r'}
    label_color = [LABEL_COLOR_MAP[l] for l in X_clustered]
    label = ['cluster %s' % (l) for l in X_clustered]
    plt.title("Scatter Plot of Clusters for %s using %s" %(dataset_name, estimator_name))
    plt.scatter(X_data[:, 0], X_data[:, 2], c=label_color, alpha=0.5)
    return plt

def evaluate_k_means(n_clust, X_train, y_train, dataset_name, dr_algo_name):
    """
    Function k_means applies k-means clustering alrorithm on dataset and prints the crosstab of cluster and actual labels
    and clustering performance parameters.

    Input:
    n_clust - number of clusters (k value)
    X_train - dataset we want to cluster
    y_train - original labels

    Output: Metrics / performance table
    Returns: crosstab of cluster and actual labels

    """
    warnings.simplefilter(action='ignore', category=FutureWarning)
    k_means = KMeans(n_clusters=n_clust, init='k-means++', random_state=17, n_init=10, max_iter=300,
        tol=0.0001)
    k_means.fit(X_train)
    c_labels = k_means.labels_
    df = pd.DataFrame({'clust_label': c_labels, 'orig_label': y_train})
    ct = pd.crosstab(df['clust_label'], df['orig_label'])
    y_clust = k_means.predict(X_train)
    print("Stats for K-Means analysis on %s using %s DR algorithm" %(dataset_name, dr_algo_name))
    print('% 9s' % 'inertia  homo    compl   v-meas   ARI     AMI     silhouette')
    print('%i   %.3f   %.3f   %.3f   %.3f   %.3f    %.3f'
          % (k_means.inertia_,
             homogeneity_score(y_train, y_clust),
             completeness_score(y_train, y_clust),
             v_measure_score(y_train, y_clust),
             adjusted_rand_score(y_train, y_clust),
             adjusted_mutual_info_score(y_train, y_clust),
             silhouette_score(X_train, y_clust, metric='euclidean')))

    return ct

def evaluate_em(n_clust, covariance, X_train, y_train, dataset_name, dr_algo_name):
    """
    Function k_means applies k-means clustering alrorithm on dataset and prints the crosstab of cluster and actual labels
    and clustering performance parameters.

    Input:
    n_clust - number of clusters (k value)
    X_train - dataset we want to cluster
    y_train - original labels

    Output: Metrics / performance table
    Returns: crosstab of cluster and actual labels

    """
    warnings.simplefilter(action='ignore', category=FutureWarning)
    gmm = GaussianMixture(n_components=n_clust, covariance_type=covariance, random_state=17)
    gmm.fit(X_train)
    bic = gmm.bic(X_train)
    y_clust = gmm.predict(X_train)
    df = pd.DataFrame({'clust_label': y_clust, 'orig_label': y_train})
    ct = pd.crosstab(df['clust_label'], df['orig_label'])
    print("Stats for EM analysis on %s using %s DR algorithm" %(dataset_name, dr_algo_name))
    print('% 9s' % 'homo    compl   v-meas   ARI     AMI     BIC')
    print('%.3f    %.3f     %.3f   %.3f   %.3f   %.0f'
          % (homogeneity_score(y_train, y_clust),
             completeness_score(y_train, y_clust),
             v_measure_score(y_train, y_clust),
             adjusted_rand_score(y_train, y_clust),
             adjusted_mutual_info_score(y_train, y_clust),
             bic))

    return ct

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

# One Hot Encode features where numerical categories are not ordinal
col_to_one_hot = ['Wifes_religion', 'Wifes_now_working%3F', 'Husbands_occupation', 'Media_exposure']
df_one_hot = contra_df[col_to_one_hot]
df_one_hot = pd.get_dummies(df_one_hot) # converts series to dummy codes
df_remaining_cols = contra_df.drop(col_to_one_hot, axis=1) # the rest of the dataframe is maintained
contra_df = pd.concat([df_one_hot, df_remaining_cols], axis=1)

# move results (output) column to the front for consistency
column_order = list(contra_df)
column_order.insert(0, column_order.pop(column_order.index('Contraceptive_method_used')))
contra_df = contra_df.loc[:, column_order]  # move the target variable to the front for consistency

contra_df.describe(include='all') # Prove that we didn't mess it up

# Grab the classes from the y-data (which look like numbers but actually have more semantic meaning to be used later).
contra_labels = ['No-use', "Long-term", "Short-term"]


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

# One Hot Encode features where numerical categories are not ordinal
col_to_one_hot = ['sex']
df_one_hot = ekg_df[col_to_one_hot]
df_one_hot = pd.get_dummies(df_one_hot) # converts series to dummy codes
df_remaining_cols = ekg_df.drop(col_to_one_hot, axis=1) # the rest of the dataframe is maintained
ekg_df = pd.concat([df_one_hot, df_remaining_cols], axis=1)

# move results (output) column to the front for consistency
column_order = list(ekg_df)
column_order.insert(0, column_order.pop(column_order.index('class')))
ekg_df = ekg_df.loc[:, column_order]  # move the target variable to the front for consistency

ekg_df.describe(include='all')  # Prove that we didn't mess it up

# Collapse all the Arrhythmia classes into a single class, keep Normal, and keep the Unclassified class
# For a total of three prediction classes
ekg_df['class'].replace({3:2, 4:2, 5:2, 6:2, 7:2, 8:2, 9:2, 10:2, 11:2, 12:2, 13:2, 14:2, 15:2, 16:3},inplace=True)

# Grab the classes from the y-data (which look like numbers but actually have more semantic meaning to be used later).
ekg_labels = ['Normal', 'Classified Arrhythmia', 'Unclassified Arrhythmia']

# Shuffle the DF
from sklearn.utils import shuffle
ekg_df = shuffle(ekg_df)


# ===== Data Pre-Processing =====#
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler

testSize = 0.25  # default is 0.25, manually tune if interesting

# Pull the data from the dataframe
X_contra, y_contra, contra_features = import_data(contra_df)
X_ekg, y_ekg, ekg_features = import_data(ekg_df)

# split the data into train and test sets
X_train_contra, X_test_contra, y_train_contra, y_test_contra = train_test_split(np.array(X_contra), np.array(y_contra), test_size=testSize, random_state=17)
X_train_ekg, X_test_ekg, y_train_ekg, y_test_ekg = train_test_split(np.array(X_ekg), np.array(y_ekg), test_size=testSize, random_state=17)
X_train_ekg_not_resampled = X_train_ekg.copy()
y_train_ekg_not_resampled = y_train_ekg.copy()

# We need to resample the Arrhythmia dataset because the majority class is more than 50% of the dataset AND all other classes represent <10% AND our total dataset size (rows) is <1000, we need to oversample the minority classes. Random usually works better than picking.
ros1 = RandomOverSampler(sampling_strategy='not majority')
X_train_ekg, y_train_ekg = ros1.fit_resample(X_train_ekg, y_train_ekg)

# Prove that you've correctly resampled
def confirm_resampling(original_labels):
    print('Resampled training dataset shape %s' % len(y_train_ekg))
    unique, counts = np.unique(y_train_ekg, return_counts=True)
    res_classes = dict(zip(unique, counts))
    train_plot = plt.figure()
    plt.bar(*zip(*res_classes.items()))
    plt.xticks((1,2,3), labels=original_labels)
    plt.title("Resampling (Oversampling) of Training Data")

    test_plot = plt.figure(2)
    unique, counts = np.unique(y_test_ekg, return_counts=True)
    res_classes = dict(zip(unique, counts))
    plt.bar(*zip(*res_classes.items()), color="r")
    plt.xticks((1,2,3), labels=original_labels)
    plt.title("Distribution of Test Data")
    return train_plot, test_plot

# train_resampled_plot, test_untouched_plot = confirm_resampling(original_labels=ekg_labels)

# We need to scale both datasets for Neural Network to perform well.
# Note that the training will be done on the rebalanced/resampled training dataset
# But evaluation will be done on the original holdout test set with NO resampling.
scaler = StandardScaler()
X_train_contra = scaler.fit_transform(X_train_contra.astype(np.float))
X_test_contra = scaler.transform(X_test_contra.astype(np.float))

X_train_ekg = scaler.fit_transform(X_train_ekg.astype(np.float))
X_train_ekg_not_resampled = scaler.fit_transform(X_train_ekg_not_resampled.astype(np.float))
X_test_ekg = scaler.transform(X_test_ekg.astype(np.float))

##########################################
# ####### Non-DR K-Means Clustering ######
##########################################
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, homogeneity_score, completeness_score, v_measure_score, adjusted_rand_score, adjusted_mutual_info_score, silhouette_score
maxk = 30

def elbow(X, max_k):
    # calculate distortion for a range of number of clusters
    distortions = []
    for i in range(1, max_k+1):
        km = KMeans(
            n_clusters=i, init='k-means++',
            n_init=10, max_iter=300,
            tol=0.0001, random_state=17)
        km.fit(X)
        distortions.append(km.inertia_)
    return distortions

def silho(X, max_k):
    sil = []
    for k in range(2, max_k+1):
      km = KMeans(
          n_clusters=k, init='k-means++',
          n_init=10, max_iter=300,
          tol=0.0001, random_state=17)
      kmeans = km.fit(X)
      labels = kmeans.labels_
      sil.append(silhouette_score(X, labels, metric = 'euclidean'))
    return sil

def homogen(X, max_k, y):
    hom = []
    for k in range(2, max_k+1):
      km = KMeans(
          n_clusters=k, init='k-means++',
          n_init=10, max_iter=300,
          tol=0.0001, random_state=17)
      kmeans = km.fit(X)
      y_kmeans = kmeans.predict(X)
      hom.append(homogeneity_score(labels_pred=y_kmeans, labels_true=y))
    return hom

def completeness(X, max_k, y):
    compl = []
    for k in range(2, max_k+1):
      km = KMeans(
          n_clusters=k, init='k-means++',
          n_init=10, max_iter=300,
          tol=0.0001, random_state=17)
      kmeans = km.fit(X)
      compl.append(completeness_score(labels_pred=kmeans.labels_, labels_true=y))
    return compl

def kmeans_accuracy(X, kmeans, y):
    correct = 0
    for i in range(len(X)):
        predict_me = np.array(X[i].astype(float))
        predict_me = predict_me.reshape(-1, len(predict_me))
        prediction = kmeans.predict(predict_me)
        if prediction[0] == y[i]:
            correct += 1
    return(correct / len(X))

# # Use the elbow method to determine the best value of k for each dataset. Use scaled data but not resampled.
# distortions_contra = elbow(X_train_contra, max_k=maxk)
# distortions_ekg = elbow(X_train_ekg_not_resampled, max_k=maxk)
# k_means_elbows_plot = plot_elbows(distortions_contra=distortions_contra, distortions_ekg=distortions_ekg)

k_contra_elbow = 7  # Found in Elbow method
k_ekg_elbow = 19     # Found in Elbow method

# # Complement Elbow method results using the Silhouette Method, which optimizes at the global optimum K value
# sil_contra = silho(X_train_contra, max_k=maxk)
# sil_ekg = silho(X_train_ekg_not_resampled, max_k=maxk)
# k_means_sil_plot = plot_silhouette(sil_contra=sil_contra, sil_ekg=sil_ekg)

k_contra_sil = 6  # Found in Silhouette method
k_ekg_sil = 2     # Found in Silhouette method
k_ekg_sil2 = 5  # Found in Silhouette method

# # Compare results to true labels using homogeneity score
# hom_contra = homogen(X_train_contra, max_k=maxk, y=y_train_contra)
# hom_ekg = homogen(X_train_ekg_not_resampled, max_k=maxk, y=y_train_ekg_not_resampled)
# k_means_hom_plot = plot_homogeneity(hom_contra=hom_contra, hom_ekg=hom_ekg)
#
# # Compare results to true labels using completeness score
# compl_contra = completeness(X_train_contra, max_k=maxk, y=y_train_contra)
# compl_ekg = completeness(X_train_ekg_not_resampled, max_k=maxk, y=y_train_ekg_not_resampled)
# k_means_compl_plot = plot_completeness(compl_contra=compl_contra, compl_ekg=compl_ekg)

k_contra_compl = 3  # Found using Completeness
k_ekg_compl = 5     # Found using Completeness


#####  k-Means for Contra without DR #####
contra_km = KMeans(
            n_clusters=k_contra_elbow, init='k-means++',
            n_init=10, max_iter=300,
            tol=0.0001, random_state=17)
contra_y_km = contra_km.fit_predict(X_train_contra)
# contra_predictions_dist = plot_predictions(y_pred=contra_y_km, k=k_contra_elbow, dataset_name='Contraceptive Methods', estimator='k-means')
# contra_cluster_plot = plot_clusters(X_clustered=contra_y_km, X_data=X_train_contra, dataset_name="Contraceptive Methods", estimator_name='K-Means without DR"')
table_pred_kmeans_contra = evaluate_k_means(n_clust=k_contra_elbow, dataset_name="Contraceptive Methods", X_train=X_train_contra, y_train=y_train_contra, dr_algo_name="No DR")
print(table_pred_kmeans_contra)

#####  k-Means for Arrhythmia with No DR #####
ekg_km = KMeans(
            n_clusters=k_ekg_elbow, init='k-means++',
            n_init=10, max_iter=300,
            tol=0.0001, random_state=17)
ekg_y_km = ekg_km.fit_predict(X_train_ekg_not_resampled)
# ekg_predictions_dist = plot_predictions(y_pred=ekg_y_km, k=k_ekg_elbow, dataset_name='Arrhythmia', estimator='k-means')
# ekg_cluster_plot = plot_clusters(X_clustered=ekg_y_km, X_data=X_train_ekg_not_resampled, dataset_name="Arrhythmia", estimator_name='K-Means without DR"')
table_pred_kmeans_ekg = evaluate_k_means(n_clust=k_ekg_elbow, dataset_name="Arrhythmia", X_train=X_train_ekg_not_resampled, y_train=y_train_ekg_not_resampled, dr_algo_name="No DR")
print(table_pred_kmeans_ekg)


##############################################
####### Non_DR Expectation Maximization ######
##############################################
from sklearn.mixture import GaussianMixture

def bic_analyze(X, clusters):
    lowest_bic = np.infty
    bic = []
    n_components_range = range(1, clusters+1)
    cv_types = ['spherical', 'tied', 'diag', 'full']   # Removed 'diag', 'full' since it's the same as KMeans
    for cv_type in cv_types:
        for n_components in n_components_range:
            # Fit a Gaussian mixture with EM
            gmm = GaussianMixture(n_components=n_components,
                                          covariance_type=cv_type)
            gmm.fit(X)
            bic.append(gmm.bic(X))
            if bic[-1] < lowest_bic:
                lowest_bic = bic[-1]
                best_gmm = gmm

    return np.array(bic), best_gmm


def plot_bic(bic, clusters, dataset_name):
    cv_types = ['spherical', 'tied', 'diag', 'full']
    n_components_range = range(1, clusters + 1)
    color_iter = itertools.cycle(['magenta', 'blue','red', 'darkorange'])
    bars = []

    # Plot the BIC scores
    plt.figure(figsize=(8, 6))
    for i, (cv_type, color) in enumerate(zip(cv_types, color_iter)):
        xpos = np.array(n_components_range) + .2 * (i - 2)
        bars.append(plt.bar(xpos, bic[i * len(n_components_range):
                                      (i + 1) * len(n_components_range)],
                            width=.2, color=color))
    plt.xticks(n_components_range)
    plt.ylim([bic.min() * 1.01 - .01 * bic.max(), bic.max()])
    plt.title('BIC score per covariance model, for %s' %(dataset_name))
    xpos = np.mod(bic.argmin(), len(n_components_range)) + .65 +\
        .2 * np.floor(bic.argmin() / len(n_components_range))
    plt.text(xpos, bic.min() * 0.97 + .03 * bic.max(), '*', fontsize=14)
    plt.xlabel('Number of components')
    plt.ylabel('BIC Score')
    plt.legend([b[0] for b in bars], cv_types)
    return plt

# # ### BIC Contra ###
c = 30  # looking up to 30 clusters
# bic_contra, best_gmm_contra = bic_analyze(X=X_train_contra, clusters=c)
# bic_plot_contra = plot_bic(bic=bic_contra, clusters=c, dataset_name='Contraceptive Methods')
em_cov_contra = 'diag'
em_clusters_contra = 28  # was 28

contra_gmm = GaussianMixture(n_components=em_clusters_contra, covariance_type=em_cov_contra,
                             random_state=17)
contra_y_gmm = contra_gmm.fit_predict(X_train_contra)
# contra_gmm_cluster_plot = plot_predictions(y_pred=contra_y_gmm, k=em_clusters_contra, dataset_name='Contraceptive Methods', estimator='EM')
contra_gmm_scores = evaluate_em(n_clust=em_clusters_contra, covariance=em_cov_contra, dataset_name="Contra", X_train=X_train_contra, y_train=y_train_contra, dr_algo_name="No DR")
print(contra_gmm_scores)

# ### BIC Arrhythmia ###
# bic_ekg, best_gmm_ekg = bic_analyze(X=X_train_ekg_not_resampled, clusters=c)
# bic_plot_ekg = plot_bic(bic=bic_ekg, clusters=c, dataset_name='Arrhythmia')
em_cov_ekg = 'diag'
em_clusters_ekg = 26
ekg_gmm = GaussianMixture(n_components=em_clusters_ekg, covariance_type=em_cov_ekg, random_state=17)
ekg_y_gmm = ekg_gmm.fit_predict(X_train_ekg)
# ekg_gmm_cluster_plot = plot_predictions(y_pred=ekg_y_gmm, k=em_clusters_ekg, dataset_name='Arrhythmia', estimator='EM')
ekg_gmm_scores = evaluate_em(n_clust=em_clusters_ekg, covariance=em_cov_ekg, dataset_name="Arrhythmia", X_train=X_train_ekg_not_resampled, y_train=y_train_ekg_not_resampled, dr_algo_name="No DR")
print(ekg_gmm_scores)

############################
############ PCA ###########
############################
from sklearn.decomposition import PCA

def pca_transform(n_comp, X_train):
    pca = PCA(n_components=n_comp, random_state=17)
    reduced_data = pca.fit_transform(X_train)
    print('Shape of the PCA Data df: ' + str(reduced_data.shape))
    return reduced_data, pca


# ### K-Means with PCA on Contra ###

# # Find optimal # of new components
# pca_contra_kmeans = PCA(n_components=len(contra_features))
# fitted_pca_contra = pca_contra_kmeans.fit_transform(X=X_train_contra)
# pca_contra_var_plot = plot_pca_variance(pca_contra_kmeans)
pca_high_var_components_contra = 4  # to get 70% of the variance

# With the optimized n_components, create the K-Means using PCA
df_reduced_contra_pca, contra_pca = pca_transform(n_comp=pca_high_var_components_contra, X_train=X_train_contra)
PCA_components_contra = pd.DataFrame(df_reduced_contra_pca)
# pca_components_plot_contra = plot_pca_components(PCA_components_contra, dataset='Contraceptives')
km_pca_contra = KMeans(n_clusters=k_contra_elbow, init='k-means++',n_init=10,
                       max_iter=300, tol=0.0001, random_state=17)
y_clusters_pca_kmeans_contra = km_pca_contra.fit_predict(df_reduced_contra_pca)

# # Plots and metrics for K-Means with PCA
# contra_predictions_dist_pca_kmeans = plot_predictions(y_pred=y_clusters_pca_kmeans_contra, k=k_contra_elbow, dataset_name='Contraceptives', estimator='K-Means with PCA')
# pca_contra_kmeans_plot = plot_clusters(X_clustered=y_clusters_pca_kmeans_contra, X_data=df_reduced_contra, dataset_name="Contraceptives", estimator_name="K-Means with PCA")
table_pred_kmeans_pca_contra = evaluate_k_means(n_clust=k_contra_elbow, dataset_name="Contraceptive Methods", X_train=df_reduced_contra_pca, y_train=y_train_contra, dr_algo_name="PCA")
print(table_pred_kmeans_pca_contra)

# ### K-Means with PCA on EKG ###

import math
# # Find optimal # of new components
optimal_n_ekg = int((4 * math.sqrt(len(ekg_features))) / math.sqrt(3)) # https://arxiv.org/abs/1305.5870
# pca_ekg_kmeans = PCA(n_components=int((optimal_n + 1)))
# fitted_pca_ekg = pca_ekg_kmeans.fit_transform(X=X_train_ekg_not_resampled)
# pca_ekg_var_plot = plot_pca_variance(pca_ekg_kmeans)
pca_high_var_components_ekg = 38  # to get X% of the variance

# With the optimized n_components, create the K-Means using PCA
df_reduced_ekg_pca, ekg_pca = pca_transform(n_comp=pca_high_var_components_ekg, X_train=X_train_ekg_not_resampled)
PCA_components_ekg = pd.DataFrame(df_reduced_ekg_pca)
# pca_components_plot_ekg = plot_pca_components(PCA_components_ekg, dataset="Arrhythmia")
km_pca_ekg = KMeans(n_clusters=k_ekg_elbow, init='k-means++', n_init=10, max_iter=300,
    tol=0.0001, random_state=17)
y_clusters_pca_kmeans_ekg = km_pca_ekg.fit_predict(df_reduced_ekg_pca)

# Plots and metrics for K-Means with PCA
# ekg_predictions_dist_pca_kmeans = plot_predictions(y_pred=y_clusters_pca_kmeans_ekg, k=k_ekg_elbow, dataset_name='Arrhythmia', estimator='K-Means with PCA')
# pca_ekg_kmeans_plot = plot_clusters(X_clustered=y_clusters_pca_kmeans_ekg, X_data=df_reduced_ekg, dataset_name="Arrhythmia", estimator_name="K-Means with PCA")
table_pred_kmeans_pca_ekg = evaluate_k_means(n_clust=k_ekg_elbow, dataset_name="Arrhythmia", X_train=df_reduced_ekg_pca, y_train=y_train_ekg_not_resampled, dr_algo_name="PCA")
print(table_pred_kmeans_pca_ekg)


# ### EM with PCA on Contra ###

# With the optimized n_components, create the EM using PCA
em_pca_contra = GaussianMixture(n_components=em_clusters_contra, covariance_type=em_cov_contra,
                                random_state=17)
y_clusters_pca_em_contra = em_pca_contra.fit_predict(df_reduced_contra_pca)

# # Plots and metrics for EM with PCA
# contra_predictions_dist_pca_em = plot_predictions(y_pred=y_clusters_pca_em_contra, k=em_clusters_contra, dataset_name='Contraceptives', estimator='EM with PCA')
# pca_contra_em_pca_plot = plot_clusters(X_clustered=y_clusters_pca_em_contra, X_data=df_reduced_contra_pca, dataset_name="Contraceptives", estimator_name="EM with PCA")
table_pred_em_pca_contra = evaluate_em(n_clust=k_contra_elbow, covariance=em_cov_contra, dataset_name="Contraceptive Methods", X_train=df_reduced_contra_pca, y_train=y_train_contra, dr_algo_name="PCA")
print(table_pred_em_pca_contra)

# ### EM with PCA on Arrhythmia ###

# With the optimized n_components, create the EM using PCA
em_pca_ekg = GaussianMixture(n_components=em_clusters_ekg, covariance_type=em_cov_ekg,
                                random_state=17)
y_clusters_pca_em_ekg = em_pca_ekg.fit_predict(df_reduced_ekg_pca)

# # Plots and metrics for EM with PCA
# ekg_predictions_dist_pca_em = plot_predictions(y_pred=y_clusters_pca_em_ekg, k=em_clusters_ekg, dataset_name='Arrhythmia', estimator='EM with PCA')
# pca_ekg_em_pca_plot = plot_clusters(X_clustered=y_clusters_pca_em_ekg, X_data=df_reduced_ekg_pca, dataset_name='Arrhythmia', estimator_name="EM with PCA")
table_pred_em_pca_ekg = evaluate_em(n_clust=k_ekg_elbow, covariance=em_cov_ekg, dataset_name='Arrhythmia', X_train=df_reduced_ekg_pca, y_train=y_train_ekg_not_resampled, dr_algo_name="PCA")
print(table_pred_em_pca_ekg)


############################
############ ICA ###########
############################
from sklearn.decomposition import FastICA

def ica_transform(n_comp, X_train):
    ica = FastICA(n_components=n_comp, random_state=17, whiten=True)
    reduced_data = ica.fit_transform(X_train)
    print('Shape of the ICA Data df: ' + str(reduced_data.shape))
    return reduced_data, ica

def ica_component_finder(n_features, X_train, X_test, y_train, y_test, dataset_name):
    # mse = []
    # for i in range(1, n+1):
    #     ica =     ica = FastICA(n_components=i, random_state=17, whiten=True)
    #     reduced_data = ica.fit_transform(X_train)
    #     X_restored = ica.inverse_transform(reduced_data)
    #     mse.append(((X_train - X_restored)**2).mean(axis=None))
    #
    # plt.figure()
    # plt.title("ICA component analysis")
    # components = range(1, n+1)
    # plt.bar(components, mse, color='black')
    # plt.xlabel('ICA Components')
    # plt.ylabel('MSE')
    # plt.xticks(components)

    n_components = np.linspace(1, n_features * 0.8, min(n_features - 2, 10)).astype('int')
    f1 = []
    for comp in n_components:
        ica = FastICA(n_components=comp, random_state=17, whiten=True)
        reduced_data = ica.fit_transform(X_train)

        # train a classifier on the reduced data
        model = MLPClassifier(solver='lbfgs', random_state=17)
        model.fit(reduced_data, y_train)

        # evaluate the model and update the list of accuracies
        test = ica.transform(X_test)
        f1.append(f1_score(model.predict(test), y_test, average="weighted"))

    # Create a baseline
    model2 = MLPClassifier(solver='lbfgs', random_state=17)
    model2.fit(X_train, y_train)
    baseline = f1_score(model2.predict(X_test), y_test, average="weighted")

    # create the figure
    plt.figure()
    plt.suptitle("F1 Score of ICA on %s" %(dataset_name))
    plt.xlabel("Number of Components")
    plt.ylabel("Model F1 Score")

    # plot the baseline and random projection accuracies
    plt.plot(n_components, [baseline] * len(f1), color="r")
    plt.plot(n_components, f1)
    return plt

# K-Means with ICA on Contra

# Find optimal # of new components
# ica_features_contra_plot = ica_component_finder(n_features=len(contra_features), X_train=X_train_contra, X_test=X_test_contra, y_train=y_train_contra, y_test=y_test_contra, dataset_name="Contraceptives")
ica_features_contra = 5  # highest F1 score

# With the optimized n_components, create the K-Means using ICA
ica_contra_kmeans = FastICA(n_components=ica_features_contra, random_state=17, whiten=True)
df_reduced_contra_ica, ica_contra = ica_transform(n_comp=ica_features_contra, X_train=X_train_contra)
ICA_components_contra = pd.DataFrame(df_reduced_contra_ica)
km_ica_contra = KMeans(n_clusters=k_contra_elbow, init='k-means++',n_init=10,
                       max_iter=300, tol=0.0001, random_state=17)
y_clusters_ica_kmeans_contra = km_ica_contra.fit_predict(df_reduced_contra_ica)

# Plots and metrics for K-Means with ICA
# contra_predictions_dist_ica_kmeans = plot_predictions(y_pred=y_clusters_ica_kmeans_contra, k=k_contra_elbow, dataset_name='Contraceptives', estimator='K-Means with ICA')
# ica_contra_kmeans_plot = plot_clusters(X_clustered=y_clusters_ica_kmeans_contra, X_data=df_reduced_contra_ica, dataset_name="Contraceptives", estimator_name="K-Means with ICA")
table_pred_kmeans_ica_contra = evaluate_k_means(n_clust=k_contra_elbow, dataset_name="Contraceptive Methods", X_train=df_reduced_contra_ica, y_train=y_train_contra, dr_algo_name="ICA")
print(table_pred_kmeans_ica_contra)


# K-Means with ICA on EKG

# Find optimal # of new components
# ica_features_ekg_plot = ica_component_finder(n_features=2 * optimal_n_ekg, X_train=X_train_ekg_not_resampled, X_test=X_test_ekg, y_train=y_train_ekg_not_resampled, y_test=y_test_ekg, dataset_name="Arrhythmia")
ica_features_ekg = 14   # highest F1 score

# With the optimized n_components, create the K-Means using ICA
ica_ekg_kmeans = FastICA(n_components=k_ekg_elbow, whiten=True)
df_reduced_ekg_ica, ica_ekg = ica_transform(n_comp=k_ekg_elbow, X_train=X_train_ekg_not_resampled)
ICA_components_ekg = pd.DataFrame(df_reduced_ekg_ica)
km_ica_ekg = KMeans(n_clusters=k_ekg_elbow, init='k-means++', n_init=10, max_iter=300,
    tol=0.0001, random_state=17)
y_clusters_ica_kmeans_ekg = km_ica_ekg.fit_predict(df_reduced_ekg_ica)

# Plots and metrics for K-Means with ICA
# ekg_predictions_dist_ica_kmeans = plot_predictions(y_pred=y_clusters_ica_kmeans_ekg, k=k_ekg_elbow, dataset_name='Arrhythmia', estimator='K-Means with ICA')
# ica_ekg_kmeans_plot = plot_clusters(X_clustered=y_clusters_ica_kmeans_ekg, X_data=df_reduced_ekg_ica, dataset_name="Arrhythmia", estimator_name="K-Means with ICA")
table_pred_kmeans_ica_ekg = evaluate_k_means(n_clust=k_ekg_elbow, dataset_name="Arrhythmia", X_train=df_reduced_ekg_ica, y_train=y_train_ekg_not_resampled, dr_algo_name="ICA")
print(table_pred_kmeans_ica_ekg)


# EM with ICA on Contra
# With the optimized n_components, create the EM using ICA
em_ica_contra = GaussianMixture(n_components=em_clusters_contra, covariance_type=em_cov_contra, random_state=17)
y_clusters_ica_em_contra = em_ica_contra.fit_predict(df_reduced_contra_ica)

# Plots and metrics for EM with ICA
# contra_predictions_dist_ica_em = plot_predictions(y_pred=y_clusters_ica_em_contra, k=em_clusters_contra, dataset_name='Contraceptives', estimator='EM with ICA')
# ica_contra_em_plot = plot_clusters(X_clustered=y_clusters_ica_em_contra, X_data=df_reduced_contra_ica, dataset_name="Contraceptives", estimator_name="EM with ICA")
table_pred_em_ica_contra = evaluate_em(n_clust=em_clusters_contra, covariance=em_cov_contra, dataset_name="Contraceptive Methods", X_train=df_reduced_contra_ica, y_train=y_train_contra, dr_algo_name="ICA")
print(table_pred_em_ica_contra)

# EM with ICA on Arrhythmia
# With the optimized n_components, create the EM using ICA
em_ica_ekg = GaussianMixture(n_components=em_clusters_ekg, covariance_type=em_cov_ekg, random_state=17)
y_clusters_ica_em_ekg= em_ica_ekg.fit_predict(df_reduced_ekg_ica)

# Plots and metrics for EM with ICA
# ekg_predictions_dist_ica_em = plot_predictions(y_pred=y_clusters_ica_em_ekg, k=em_clusters_ekg, dataset_name='Arrhythmia', estimator='EM with ICA')
# ica_ekg_em_plot = plot_clusters(X_clustered=y_clusters_ica_em_ekg, X_data=df_reduced_ekg_ica, dataset_name="Arrhythmia", estimator_name="EM with ICA")
table_pred_em_ica_ekg = evaluate_em(n_clust=em_clusters_ekg, covariance=em_cov_ekg, dataset_name="Arrhythmia", X_train=df_reduced_ekg_ica, y_train=y_train_ekg_not_resampled, dr_algo_name="ICA")
print(table_pred_em_ica_ekg)


###########################################
############ Random Projections ###########
###########################################
from sklearn.random_projection import SparseRandomProjection

def rp_transform(n_comp, X_train):
    rp = SparseRandomProjection(n_components=n_comp, random_state=17)
    reduced_data = rp.fit_transform(X_train)
    print('Shape of the Sparse Random Projection Data df: ' + str(reduced_data.shape))
    return reduced_data, rp

def rp_component_finder(X_train, X_test, y_train, y_test, dataset_name, n_features):
    n_components = np.linspace(1, n_features * 0.8, min(n_features - 2, 10)).astype('int')
    f1 = []
    for comp in n_components:
        rp = SparseRandomProjection(n_components=comp, random_state=17)
        reduced_data = rp.fit_transform(X_train)

        # train a classifier on the reduced data
        model = MLPClassifier(solver='lbfgs', random_state=17)
        model.fit(reduced_data, y_train)

        # evaluate the model and update the list of accuracies
        test = rp.transform(X_test)
        f1.append(f1_score(model.predict(test), y_test, average="weighted"))

    # Create a baseline
    model2 = MLPClassifier(solver='lbfgs', random_state=17)
    model2.fit(X_train, y_train)
    baseline = f1_score(model2.predict(X_test), y_test, average="weighted")

    # create the figure
    plt.figure()
    plt.suptitle("F1 Score of Sparse Random Projection on %s" %(dataset_name))
    plt.xlabel("Number of Components")
    plt.ylabel("Model F1 Score")

    # plot the baseline and random projection accuracies
    plt.plot(n_components, [baseline] * len(f1), color="r")
    plt.plot(n_components, f1)
    return plt


# K-Means with RP on Contra
# Find optimal # of new components
# rp_contra_plot = rp_component_finder(X_train=X_train_contra, X_test=X_test_contra, y_train=y_train_contra, y_test=y_test_contra ,dataset_name="Contraceptives", n_features=contra_features.size)
optimal_rp_components_contra = 5  # highest F1 Score

# With the optimized n_components, create the K-Means K-Means with Sparse RP
rp_contra_kmeans = SparseRandomProjection(n_components=optimal_rp_components_contra, random_state=17)
df_reduced_contra_rp, rp_contra = rp_transform(n_comp=optimal_rp_components_contra, X_train=X_train_contra)
rp_components_contra = pd.DataFrame(df_reduced_contra_rp)
km_rp_contra = KMeans(n_clusters=k_contra_elbow, init='k-means++',n_init=10,
                       max_iter=300, tol=0.0001, random_state=17)
y_clusters_rp_kmeans_contra = km_rp_contra.fit_predict(df_reduced_contra_rp)

# Plots and metrics for K-Means with Sparse RP
# contra_predictions_dist_rp_kmeans = plot_predictions(y_pred=y_clusters_rp_kmeans_contra, k=k_contra_elbow, dataset_name='Contraceptives', estimator='K-Means with Sparse Random Projections')
# rp_contra_kmeans_plot = plot_clusters(X_clustered=y_clusters_rp_kmeans_contra, X_data=df_reduced_contra_rp, dataset_name="Contraceptives", estimator_name="K-Means with Sparse Random Projections")
table_pred_kmeans_rp_contra = evaluate_k_means(n_clust=k_contra_elbow, dataset_name="Contraceptive Methods", X_train=df_reduced_contra_rp, y_train=y_train_contra, dr_algo_name="Sparse RP")
print(table_pred_kmeans_rp_contra)

# K-Means with RP on Arrhythmia
# Find optimal # of new components
# rp_ekg_plot = rp_component_finder(X_train=X_train_ekg_not_resampled, X_test=X_test_ekg, y_train=y_train_ekg_not_resampled, y_test=y_test_ekg ,dataset_name="Arrhythmia", n_features=ekg_features.size)
optimal_rp_components_ekg = 150  # highest F1 Score

# With the optimized n_components, create the K-Means K-Means with Sparse RP
rp_ekg_kmeans = SparseRandomProjection(n_components=optimal_rp_components_ekg, random_state=17)
df_reduced_ekg_rp, rp_ekg = rp_transform(n_comp=optimal_rp_components_ekg, X_train=X_train_ekg_not_resampled)
rp_components_ekg = pd.DataFrame(df_reduced_ekg_rp)
km_rp_ekg = KMeans(n_clusters=k_ekg_elbow, init='k-means++',n_init=10,
                       max_iter=300, tol=0.0001, random_state=17)
y_clusters_rp_kmeans_ekg = km_rp_ekg.fit_predict(df_reduced_ekg_rp)

# Plots and metrics for K-Means with Sparse RP
# ekg_predictions_dist_rp_kmeans = plot_predictions(y_pred=y_clusters_rp_kmeans_ekg, k=k_ekg_elbow, dataset_name="Arrhythmia", estimator='K-Means with Sparse Random Projections')
# rp_ekg_kmeans_plot = plot_clusters(X_clustered=y_clusters_rp_kmeans_ekg, X_data=df_reduced_ekg_rp, dataset_name="Arrhythmia", estimator_name="K-Means with Sparse Random Projections")
table_pred_kmeans_rp_ekg = evaluate_k_means(n_clust=k_ekg_elbow, dataset_name="Arrhythmia", X_train=df_reduced_ekg_rp, y_train=y_train_ekg_not_resampled, dr_algo_name="Sparse RP")
print(table_pred_kmeans_rp_ekg)



# EM with RP on Contra
# With the optimized n_components, create the EM using RP
em_rp_contra = GaussianMixture(n_components=em_clusters_contra, covariance_type=em_cov_contra, random_state=17)
y_clusters_rp_em_contra = em_rp_contra.fit_predict(df_reduced_contra_rp)

# Plots and metrics for EM with RP
# contra_predictions_dist_rp_em = plot_predictions(y_pred=y_clusters_rp_em_contra, k=em_clusters_contra, dataset_name='Contraceptives', estimator='EM with Sparse RP')
# rp_contra_em_plot = plot_clusters(X_clustered=y_clusters_rp_em_contra, X_data=df_reduced_contra_rp, dataset_name="Contraceptives", estimator_name="EM with Sparse RP")
table_pred_em_rp_contra = evaluate_em(n_clust=em_clusters_contra, covariance=em_cov_contra, dataset_name="Contraceptive Methods", X_train=df_reduced_contra_rp, y_train=y_train_contra, dr_algo_name="Sparse RP")
print(table_pred_em_rp_contra)

# EM with RP on Arrhythmia
# With the optimized n_components, create the EM using RP
em_rp_ekg = GaussianMixture(n_components=em_clusters_ekg, covariance_type=em_cov_ekg, random_state=17)
y_clusters_rp_em_ekg= em_rp_ekg.fit_predict(df_reduced_ekg_rp)

# Plots and metrics for EM with RP
# ekg_predictions_dist_rp_em = plot_predictions(y_pred=y_clusters_rp_em_ekg, k=em_clusters_ekg, dataset_name='Arrhythmia', estimator='EM with Sparse RP')
# rp_ekg_em_plot = plot_clusters(X_clustered=y_clusters_rp_em_ekg, X_data=df_reduced_ekg_rp, dataset_name="Arrhythmia", estimator_name="EM with Sparse RP")
table_pred_em_rp_ekg = evaluate_em(n_clust=em_clusters_ekg, covariance=em_cov_ekg, dataset_name="Arrhythmia", X_train=df_reduced_ekg_rp, y_train=y_train_ekg_not_resampled, dr_algo_name="Sparse RP")
print(table_pred_em_rp_ekg)


###############################
############ ISOMAP ###########
###############################
from sklearn.manifold import Isomap

def isomap_transform(n_comp, n_neighbors, X_train):
    isomap =Isomap(n_components=n_comp, n_neighbors=n_neighbors)
    reduced_data = isomap.fit_transform(X_train)
    print('Shape of the Isomap Data df: ' + str(reduced_data.shape))
    return reduced_data, isomap

def isomap_component_finder(X_train, X_test, y_train, y_test, dataset_name, n_features):
    warnings.filterwarnings('ignore', 'F-score is ill-defined.*')
    n_components = np.linspace(3, max((n_features / 2), 8), min(n_features - 2, 10)).astype('int')
    n_neighbors = np.linspace(2, len(X_train)**(1/2), 9).astype('int')
    last_neighbor = n_neighbors[len(n_neighbors) - 1]
    f1 = {}
    for comp in n_components:
        acc = []
        for neighbors in n_neighbors:
            isomap = Isomap(n_components=comp, n_neighbors=neighbors)
            reduced_data = isomap.fit_transform(X_train)

            # train a classifier with the reduced data
            model = MLPClassifier(solver='lbfgs', random_state=17)
            model.fit(reduced_data, y_train)

            # evaluate the model and update the list of accuracies
            test = isomap.transform(X_test)
            acc.append(f1_score(model.predict(test), y_test, average="weighted"))
            if neighbors == last_neighbor:
                f1[comp] = acc

    # Create a baseline
    model2 = MLPClassifier(solver='lbfgs', random_state=17)
    model2.fit(X_train, y_train)
    baseline = f1_score(model2.predict(X_test), y_test, average="weighted")

    # create the figure
    plt.figure()
    plt.suptitle("F1 Score of Isomap on %s" % (dataset_name))
    plt.xlabel("Number of Neighbors")
    plt.ylabel("Model F1 Score")

    # plot the baseline and F1 Scores
    plt.plot(n_neighbors, [baseline] * len(n_neighbors), color="r")
    for comp, f1_values in f1.items():
        plt.plot(n_neighbors, f1_values, '-o', label='n_components' + ': ' + str(comp))
    plt.legend()

    return plt


# K-Means with Isomap on Contra
# Find optimal # of components and neighbors
# isomap_contra_plot = isomap_component_finder(X_train=X_train_contra, X_test=X_test_contra, y_train=y_train_contra, y_test=y_test_contra ,dataset_name="Contraceptives", n_features=contra_features.size)
isomap_components_contra = 5  # highest F1 Score
isomap_neighbors_contra = 5  # highest F1 Score

# With the optimized n_components, create the K-Means with Isomap
isomap_contra_kmeans = Isomap(n_components=isomap_components_contra, n_neighbors=isomap_neighbors_contra)
df_reduced_contra_isomap, isomap_contra = isomap_transform(n_comp=isomap_components_contra, X_train=X_train_contra, n_neighbors=isomap_neighbors_contra)
isomap_components_contra = pd.DataFrame(df_reduced_contra_isomap)
km_isomap_contra = KMeans(n_clusters=k_contra_elbow, init='k-means++',n_init=10,
                       max_iter=300, tol=0.0001, random_state=17)
y_clusters_isomap_kmeans_contra = km_isomap_contra.fit_predict(df_reduced_contra_isomap)

# Plots and metrics for K-Means with Isomap
# contra_predictions_dist_isomap_kmeans = plot_predictions(y_pred=y_clusters_isomap_kmeans_contra, k=k_contra_elbow, dataset_name='Contraceptives', estimator='K-Means with Isomap Projections')
# isomap_contra_kmeans_plot = plot_clusters(X_clustered=y_clusters_isomap_kmeans_contra, X_data=df_reduced_contra_isomap, dataset_name="Contraceptives", estimator_name="K-Means with Isomap Projections")
table_pred_kmeans_isomap_contra = evaluate_k_means(n_clust=k_contra_elbow, dataset_name="Contraceptive Methods", X_train=df_reduced_contra_isomap, y_train=y_train_contra, dr_algo_name="ISOMAP")
print(table_pred_kmeans_isomap_contra)

# K-Means with Isomap on Arrhythmia data
# Find optimal # of components and neighbors
# isomap_ekg_plot = isomap_component_finder(X_train=X_train_ekg_not_resampled, X_test=X_test_ekg, y_train=y_train_ekg_not_resampled, y_test=y_test_ekg ,dataset_name="Arrhythmia", n_features=ekg_features.size)
isomap_components_ekg = 139  # highest F1 Score
isomap_neighbors_ekg = 10  # highest F1 Score

# With the optimized n_components, create the K-Means K-Means with Isomap
isomap_ekg_kmeans = Isomap(n_components=isomap_components_ekg, n_neighbors=isomap_neighbors_ekg)
df_reduced_ekg_isomap, isomap_ekg = isomap_transform(n_comp=isomap_components_ekg, X_train=X_train_ekg_not_resampled, n_neighbors=isomap_neighbors_ekg)
ekg_components_ekg = pd.DataFrame(df_reduced_ekg_isomap)
km_isomap_ekg = KMeans(n_clusters=k_ekg_elbow, init='k-means++',n_init=10,
                       max_iter=300, tol=0.0001, random_state=17)
y_clusters_isomap_kmeans_ekg = km_isomap_ekg.fit_predict(df_reduced_ekg_isomap)

# Plots and metrics for K-Means with Isomap
# ekg_predictions_dist_isomap_kmeans = plot_predictions(y_pred=y_clusters_isomap_kmeans_ekg, k=k_ekg_elbow, dataset_name='Arrhythmia', estimator='K-Means with Isomap Projections')
# isomap_ekg_kmeans_plot = plot_clusters(X_clustered=y_clusters_isomap_kmeans_ekg, X_data=df_reduced_ekg_isomap, dataset_name="Arrhythmia", estimator_name="K-Means with Isomap Projections")
table_pred_kmeans_isomap_ekg = evaluate_k_means(n_clust=k_ekg_elbow, dataset_name="Arrhythmia", X_train=df_reduced_ekg_isomap, y_train=y_train_ekg_not_resampled, dr_algo_name="ISOMAP")
print(table_pred_kmeans_isomap_ekg)


# EM with ISOMAP on Contra
# With the optimized n_components, create the EM using ISOMAP
em_isomap_contra = GaussianMixture(n_components=em_clusters_contra, covariance_type=em_cov_contra, random_state=17)
y_clusters_isomap_em_contra = em_isomap_contra.fit_predict(df_reduced_contra_isomap)

# Plots and metrics for EM with ISOMAP
# contra_predictions_dist_isomap_em = plot_predictions(y_pred=y_clusters_isomap_em_contra, k=em_clusters_contra, dataset_name='Contraceptives', estimator='EM with ISOMAPP')
# isomap_contra_em_plot = plot_clusters(X_clustered=y_clusters_isomap_em_contra, X_data=df_reduced_contra_isomap, dataset_name="Contraceptives", estimator_name="EM with ISOMAP")
table_pred_em_isomap_contra = evaluate_em(n_clust=em_clusters_contra, covariance=em_cov_contra, dataset_name="Contraceptive Methods", X_train=df_reduced_contra_isomap, y_train=y_train_contra, dr_algo_name="ISOMAP")
print(table_pred_em_isomap_contra)

# EM with ISOMAP on Arrhythmia
# With the optimized n_components, create the EM using RP
em_isomap_ekg = GaussianMixture(n_components=em_clusters_ekg, covariance_type=em_cov_ekg, random_state=17)
y_clusters_isomap_em_ekg= em_isomap_ekg.fit_predict(df_reduced_ekg_isomap)

# Plots and metrics for EM with RP
# ekg_predictions_dist_isomap_em = plot_predictions(y_pred=y_clusters_isomap_em_ekg, k=em_clusters_ekg, dataset_name='Arrhythmia', estimator='EM with ISOMAP')
# isomap_ekg_em_plot = plot_clusters(X_clustered=y_clusters_isomap_em_ekg, X_data=df_reduced_ekg_isomap, dataset_name="Arrhythmia", estimator_name="EM with ISOMAP")
table_pred_em_isomap_ekg = evaluate_em(n_clust=em_clusters_ekg, covariance=em_cov_ekg, dataset_name="Arrhythmia", X_train=df_reduced_ekg_isomap, y_train=y_train_ekg_not_resampled, dr_algo_name="ISOMAP")
print(table_pred_em_isomap_ekg)


################################################
#### Feature Importance using Random Forest ####
################################################
from sklearn.ensemble import RandomForestClassifier
def find_important_features(X_train, y_train, dataset_features, dataset_name, n_features):
    plt.figure()
    model = RandomForestClassifier(random_state=17, max_depth=10)
    model.fit(X=X_train, y=y_train)
    importances = model.feature_importances_
    indices = np.argsort(importances)[-(n_features-1):]  # top important features
    plt.title('Feature Importances for %s' %(dataset_name))
    plt.barh(range(len(indices)), importances[indices], color='b', align='center')
    plt.yticks(range(len(indices)), [dataset_features[i] for i in indices], rotation=45)
    plt.xlabel('Relative Importance')
    return plt

# contra_feature_importance_plot = find_important_features(X_train=X_train_contra, y_train=y_train_contra, dataset_features=contra_features, dataset_name="Contraceptives", n_features=5)
# ekg_feature_importance_plot = find_important_features(X_train=X_train_ekg, y_train=y_train_ekg, dataset_features=ekg_features, dataset_name="Arrhythmia", n_features=35)


################################################
#### Neural Network - PCA DR only ####
################################################
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
                         color="black" if cm[i, j] <= thresh else "white", fontsize=14)
            elif i == max(range(cm.shape[0])):
                plt.text(j, i - 0.25, "{0:0.1%}".format(cm[i, j]),
                         horizontalalignment="center",
                         color="black" if cm[i, j] <= thresh else "white", fontsize=14)
            else:
                plt.text(j, i, "{0:0.1%}".format(cm[i, j]),
                         horizontalalignment="center",
                         color="black" if cm[i, j] <= thresh else "white", fontsize=14)

        else:
            plt.text(j, i + 0.25, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="black" if cm[i, j] <= thresh else "white", fontsize=14)

    plt.tight_layout()
    plt.ylabel('True label', fontsize=14)
    plt.xlabel('Predicted label\naccuracy={:0.3f}; misclass={:0.3f}'.format(accuracy, misclass), fontsize=14)
    return plt

def evaluate_classifier(classifier, learner, dataset, X_train, X_test, y_train, y_test, class_names, feature_names):
    warnings.filterwarnings('ignore', 'F-score is ill-defined.*')
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
    distribution_plot = plt.figure()
    unique, counts = np.unique(y_pred, return_counts=True)
    res_classes = dict(zip(unique, counts))
    plt.bar(*zip(*res_classes.items()))
    plt.xticks((1, 2, 3), labels=class_names)
    plt.title("Distribution of Predictions, over all Classes")

    # Standard Metrics
    f1 = f1_score(y_test, y_pred, average='weighted', labels=np.unique(y_pred))  # Use average = 'micro', 'macro' or 'weighted' since we have non-binary classes. Don't count classes that are never predicted.
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', labels=np.unique(y_pred))
    recall = recall_score(y_test, y_pred, average='weighted', labels=np.unique(y_pred))
    confusion_matrix_plot = plot_confusion_matrix(y_test, y_pred, learner=learner, dataset=dataset, target_names=class_names)

    print("Metrics for the %s on the %s Dataset" %(learner, dataset))
    print("Model Training Time (ms):   " + "{:.1f}".format(training_time))
    print("Model Prediction Time (ms): " + "{:.1f}\n".format(pred_time))
    print("F1 Score:  " + "{:.2f}".format(f1))
    print("Check the confusion plot!")
    print("Accuracy:  " + "{:.2f}".format(accuracy))
    print("Precision: " + "{:.2f}".format(precision))
    print("Recall:    " + "{:.2f}".format(recall))
    return "{0:.2f}".format(f1), "{0:.2f}".format(accuracy), "{0:.2f}".format(precision), "{0:.2f}".format(recall), "{:.1f}".format(training_time), "{:.1f}".format(pred_time), distribution_plot, confusion_matrix_plot

def ohe_ndarray(ndarray, n_clusters):
    num_values = n_clusters
    ohe = np.eye(num_values)[ndarray.reshape(-1)]
    return ohe

# Arrhythmia neural networks
ekg_nn_activation = 'relu'
ekg_nn_hls = [150,]
ekg_nn_estimator = MLPClassifier(activation=ekg_nn_activation, max_iter=400, solver='lbfgs', hidden_layer_sizes=ekg_nn_hls, random_state=17)

# Baseline NN (supervised learning)
ekg_nn_test_f1, ekg_nn_test_acc, ekg_nn_test_precision, ekg_nn_test_recall, ekg_nn_train_time, ekg_nn_test_time, ekg_nn_distribution_plot_baseline, confusion_matrix_plot_baseline = evaluate_classifier(classifier=ekg_nn_estimator, X_train=X_train_ekg, X_test=X_test_ekg, y_train=y_train_ekg, y_test=y_test_ekg, learner='Supervised Learning Neural Network', dataset="Arrhythmia (EKG)", class_names=ekg_labels, feature_names=ekg_features)
ekg_nn_results = ['ekg_nn_results', ekg_nn_test_f1, ekg_nn_test_acc, ekg_nn_test_precision, ekg_nn_test_recall, ekg_nn_train_time, ekg_nn_test_time]


# ... with PCA DR but without clustering: x-validation and final classifier evaluation
X_train_reduced_ekg_pca, ekg_pca_nn = pca_transform(n_comp=pca_high_var_components_ekg, X_train=X_train_ekg)
X_test_ekg_pca_transformed = ekg_pca_nn.transform(X_test_ekg)
ekg_nn_test_f1_pca, ekg_nn_test_acc_pca, ekg_nn_test_precision_pca, ekg_nn_test_recall_pca, ekg_nn_train_time_pca, ekg_nn_test_time_pca, ekg_nn_distribution_plot_pca, confusion_matrix_plot_pca = evaluate_classifier(classifier=ekg_nn_estimator, X_train= X_train_reduced_ekg_pca, X_test=X_test_ekg_pca_transformed, y_train=y_train_ekg, y_test=y_test_ekg, learner='Neural Network with PCA DR', dataset="Arrhythmia (EKG)", class_names=ekg_labels, feature_names=ekg_features)
ekg_nn_results_pca = ['ekg_nn_results_pca', ekg_nn_test_f1_pca, ekg_nn_test_acc_pca, ekg_nn_test_precision_pca, ekg_nn_test_recall_pca, ekg_nn_train_time_pca, ekg_nn_test_time_pca]


# ... with ICA DR but without clustering: x-validation and final classifier evaluation
X_train_reduced_ekg_ica, ekg_ica_nn = ica_transform(n_comp=ica_features_ekg, X_train=X_train_ekg)
X_test_ekg_ica_transformed = ekg_ica_nn.transform(X_test_ekg)
ekg_nn_test_f1_ica, ekg_nn_test_acc_ica, ekg_nn_test_precision_ica, ekg_nn_test_recall_ica, ekg_nn_train_time_ica, ekg_nn_test_time_ica, ekg_nn_distribution_plot_ica, confusion_matrix_plot_ica = evaluate_classifier(classifier=ekg_nn_estimator, X_train= X_train_reduced_ekg_ica, X_test=X_test_ekg_ica_transformed, y_train=y_train_ekg, y_test=y_test_ekg, learner='Neural Network with ICA DR', dataset="Arrhythmia (EKG)", class_names=ekg_labels, feature_names=ekg_features)
ekg_nn_results_ica = ['ekg_nn_results_ica', ekg_nn_test_f1_ica, ekg_nn_test_acc_ica, ekg_nn_test_precision_ica, ekg_nn_test_recall_ica, ekg_nn_train_time_ica, ekg_nn_test_time_ica]


# ... with RP DR but without clustering: x-validation and final classifier evaluation
X_train_reduced_ekg_rp, ekg_rp_nn = rp_transform(n_comp=optimal_rp_components_ekg, X_train=X_train_ekg)
X_test_ekg_rp_transformed = ekg_rp_nn.transform(X_test_ekg)
ekg_nn_test_f1_rp, ekg_nn_test_acc_rp, ekg_nn_test_precision_rp, ekg_nn_test_recall_rp, ekg_nn_train_time_rp, ekg_nn_test_time_rp, ekg_nn_distribution_plot_rp, confusion_matrix_plot_rp = evaluate_classifier(classifier=ekg_nn_estimator, X_train= X_train_reduced_ekg_rp, X_test=X_test_ekg_rp_transformed, y_train=y_train_ekg, y_test=y_test_ekg, learner='Neural Network with Sparse RP DR', dataset="Arrhythmia (EKG)", class_names=ekg_labels, feature_names=ekg_features)
ekg_nn_results_rp = ['ekg_nn_results_rp', ekg_nn_test_f1_rp, ekg_nn_test_acc_rp, ekg_nn_test_precision_rp, ekg_nn_test_recall_rp, ekg_nn_train_time_rp, ekg_nn_test_time_rp]


# ... with ISOMAP DR but without clustering: x-validation and final classifier evaluation
X_train_reduced_ekg_isomap, ekg_isomap_nn = isomap_transform(n_comp=isomap_components_ekg, n_neighbors=isomap_neighbors_ekg, X_train=X_train_ekg)
X_test_ekg_isomap_transformed = ekg_isomap_nn.transform(X_test_ekg)
ekg_nn_test_f1_isomap, ekg_nn_test_acc_isomap, ekg_nn_test_precision_isomap, ekg_nn_test_recall_isomap, ekg_nn_train_time_isomap, ekg_nn_test_time_isomap, ekg_nn_distribution_plot_isomap, confusion_matrix_plot_isomap = evaluate_classifier(classifier=ekg_nn_estimator, X_train= X_train_reduced_ekg_isomap, X_test=X_test_ekg_isomap_transformed, y_train=y_train_ekg, y_test=y_test_ekg, learner='Neural Network with ISOMAP DR', dataset="Arrhythmia (EKG)", class_names=ekg_labels, feature_names=ekg_features)
ekg_nn_results_isomap = ['ekg_nn_results_isomap', ekg_nn_test_f1_isomap, ekg_nn_test_acc_isomap, ekg_nn_test_precision_isomap, ekg_nn_test_recall_isomap, ekg_nn_train_time_isomap, ekg_nn_test_time_isomap]


# ... with only EM clustering: x-validation and final classifier evaluation
ekg_train_em_clusters = ekg_gmm.fit_predict(X_train_ekg)
ekg_test_em_clusters = ekg_gmm.predict(X_test_ekg)

# one hot encode the new X values (clusters) since they are ordinal
ekg_train_em_clusters_ohe = ohe_ndarray(ekg_train_em_clusters, n_clusters=em_clusters_ekg)
ekg_test_em_clusters_ohe = ohe_ndarray(ekg_test_em_clusters, n_clusters=em_clusters_ekg)

ekg_nn_test_f1_em, ekg_nn_test_acc_em, ekg_nn_test_precision_em, ekg_nn_test_recall_em, ekg_nn_train_time_em, ekg_nn_test_time_em, ekg_nn_distribution_plot_em, confusion_matrix_plot_em = evaluate_classifier(classifier=ekg_nn_estimator, X_train= ekg_train_em_clusters_ohe, X_test=ekg_test_em_clusters_ohe, y_train=y_train_ekg, y_test=y_test_ekg, learner='Neural Network with EM Clustering', dataset="Arrhythmia (EKG)", class_names=ekg_labels, feature_names=ekg_features)
ekg_nn_results_em = ['ekg_nn_results_em', ekg_nn_test_f1_em, ekg_nn_test_acc_em, ekg_nn_test_precision_em, ekg_nn_test_recall_em, ekg_nn_train_time_em, ekg_nn_test_time_em]


# ... with only KMeans clustering: x-validation and final classifier evaluation
ekg_train_kmeans_clusters = ekg_km.fit_predict(X_train_ekg)
ekg_test_kmeans_clusters = ekg_km.predict(X_test_ekg)

# one hot encode the new X values (clusters) since they are ordinal
ekg_train_kmeans_clusters_ohe = ohe_ndarray(ekg_train_kmeans_clusters, n_clusters=k_ekg_elbow)
ekg_test_kmeans_clusters_ohe = ohe_ndarray(ekg_test_kmeans_clusters, n_clusters=k_ekg_elbow)

ekg_nn_test_f1_kmeans, ekg_nn_test_acc_kmeans, ekg_nn_test_precision_kmeans, ekg_nn_test_recall_kmeans, ekg_nn_train_time_kmeans, ekg_nn_test_time_kmeans, ekg_nn_distribution_plot_kmeans, confusion_matrix_plot_kmeans = evaluate_classifier(classifier=ekg_nn_estimator, X_train= ekg_train_kmeans_clusters_ohe, X_test=ekg_test_kmeans_clusters_ohe, y_train=y_train_ekg, y_test=y_test_ekg, learner='Neural Network with K-Means', dataset="Arrhythmia (EKG)", class_names=ekg_labels, feature_names=ekg_features)
ekg_nn_results_kmeans = ['ekg_nn_results_kmeans', ekg_nn_test_f1_kmeans, ekg_nn_test_acc_kmeans, ekg_nn_test_precision_kmeans, ekg_nn_test_recall_kmeans, ekg_nn_train_time_kmeans, ekg_nn_test_time_kmeans]

#========= Tabulate the final data ========#
from tabulate import tabulate
print(tabulate([ekg_nn_results, ekg_nn_results_pca, ekg_nn_results_ica, ekg_nn_results_rp, ekg_nn_results_isomap, ekg_nn_results_em, ekg_nn_results_kmeans], headers=['Learner', 'F1_Weighted', 'Accuracy', 'Precision', 'Recall', 'Training time (ms)', 'Prediction time (ms)']))

plt.show()
print("You got here!")

