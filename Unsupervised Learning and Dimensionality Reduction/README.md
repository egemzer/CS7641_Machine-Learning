## CS 7641_Machine_Learning 
### Assignment 3: Unsupervised Learning and Dimensionality Reduction

### Code Location
https://github.com/egemzer/CS7641_Machine_Learning/tree/master/Unsupervised%20Learning%20and%20Dimensionality%20Reduction

### Overview

This project was completed by @egemzer (gth659q) for master's degree course in Machine Learning at Georgia Tech. 

#### Part 1 
To better understand unsupervised learning and the importance of dimensionality reduction, we implement the following size algorithms and compare their performance across two datasets:
- K-Means Clustering
- Expectation Maximization (EM) Clustering
- PCA + Clustering (K-Means + EM)
- ICA + Clustering (K-Means + EM)
- Randomized Projections (RP) + Clustering (K-Means + EM)
- ISOMAP  + Clustering (K-Means + EM)

#### Part 2
To compare unsupervised learning to supervised learning, I created a Neural Network using datasets whose dimensions were reduced using:
- K-Means Clustering (clusters -> features)
- EM (clusters -> features)
- PCA
- ICA
- Sparse RP
- ISOMAP

### Datasets
Dataset 1: **Contraceptives Dataset** - in repo and available at https://www.openml.org/d/23
    
Dataset 2: **Arrhythmia** - in repo and available at https://www.openml.org/d/5 

### Libraries
python 3.7.x, sklearn, matplotlib, pandas, numpy, timeit, datetime, copy, imblearn, itertools, math, warnings, 

### How to Run the Code
To test this code on a local machine, you'll want to download the dataset and put it in the same directory as the code.

Run the program using the command 'python3 uldp.py'

### Plots
The uldp.py Python file outputs plots (as pop-ups), including:
- Final sample label predictions for the seven neural networks
- Confusion matrices for the seven neural networks

... you will notice that there are a lot of commented out lines. This is because there are more than 85 plots generated in this one file, and to save your Python interpreter, I turned those plots off. If you want to see them, uncomment any line that has a variable with the word "plot" in it.

### Fair warning on reuse
Some of the dataset cleanup is fairly manual (as it always is), so using this repo for other datasets will require modification to the data loading and data processing sections (lines ~200 to ~350) which are mostly normalizing, cleaning, and otherwise preprocessing the two datasets to be classified.