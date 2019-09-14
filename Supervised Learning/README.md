## CS 7641_Machine_Learning 
### Assignment 1: Supervised Learning Classification

This project was completed by @egemzer (gth659q) for master's degree course in Machine Learning at Georgia Tech.
### Libraries Used
python 3.7.x, scikit-learn, matplotlib, pandas, numpy, timeit

### Datasets
Dataset 1: **Contraceptives Dataset** - in repo and available at https://www.openml.org/d/23
    
Dataset 2: **Arrhythmia** - in repo and available at https://www.openml.org/d/5

### How to Run the Code
To test this code on a local machine, you'll want to download the two datasets and put them in the same directory as the code, then run Learners.py using the command 'python Learners.py'

Note: I implemented a gridsearch with cross validation over three different decision tree classifiers. It takes about two minutes with as many cores running in parallel as your computer will spare, and you'll see output in the console while it runs. This is verbose logging, and it's expected behavior.
### Interpreting the Plots
TODO

### Fair warning on reuse
Some of the dataset cleanup is fairly manual (as it always is), so using this repo for other datasets will require modification to the first ~110 lines which are mostly normalizing, cleaning, and otherwise preprocessing the two datasets under test.