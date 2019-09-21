## CS 7641_Machine_Learning 
### Assignment 1: Supervised Learning Classification

This project was completed by @egemzer (gth659q) for master's degree course in Machine Learning at Georgia Tech.
### Libraries
python 3.7.x, sklearn, matplotlib, pandas, numpy, timeit, copy, imblearn, warnings, itertools, pydotplus, tabulate

### Datasets
Dataset 1: **Contraceptives Dataset** - in repo and available at https://www.openml.org/d/23
    
Dataset 2: **Arrhythmia** - in repo and available at https://www.openml.org/d/5

### How to Run the Code
To test this code on a local machine, you'll want to download the two datasets and put them in the same directory as the code, then run Learners.py using the command 'python Learners.py'

Note: I implemented a gridsearch with 3-fold cross validation. It takes some time, with as many cores running in parallel as your computer will spare, and you'll see output in the console while it runs. This is verbose logging, and it's expected behavior. Also, if your async thread manager is as poor as mine, you may also have frequent thread drops, which is annoying and I apologize, but you'll need to rerun the script until you get to the end, where you'll see a consolidate table of results.

### Plots
Each model outputs plots (as pop-ups), at a minimum:
- model complexity / validation curves (based on one or two hyperparameters)
- learning curves (time based on F1 model performance based)
- prediction distributions
- a confusion matrix

Additionally,
- Iterative models (Neural Networks and SVM) additionally output a plot that compares model score to the number of iterations.
- Tree-based models (Decision Trees and Boosting) output tree plots into the curent directory, not as a popup.

### Fair warning on reuse
Some of the dataset cleanup is fairly manual (as it always is), so using this repo for other datasets will require modification to the first ~110 lines which are mostly normalizing, cleaning, and otherwise preprocessing the two datasets under test.