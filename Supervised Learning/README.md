## CS 7641_Machine_Learning 
### Assignment 1: Supervised Learning Classification

This project was completed by @egemzer (gth659q) for master's degree course in Machine Learning at Georgia Tech. The project seeks to compare and contrast the tuning and performance of five different machine learning algorithms, or "learners", on two different classification problems, or datasets.

### Code Location
https://github.com/egemzer/CS7641_Machine_Learning/tree/master/Supervised%20Learning

### Datasets
Dataset 1: **Contraceptives Dataset** - in repo and available at https://www.openml.org/d/23
    
Dataset 2: **Arrhythmia** - in repo and available at https://www.openml.org/d/5

### Libraries
python 3.7.x, sklearn, matplotlib, pandas, numpy, timeit, copy, imblearn, warnings, itertools, pydotplus, tabulate

### How to Run the Code
To test this code on a local machine, you'll want to download the two datasets and put them in the same directory as the code, then run Learners.py using the command 'python3 Learners.py'

Note: I implemented a gridsearch with 3-fold cross validation. It takes some time, with as many cores running in parallel as your computer will spare, and you'll see output in the console while it runs. This is verbose logging, and it's expected behavior. 

Also, if your async thread manager is as poor as mine, you may also have frequent thread drops, which is annoying and I apologize, but you'll need to rerun the script until you get to the end, where you'll see a consolidated table of results in the terminal.

### Plots
Each model outputs plots (as pop-ups), at a minimum:
- model complexity / validation curves (based on one or two hyperparameters)
- learning curves (time based on F1 model performance based)
- prediction distributions (model's predictions on holdout test data)
- a confusion matrix

Additionally,
- Iterative models (Neural Networks and SVM) output a plot that compares model F1 score to the number of iterations.
- Tree-based models (Decision Trees and Boosting) output tree plots into the current directory, not as a popup.

### Fair warning on reuse
Some of the dataset cleanup is fairly manual (as it always is), so using this repo for other datasets will require modification to the first ~125 lines which are mostly normalizing, cleaning, and otherwise preprocessing the two datasets to be classified.