## CS 7641_Machine_Learning 
### Assignment 2: Randomized Optimization

### Code Location
https://github.com/egemzer/CS7641_Machine_Learning/tree/master/Randomized%20Optimization

### Overview

This project was completed by @egemzer (gth659q) for master's degree course in Machine Learning at Georgia Tech. 

The project has two parts:
- In **Part 1**, we compare and contrast four different random search algorithms (RSAs) against four canonical optimization problems (n-Queens, Travelling Salesperson, FlipFlop, and Knapsack), to exercise and demonstrate the relative strengths and weaknesses of each RSA. 
    -    The four RSAs are: Random Hill Climbing (RHC), Genetic Algorithms (GA), Simulated Annealing (SA), and MIMIC.
- In **Part 2**, we use three of the four search algorithms, comparing to gradient descent, to determine the continuous, real-valued weights for a neural network model for the dataset below.

### Dataset
**Contraceptives Dataset** - in repo as a CSV (runs with the code, no effort required) and available at https://www.openml.org/d/23

### Libraries
python 3.7.x, sklearn, matplotlib, pandas, numpy, timeit, datetime, copy, imblearn, warnings, 

also: mlrose hiive version 1.2.0 at commit 452810cde6502963afd75bb44354edf24e562063
- to install: '-m pip install git+https://github.com/hiive/mlrose.git@452810cde6502963afd75bb44354edf24e562063'

### How to Run the Code
To test this code on a local machine, you'll want to download the dataset and put it in the same directory as the code.

#### ...for Optimization Problems
All of the optimization problems are set to a high complexity problem domain (22 Queens, 22 City TSP, 20 Item Knapsack, and 50-bit FlipFlop) to illustrate the strengths and weaknesses of the RSAs.
**Compute Warning**: These algorithms, especially MIMIC and GA on the high complexity problems, take hours or even days to converge.

- run 22-Queens using the command 'python3 queens.py'
- run 22-City TSP using the command 'python3 travelling_salesman.py'
- run 50-bit FlipFlop using the command 'python3 flipflop.py'
- run 20-item Knapsack using the command 'python3 knapsack.py'

#### ...for Neural Network Weight Optimization
- run gradient descent, RHC, SA, and GA neural network weight optimization (sequentially) using the command 'python3 neural_net_opt.py'

### Plots
Each Python file outputs plots (as pop-ups), including:
- learning curves (time based on F1 model performance based)
- prediction distributions (model's predictions on holdout test data)
- a summary of all algorithms' performances on a given optimization problem

### Fair warning on reuse
(1) the hiive mlrose library is stable but evolving daily, and this code will inevitably be broken in the near future due to backward compatibility issues with newer mlrose versions. Use the command above to install a known-working commit before running this code.

(2) Some of the dataset cleanup is fairly manual (as it always is), so using the neural_net_opt.py for other datasets will require modification to the first ~100 lines which are mostly normalizing, cleaning, and otherwise preprocessing the dataset for classification.