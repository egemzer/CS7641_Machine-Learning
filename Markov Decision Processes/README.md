## CS 7641_Machine_Learning 
### Assignment 4: Markov Decision Processes

### Code Location
https://github.com/egemzer/CS7641_Machine_Learning/tree/master/Markov%20Decision%20Processes

### Overview

This project was completed by @egemzer (gth659q) for master's degree course in Machine Learning at Georgia Tech. 

In this project, I explore Markov decision processes (MDPs) and reinforcement learning by examining and comparing two different MDPs solved via value iteration, policy iteration, and reinforcement learning.
+ A modified version of the taxi gridworld problem from OpenAi Gym
+ A non-gridworld forest management problem from pymdptoolbox

#### Part 1 
Solve and compare two MDPs using (1) Value Iteration and (2) Policy Iteration

#### Part 2
Solve both MDPs using Q-Learning (a type of Reinforcement Learning) and compare outcomes and performance with Part 1.

### Libraries
python 3.7.x, sklearn, matplotlib, pandas, numpy, timeit, datetime, warnings, mdptoolbox, seaborn, IPython.display, gym

### How to Run the Code
Run the Taxi mdp using the command 'python3 taxi_mdp.py'. You will need large_taxi.py in the same directory.

Run the Forest Management mdp using the command 'python3 forest_mdp.py'

### Plots
The Python file outputs plots (as pop-ups), including:
- Probability curves for each solution method