import mlrose
import random
import timeit
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
matplotlib.use('TkAgg')
import datetime

from mlrose.generators import KnapsackGenerator

"""
Knapsack Problem:
Given a set of items, each with a weight and a value, determine the number of each item to include in a collection 
so that the total weight is less than or equal to a given limit and the total value is as large as possible.
Citation: https://en.wikipedia.org/wiki/Knapsack_problem
"""

iterations_range = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
OUTPUT_DIRECTORY = './Experiments/Knapsack'
item_types = 35
max_items = 20
print('%s Item Type, %s Item Knapsack Value Optimization' %(item_types, max_items))

# Generate the fitness problem and the optimization function
problem = KnapsackGenerator.generate(seed=random.seed(17), number_of_items_types=item_types,
									max_item_count=max_items,max_weight_per_item=5,
									 max_value_per_item=5,max_weight_pct=0.6)

def plot_fitness_single_algo(title, iterations, fitness, algorithm):
	plt.title(title)
	plt.xlabel("Number of Iterations")
	plt.ylabel("Fitness, ie Maximum Value")
	plt.plot(iterations, fitness, 'o-', color="r", label=algorithm)
	plt.legend(loc="best")


def plot_time_single_algo(title, iterations, time, algorithm):
	plt.title(title)
	plt.xlabel("Number of Iterations")
	plt.ylabel("Convergence Time (seconds)")
	plt.plot(iterations, time, 'o-', color="b", label=algorithm)
	plt.legend(loc="best")


#========== Random Hill Climb ==========#
time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
print("Starting Random Hill Climbing at: " + time)
# Optimize the algorithm parameters (Manual)
def opt_rhc_params():
	rhc = mlrose.runners.RHCRunner(problem=problem,
						experiment_name='Optimal Params for Knapsack RHC',
						output_directory=OUTPUT_DIRECTORY,
						seed=random.seed(17),
						iteration_list=2 ** np.arange(1,12),
						max_attempts=1000,
						restart_list=[0, 5, 25, 75])
	rhc_df_run_stats, rhc_df_run_curves = rhc.run()
	ideal_rs = rhc_df_run_stats[['current_restart']].iloc[rhc_df_run_stats[['Fitness']].idxmax()] # from the output of the experiment above
	return rhc_df_run_stats, rhc_df_run_curves, ideal_rs

# Done on a complex example then  hard coded optimized parameter value(s).
# rhc_df_run_stats, rhc_df_run_curves, ideal_rs = opt_rhc_params()

ideal_rs = 75  # this came from the results of the experiment commented out, above.
rhc_best_state = []
rhc_best_fitness = []
rhc_convergence_time = []
for iter in iterations_range:
	start_time = timeit.default_timer()
	best_state, best_fitness, curve = mlrose.random_hill_climb(problem=problem, max_iters=iter,
														max_attempts=1000, restarts=ideal_rs,
														curve=True)
	end_time = timeit.default_timer()
	convergence_time = (end_time - start_time)  # seconds
	rhc_best_state.append(best_state)
	rhc_best_fitness.append(best_fitness)
	rhc_convergence_time.append(convergence_time)

print('The fitness at the best state found using Random Hill Climbing is: ', max(rhc_best_fitness))

#========== Genetic Algorithms ==========#
time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
print("Starting Genetic Algorithms at: " + time)
# Optimize the algorithm parameters (Manual)
def opt_ga_params():
	ga = mlrose.runners.GARunner(problem=problem,
					  experiment_name='Optimal Params for Knapsack GA',
					  output_directory=OUTPUT_DIRECTORY,
					  seed=random.seed(17),
					  iteration_list=2 ** np.arange(1,12),
					  max_attempts=1000,
					  population_sizes=[200, 300, 400, 500],
					  mutation_rates=[0.2, 0.4, 0.6, 0.8, 1])
	ga_df_run_stats, ga_df_run_curves = ga.run()
	ideal_pop_size = ga_df_run_stats[['Population Size']].iloc[ga_df_run_stats[['Fitness']].idxmax()]  # from the output of the experiment above
	ideal_mutation_rate = ga_df_run_stats[['Mutation Rate']].iloc[ga_df_run_stats[['Fitness']].idxmax()]  # from the output of the experiment above
	return ga_df_run_stats, ga_df_run_curves, ideal_pop_size, ideal_mutation_rate

# Done on a complex example then  hard coded optimized parameter value(s).
# ga_df_run_stats, ga_df_run_curves, ideal_pop_size, ideal_mutation_rate = opt_ga_params()

ideal_pop_size = 500  # this came from the results of the experiment commented out, above.
ideal_mutation_rate = 0.2  # this came from the results of the experiment commented out, above.

ga_best_state = []
ga_best_fitness = []
ga_convergence_time = []
for iter in iterations_range:
	start_time = timeit.default_timer()
	best_state, best_fitness, curve = mlrose.genetic_alg(problem=problem, curve=True,
												  mutation_prob = ideal_mutation_rate,
					      max_attempts = 1000, max_iters = iter, pop_size=ideal_pop_size)
	end_time = timeit.default_timer()
	convergence_time = (end_time - start_time)  # seconds
	ga_best_state.append(best_state)
	ga_best_fitness.append(best_fitness)
	ga_convergence_time.append(convergence_time)

print('The fitness at the best route found using genetic algorithms is: ', max(ga_best_fitness))

#========== Simulated Annealing ==========#
time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
print("Starting Simulated Annealing at: " + time)
# Optimize the algorithm parameters (Manual)
def opt_sa_params():
	sa = mlrose.runners.SARunner(problem=problem,
				  experiment_name='Optimal Params for Knapsack SA',
				  output_directory=OUTPUT_DIRECTORY,
				  seed=random.seed(17),
				  iteration_list=2 ** np.arange(12),
				  max_attempts=500,
				  temperature_list=[1, 10, 50, 100, 250, 500, 1000, 2500, 5000, 10000])
	sa_df_run_stats, sa_df_run_curves = sa.run()
	ideal_temp = sa_df_run_stats[['Temperature']].iloc[sa_df_run_stats[['Fitness']].idxmax()]  # from the output of the experiment above
	return sa_df_run_stats, sa_df_run_curves, ideal_temp

# Done on a complex example then  hard coded optimized parameter value(s).
sa_df_run_stats, sa_df_run_curves, ideal_initial_temp = opt_sa_params()

ideal_initial_temp = 1000  # this came from the results of the experiment commented out, above.

sa_best_state = []
sa_best_fitness = []
sa_convergence_time = []
for iter in iterations_range:
	start_time = timeit.default_timer()
	best_state, best_fitness, curve = mlrose.simulated_annealing(problem=problem, max_attempts = 500,
									max_iters = iter, curve=True,
									schedule=mlrose.GeomDecay(init_temp=ideal_initial_temp))
	end_time = timeit.default_timer()
	convergence_time = (end_time - start_time)  # seconds
	sa_best_state.append(best_state)
	sa_best_fitness.append(best_fitness)
	sa_convergence_time.append(convergence_time)

print('The fitness at the best state found using simulated annealing is: ', max(sa_best_fitness))

#========== Mutual-Information-Maximizing Input Clustering (MIMIC) ==========#
time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
print("Starting MIMIC at: " + time)
# Optimize the algorithm parameters (Manual)
def opt_mimic_params():
	mmc = mlrose.runners.MIMICRunner(problem=problem,
				  experiment_name='Optimal Params for Knapsack MIMIC',
				  output_directory=OUTPUT_DIRECTORY,
				  seed=random.seed(17),
				  iteration_list=2 ** np.arange(13),
				  max_attempts=1000, population_sizes=[300, 400, 500],
				  keep_percent_list=[0.1, 0.25, 0.5, 0.75])
	mmc_df_run_stats, mmc_df_run_curves = mmc.run()
	ideal_keep_prcnt = mmc_df_run_stats[['Keep Percent']].iloc[mmc_df_run_stats[['Fitness']].idxmax()]  # from the output of the experiment above
	ideal_pop_size = mmc_df_run_stats[['Population Size']].iloc[mmc_df_run_stats[['Fitness']].idxmax()]  # from the output of the experiment above
	return mmc_df_run_stats, mmc_df_run_curves, ideal_keep_prcnt, ideal_pop_size

# Done on a complex example then  hard coded optimized parameter value(s).
# mmc_df_run_stats, mmc_df_run_curves, ideal_keep_prcnt, ideal_pop_size = opt_mimic_params()

ideal_pop_size_mimic = 500  # this came from the results of the experiment commented out, above.
ideal_keep_prcnt = 0.5 # this came from the results of the experiment commented out, above.

mimic_best_state = []
mimic_best_fitness = []
mimic_convergence_time = []
for iter in iterations_range:
	start_time = timeit.default_timer()
	best_state, best_fitness, curve = mlrose.mimic(problem=problem, keep_pct=ideal_keep_prcnt,
											max_attempts=1000, max_iters=iter,
											pop_size=ideal_pop_size_mimic, curve=True)
	end_time = timeit.default_timer()
	convergence_time = (end_time - start_time)  # seconds
	mimic_best_state.append(best_state)
	mimic_best_fitness.append(best_fitness)
	mimic_convergence_time.append(convergence_time)

print('The fitness at the best state found using MIMIC is: ', max(mimic_best_fitness))

#======= Plots for all individual learners==========#

# Random Hill Climbing
fig1, ax1 = plt.subplots()
ax1.title.set_text("%s Item Type, %s Items Selected Knapsack, Tuned Random Hill Climbing" %(item_types, max_items))
ax2 = ax1.twinx()
ax1.plot(iterations_range, (rhc_best_fitness), 'r-')
ax2.plot(iterations_range, rhc_convergence_time, 'b-')
ax1.set_xlabel('Number of Iterations')
ax1.set_ylabel('Fitness', color='g')
ax2.set_ylabel('Time (s)', color='b')

# Genetic Algorithms
fig2, ax3 = plt.subplots()
ax3.title.set_text("%s Item Type, %s Items Selected Knapsack, Tuned Genetic Algorithms" %(item_types) %(max_items))
ax4 = ax3.twinx()
ax3.plot(iterations_range, (ga_best_fitness), 'r-')
ax4.plot(iterations_range, ga_convergence_time, 'b-')
ax3.set_xlabel('Number of Iterations')
ax3.set_ylabel('Fitness', color='g')
ax4.set_ylabel('Time (s)', color='b')

# Simulated Annealing
fig3, ax5 = plt.subplots()
ax5.title.set_text("%s Item Type, %s Items Selected Knapsack, Tuned Simulated Annealing" %(item_types, max_items))
ax6 = ax5.twinx()
ax5.plot(iterations_range, (sa_best_fitness), 'r-')
ax6.plot(iterations_range, sa_convergence_time, 'b-')
ax5.set_xlabel('Number of Iterations')
ax5.set_ylabel('Fitness', color='g')
ax6.set_ylabel('Time (s)', color='b')

# MIMIC
fig4, ax7 = plt.subplots()
ax7.title.set_text("%s Item Type, %s Items Selected Knapsack, Tuned MIMIC" %(item_types, max_items))
ax8 = ax7.twinx()
ax7.plot(iterations_range, (mimic_best_fitness), 'r-')
ax8.plot(iterations_range, mimic_convergence_time, 'b-')
ax7.set_xlabel('Number of Iterations')
ax7.set_ylabel('Fitness', color='g')
ax8.set_ylabel('Time (s)', color='b')

#======= Comparison of all four optimization algorithms ==========#
fig5, (ax9, ax10) = plt.subplots(1, 2, figsize=(15, 5))
fig5.suptitle('Comparing Random Search Optimizers on %s Item Knapsack: Fitness and Convergence Time' %(max_items))

ax9.set(xlabel="Number of Iterations", ylabel="Fitness")
ax9.grid()
ax9.plot(iterations_range, (rhc_best_fitness), 'o-', color="r", label='Random Hill Climbing')
ax9.plot(iterations_range, (ga_best_fitness), 'o-', color="b", label='Genetic Algorithms')
ax9.plot(iterations_range, (sa_best_fitness), 'o-', color="m", label='Simulated Annealing')
ax9.plot(iterations_range, (mimic_best_fitness), 'o-', color="g", label='MIMIC')
ax9.legend(loc="best")

ax10.set(xlabel="Number of Iterations", ylabel="Convergence Time (in seconds)")
ax10.grid()
ax10.plot(iterations_range, rhc_convergence_time, 'o-', color="r", label='Random Hill Climbing')
ax10.plot(iterations_range, ga_convergence_time, 'o-', color="b", label='Genetic Algorithms')
ax10.plot(iterations_range, sa_convergence_time, 'o-', color="m", label='Simulated Annealing')
ax10.plot(iterations_range, mimic_convergence_time, 'o-', color="g", label='MIMIC')
ax10.legend(loc="best")

plt.show()
print("you are done!")
