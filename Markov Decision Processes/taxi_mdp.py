import gym # openAi gym
import numpy as np
import timeit
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns
import random
from IPython.display import clear_output
import warnings
warnings.filterwarnings('ignore')

######## Helper Functions ########
def random_policy_steps_count(env):
    state = env.reset()
    counter = 0
    reward = None
    while reward != 20:
        state, reward, done, info = env.step(env.action_space.sample())
        counter += 1
    return counter

def policy_eval(policy, env, discount_factor=0.95, theta=0.001):
    """
    Evaluate a policy given an environment and a full description of the environment's dynamics.

    Args:
        policy: [S, A] shaped matrix representing the policy.
        env: OpenAI env. env.P represents the transition probabilities of the environment.
            env.P[s][a] is a list of transition tuples (prob, next_state, reward, done).
            env.nS is a number of states in the environment.
            env.nA is a number of actions in the environment.
        theta: We stop evaluation once our value function change is less than theta for all states.
        discount_factor: Gamma discount factor.

    Returns:
        Vector of length env.nS representing the value function.
        Num_iterations : the number of iterations required to converge
    """
    # Start with a random (all 0) value function
    V = np.zeros(env.env.nS)
    while True:
        delta = 0  # delta = change in value of state from one iteration to next

        for state in range(env.env.nS):  # for all states
            val = 0  # initiate value as 0

            for action, act_prob in enumerate(policy[state]):  # for all actions/action probabilities
                for prob, next_state, reward, done in env.env.P[state][action]:  # transition probabilities,state,rewards of each action
                    val += act_prob * prob * (reward + discount_factor * V[next_state])  # eqn to calculate
            delta = max(delta, np.abs(val - V[state]))
            V[state] = val
        if delta < theta:  # break if the change in value is less than the threshold (theta)
            break
    return np.array(V)


def policy_iteration(env, policy_eval_fn=policy_eval, discount_factor=0.95):
    """
    Policy Improvement Algorithm. Iteratively evaluates and improves a policy
    until an optimal policy is found.

    Args:
        env: The OpenAI envrionment.
        policy_eval_fn: Policy Evaluation function that takes 3 arguments:
            policy, env, discount_factor.
        discount_factor: gamma discount factor.

    Returns:
        A tuple (policy, V).
        policy is the optimal policy, a matrix of shape [S, A] where each state s
        contains a valid probability distribution over actions.
        V is the value function for the optimal policy.

    """

    def one_step_lookahead(state, V):
        """
        Helper function to calculate the value for all action in a given state.

        Args:
            state: The state to consider (int)
            V: The value to use as an estimator, Vector of length env.nS

        Returns:
            A vector of length env.nA containing the expected value of each action.
        """
        A = np.zeros(env.env.nA)
        for a in range(env.env.nA):
            for prob, next_state, reward, done in env.env.P[state][a]:
                A[a] += prob * (reward + discount_factor * V[next_state])
        return A

    # Start with a random policy
    policy = np.ones([env.env.nS, env.env.nA]) / env.env.nA
    count = 0

    while True:
        curr_pol_val = policy_eval_fn(policy, env, discount_factor)  # eval current policy
        policy_stable = True  # Check if policy did improve (Set it as True first)
        for state in range(env.env.nS):  # for each states
            chosen_act = np.argmax(policy[state])  # best action (Highest prob) under current policy
            act_values = one_step_lookahead(state, curr_pol_val)  # use one step lookahead to find action values
            best_act = np.argmax(act_values)  # find best action
            if chosen_act != best_act:
                policy_stable = False  # Greedily find best action
                count += 1
            policy[state] = np.eye(env.env.nA)[best_act]  # update
        if policy_stable:
            return policy, curr_pol_val, count

def value_iteration(env, theta=0.001, discount_factor=1.0):
    """
    Value Iteration Algorithm.

    Args:
        env: OpenAI env. env.P represents the transition probabilities of the environment.
            env.P[s][a] is a list of transition tuples (prob, next_state, reward, done).
            env.nS is a number of states in the environment.
            env.nA is a number of actions in the environment.
        theta: We stop evaluation once our value function change is less than theta for all states.
        discount_factor: Gamma discount factor.

    Returns:
        A tuple (policy, V) of the optimal policy and the optimal value function.
    """

    def one_step_lookahead(state, V):
        """
        Helper function to calculate the value for all action in a given state.

        Args:
            state: The state to consider (int)
            V: The value to use as an estimator, Vector of length env.nS

        Returns:
            A vector of length env.nA containing the expected value of each action.
        """
        A = np.zeros(env.env.nA)
        for act in range(env.env.nA):
            for prob, next_state, reward, done in env.env.P[state][act]:
                A[act] += prob * (reward + discount_factor * V[next_state])
        return A

    V = np.zeros(env.env.nS)
    count = 0
    while True:
        delta = 0  # checker for improvements across states
        for state in range(env.env.nS):
            act_values = one_step_lookahead(state, V)  # lookahead one step
            best_act_value = np.max(act_values)  # get best action value
            delta = max(delta, np.abs(best_act_value - V[state]))  # find max delta across all states
            V[state] = best_act_value  # update value to best action value
        if delta < theta:  # if max improvement less than threshold
            break
        else:
            count += 1
    policy = np.zeros([env.env.nS, env.env.nA])
    for state in range(env.env.nS):  # for all states, create deterministic policy
        act_val = one_step_lookahead(state, V)
        best_action = np.argmax(act_val)
        policy[state][best_action] = 1

    return policy, V, count

def total_reward(policy, env, discount_factor):
    V = policy_eval(policy, env, discount_factor)
    return V[0]

def check_same_policy(policy1, policy2):
    for x in range(len(policy1[0])):
        if not (policy1[0][x] == policy2[0][x]).all():
            return "Not the same Policy"
            break
    return "Same Policy"

def count(policy, env):
    curr_state = env.reset()
    counter = 0
    reward = None
    while reward != 20:
        state, reward, done, info = env.step(np.argmax(policy[curr_state]))
        curr_state = state
        counter += 1
    return counter

def total_q_reward(policy, env):
    curr_state = env.reset()
    total_reward = 0
    reward = None
    while reward != 20:
        state, reward, done, info = env.step(np.argmax(policy[curr_state]))
        curr_state = state
        total_reward  += reward
    return total_reward

def view_policy(policy, env):
    curr_state = env.reset()
    counter = 0
    reward = None
    while reward != 20:
        state, reward, done, info = env.step(np.argmax(policy[0][curr_state]))
        curr_state = state
        counter += 1
        env.env.s = curr_state
        env.render()


def Q_learning_train(env, alpha, gamma, epsilon, episodes):
    """Q Learning Algorithm with epsilon greedy

    Args:
        env: Environment
        alpha: Learning Rate --> Extent to which our Q-values are being updated in every iteration.
        gamma: Discount Rate --> How much importance we want to give to future rewards
        epsilon: Probability of selecting random action instead of the 'optimal' action
        episodes: No. of episodes to train on

    Returns:
        Q-learning Trained policy

    """

    """Training the agent"""

    # For plotting metrics
    all_epochs = []
    all_penalties = []

    # Initialize Q table of size (n_states x n_actions) with all zeroes
    q_table = np.zeros([env.observation_space.n, env.action_space.n])

    for i in range(1, episodes + 1):
        state = env.reset()

        epochs, penalties, reward, = 0, 0, 0
        done = False

        while not done:
            if random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()  # Explore action space randomly
            else:
                action = np.argmax(q_table[state])  # Exploit learned values by choosing optimal values

            next_state, reward, done, info = env.step(action)

            old_value = q_table[state, action]
            next_max = np.max(q_table[next_state])

            new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
            q_table[state, action] = new_value

            if reward == -10:
                penalties += 1

            state = next_state
            epochs += 1

        if i % 100 == 0:
            clear_output(wait=True)
            # print(f"Episode: {i}")
    # Start with a random policy
    policy = np.ones([env.env.nS, env.env.nA]) / env.env.nA

    for state in range(env.env.nS):  # for each states
        best_act = np.argmax(q_table[state])  # find best action
        policy[state] = np.eye(env.env.nA)[best_act]  # update

    print("Training finished.\n")
    return policy, q_table, epochs

####################
#### Large Taxi ####
####################

gym.envs.register(
    id='TaxiLarge-v0',
    entry_point='large_taxi:TaxiEnv',
    max_episode_steps=100000,)


env = gym.make('TaxiLarge-v0')
env.reset()
env.render()
n_states = env.observation_space.n
print("Total number of states: ", n_states)  #Total no. of states

# Random Iteration on Taxi
random_policy = np.ones([env.env.nS, env.env.nA]) / env.env.nA
random_iteration_start_time = timeit.default_timer()
random_policy_counts = [random_policy_steps_count(env=env) for i in range(50)]
random_iteration_end_time = timeit.default_timer()
random_iteration_solution_time = (random_iteration_end_time - random_iteration_start_time)  # in seconds
plt.figure(figsize=(8, 6))
sns.distplot(random_policy_counts)
plt.title("Number of random steps needed to solve a %s state gridworld Taxi" %(n_states))
plt.xlabel("Number of Iterations")
plt.ylabel("Probability")
plt.savefig('random-policy-taxi-%s.png' %(n_states))
print("An agent using random search takes about an average of %s steps and %s seconds to successfully complete its mission." % (str(int(np.mean(random_policy_counts))), str(int(random_iteration_solution_time))))
random_policy_total_reward = total_reward(policy=random_policy, env=env, discount_factor=0.95)


# Policy Iteration on Taxi
env.reset()
policy_iteration_start_time = timeit.default_timer()
pol_iter_policy, pol_iter_values, pol_num_iterations = policy_iteration(env, policy_eval, discount_factor=0.95)
pol_iter_policy = pol_iter_policy, pol_iter_values
policy_iteration_end_time = timeit.default_timer()
policy_iteration_solution_time = (policy_iteration_end_time - policy_iteration_start_time) # in seconds
pol_count = count(env=env, policy=pol_iter_policy[0])
pol_counts = [count(env=env, policy=pol_iter_policy[0]) for i in range(50)]
print("An agent using a policy which has been policy-iterated takes about an average of %s steps. It took %s seconds and %s iterations to determine the optimal policy." %(str(int(np.mean(pol_counts))), str(int(policy_iteration_solution_time)), str(pol_num_iterations)))
plt.figure(figsize=(8, 6))
sns.distplot(pol_counts)
plt.title("Number of policy-iterated steps needed to solve a %s state gridworld Taxi" %(n_states))
plt.xlabel("Number of Steps")
plt.ylabel("Probability")
plt.savefig('policy-iterated-taxi-%s.png' %(n_states))


# Value Iteration on Taxi
env.reset()
value_iteration_start_time = timeit.default_timer()
val_iter_policy, val_iter_values, val_num_iterations = value_iteration(env, discount_factor=0.95)
val_iter_policy = val_iter_policy, val_iter_values
value_iteration_end_time = timeit.default_timer()
value_iteration_solution_time = (value_iteration_end_time - value_iteration_start_time) # in seconds
val_count = count(env=env, policy=val_iter_policy[0])
val_counts = [count(env=env, policy=val_iter_policy[0]) for i in range(50)]
print("An agent using a policy which has been value-iterated takes about an average of %s steps. It took %s seconds and %s iterations to determine the optimal policy." %(str(int(np.mean(val_counts))), str(int(value_iteration_solution_time)), str(val_num_iterations)))
plt.figure(figsize=(8, 6))
sns.distplot(val_counts)
plt.title("Number of value-iterated steps needed to solve a %s state gridworld Taxi" %(n_states))
plt.xlabel("Number of Steps")
plt.ylabel("Probability")
plt.savefig('value-iterated-taxi-%s.png' %(n_states))

# Epsilon-greedy Q-Learner on Taxi
env.reset()
qlearner_start_time = timeit.default_timer()
Q_learn_pol, q_table, qlearner_num_iterations = Q_learning_train(env=env,alpha=0.2,gamma=0.95,epsilon=0.9,episodes=100000)
Q_learn_pol = Q_learn_pol, q_table
q_reward = total_q_reward(policy=Q_learn_pol[0], env=env)
qlearner_end_time = timeit.default_timer()
qlearner_solution_time = (qlearner_end_time - qlearner_start_time) # in seconds
Q_Learning_counts = count(env=env, policy=Q_learn_pol[0])
Q_counts = [count(env=env, policy=Q_learn_pol[0]) for i in range(50)]
print("An agent using a policy which has been Q-Learned takes about an average of %s steps. It took %s seconds and %s iterations to determine the optimal policy." %(str(int(np.mean(Q_counts))), str(int(qlearner_solution_time)), str(qlearner_num_iterations)))
plt.figure(figsize=(8, 6))
sns.distplot(Q_counts)
plt.title("Number of Q-Learner steps needed to solve a %s state gridworld Taxi" %(n_states))
plt.xlabel("Number of Steps")
plt.ylabel("Probability")
plt.savefig('qlearner-taxi-%s.png' %(n_states))

# Comparing random iteration, policy iteration, value iteration, and Q-Learner
comparison_random_policy = check_same_policy(random_policy, pol_iter_policy)
comparison_random_value = check_same_policy(random_policy,val_iter_policy)
comparison_random_qlearner = check_same_policy(random_policy,Q_learn_pol)
comparison_policy_value = check_same_policy(pol_iter_policy,val_iter_policy)
comparison_policy_qlearner = check_same_policy(pol_iter_policy,Q_learn_pol)
comparison_value_qlearner = check_same_policy(val_iter_policy,Q_learn_pol)
print("Random Iteration and Policy Iteration Results: %s" %(comparison_random_policy))
print("Random Iteration and Value Iteration Results: %s" %(comparison_random_value))
print("Random Iteration and QLearner Iteration Results: %s" %(comparison_random_qlearner))
print("Policy Iteration and Value Iteration Results: %s" %(comparison_policy_value))
print("Policy Iteration and QLearner Iteration Results: %s" %(comparison_policy_qlearner))
print("Value Iteration and QLearner Iteration Results: %s" %(comparison_value_qlearner))

# Comparing random iteration, policy iteration, value iteration, and Q-Learner total rewards
print("Random Iteration Total Reward: %s" %(str(random_policy_total_reward)))
print("Policy Iteration Total Reward: %s" %(str(pol_iter_values[0])))
print("Value Iteration Total Reward: %s" %(str(val_iter_values[0])))
print("QLearner Total Reward: %s" %(str(q_reward)))


plt.show()
print("you got this kiddo, don't give up!  Last assignment!")