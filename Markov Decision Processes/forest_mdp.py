import mdptoolbox.example
import mdptoolbox
import numpy as np
import timeit

##### Helper Functions ######
def check_same_policy(policy1, policy2):
    for x in range(len(policy1)):
        if not (policy1[x] == policy2[x]):
            return "Not the same Policy"
            break
    return "Same Policy"

def samplefrom(distribution):
    return (np.random.choice(len(distribution), 1, p=distribution))[0]

def playtransition(mdp, state, action):
    nextstate = samplefrom(mdp.P[action][state])
    return nextstate, mdp.R[state][action]

def epsilon_greedy_exploration(Q, epsilon, num_actions):
    def policy_exp(state):
        probs = np.ones(num_actions, dtype=float) * epsilon / num_actions
        best_action = np.argmax(Q[state])
        probs[best_action] += (1.0 - epsilon)
        return probs
    return policy_exp

def UCB_exploration(Q, num_actions, beta=1):
    def UCB_exp(state, N, t):
        probs = np.zeros(num_actions, dtype=float)
        Q_ = Q[state,:]/max(Q[state,:]) + np.sqrt(beta*np.log(t+1)/(2*N[state]))
        best_action = Q_.argmax()
        probs[best_action] = 1
        return probs
    return UCB_exp

def q_learning(mdp, num_states, num_actions, num_episodes, T_max, epsilon=0.01, discount=0.95):
    Q = np.zeros((num_states, num_actions))
    episode_rewards = np.zeros(num_episodes)
    policy = np.ones(num_states)
    V = np.zeros((num_episodes, num_states))
    N = np.zeros((num_states, num_actions))
    for i_episode in range(num_episodes):
        # # epsilon greedy exploration
        greedy_probs = epsilon_greedy_exploration(Q, epsilon, num_actions)

        # # UCB exploration
        # greedy_probs = UCB_exploration(Q, num_actions)
        state = np.random.choice(np.arange(num_states))
        for t in range(T_max):
            # epsilon greedy exploration
            action_probs = greedy_probs(state)

            # # UCB exploration
            # action_probs = greedy_probs(state, N=N, t=t)
            action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
            next_state, reward = playtransition(mdp, state, action)
            episode_rewards[i_episode] += reward
            N[state, action] += 1
            alpha = 1 / (t + 1) ** 0.8
            best_next_action = np.argmax(Q[next_state])
            td_target = reward + discount * Q[next_state][best_next_action]
            td_delta = td_target - Q[state][action]
            Q[state][action] += alpha * td_delta
            state = next_state
        V[i_episode, :] = Q.max(axis=1)
        policy = Q.argmax(axis=1)

    return V, policy, episode_rewards, N

####### The problem setup ###########
num_states = 5
probabilities, rewards = mdptoolbox.example.forest(S=num_states, r1=10, r2=50, p=0.1, is_sparse=False)
# note this means 5 trees in the forest, r1 is the immediate reward to cut the trees, r2 is the
# long term reward for cutting as they are more grown, p is the probability of a forest fire in a given year


##### Policy Iteration ####
policy_iteration = mdptoolbox.mdp.PolicyIteration(transitions=probabilities, reward=rewards, discount=0.95)
policy_iteration.run()
pi_values = policy_iteration.V
pi_optimal_policy = policy_iteration.policy
pi_iters = policy_iteration.iter
pi_time = policy_iteration.time * 1000  # milliseconds
print("Using Policy Iteration, it took %s milliseconds and %s iterations to determine the optimal policy." %(str(pi_time), str(pi_iters)))


##### Value Iteration ####
value_iteration = mdptoolbox.mdp.ValueIteration(transitions=probabilities, reward=rewards, discount=0.95)
value_iteration.run()
value_values = value_iteration.V
value_optimal_policy = value_iteration.policy
value_iters = value_iteration.iter
value_time = value_iteration.time * 1000  # milliseconds
print("Using Value Iteration, it took %s milliseconds and %s iterations to determine the optimal policy." %(str(value_time), str(value_iters)))


##### Q-learning ####
qlearner_iteration = mdptoolbox.mdp.QLearning(transitions=probabilities, reward=rewards, discount=0.95, n_iter=100000)
qlearner_iteration.run()
qlearner_values = qlearner_iteration.V
qlearner_optimal_policy = qlearner_iteration.policy
qlearner_q_table = qlearner_iteration.Q
qlearner_time = qlearner_iteration.time * 1000 # milliseconds
qlearner_iters = qlearner_iteration.max_iter
print("Using Q-Learning, it took %s milliseconds and %s iterations to determine the optimal policy." %(str(qlearner_time), str(qlearner_iters)))

qlearner_2_start_time = timeit.default_timer()
qlearner_2_values, qlearner_2_policy, qlearner_2_episode_rewards, qlearner_2_N = q_learning(mdp=qlearner_iteration, num_states=num_states, num_actions=2, num_episodes=50, T_max=10000, epsilon=0.1, discount=0.95)
qlearner_2_end_time = timeit.default_timer()
qlearner_2_training_time = (qlearner_2_end_time - qlearner_2_start_time) * 1000 # in millieconds

print("Using Q-Learning2, it took %s milliseconds to determine the optimal policy." %(str(qlearner_2_training_time)))

##### Comparisons #####
# Comparing policy iteration, value iteration, and Q-Learner
comparison_policy_value = check_same_policy(pi_optimal_policy,value_optimal_policy)
comparison_policy_qlearner = check_same_policy(pi_optimal_policy,qlearner_optimal_policy)
comparison_value_qlearner = check_same_policy(value_optimal_policy,qlearner_optimal_policy)
print("Policy Iteration and Value Iteration Results: %s" %(comparison_policy_value))
print("Policy Iteration and QLearner Iteration Results: %s" %(comparison_policy_qlearner))
print("Value Iteration and QLearner Iteration Results: %s" %(comparison_value_qlearner))

# Comparing random iteration, policy iteration, value iteration, and Q-Learner total rewards
print("Policy Iteration Total Reward: %s" %(str(pi_values[len(pi_values)-1])))
print("Value Iteration Total Reward: %s" %(str(value_values[len(value_values)-1])))
print("QLearner Total Reward: %s" %(str(qlearner_values[len(qlearner_values)-1])))
print("QLearner2 Total Reward: %s" %(str(qlearner_2_values[len(qlearner_2_values)-1][num_states-1])))


print("you got this kiddo, don't give up!  Last assignment!")