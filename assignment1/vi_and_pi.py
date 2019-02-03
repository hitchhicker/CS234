### MDP Value Iteration and Policy Iteration
import random

import numpy as np
import gym
import time
from lake_envs import *

np.set_printoptions(precision=3)

"""
For policy_evaluation, policy_improvement, policy_iteration and value_iteration,
the parameters P, nS, nA, gamma are defined as follows:

	P: nested dictionary
		From gym.core.Environment
		For each pair of states in [1, nS] and actions in [1, nA], P[state][action] is a
		tuple of the form (probability, nextstate, reward, terminal) where
			- probability: float
				the probability of transitioning from "state" to "nextstate" with "action"
			- nextstate: int
				denotes the state we transition to (in range [0, nS - 1])
			- reward: int
				either 0 or 1, the reward for transitioning from "state" to
				"nextstate" with "action"
			- terminal: bool
			  True when "nextstate" is a terminal state (hole or goal), False otherwise
	nS: int
		number of states in the environment
	nA: int
		number of actions in the environment
	gamma: float
		Discount factor. Number in range [0, 1)
"""


def is_convergent(V, new_V, tol):
    return np.all(np.abs(V - new_V) < tol)


def policy_evaluation(P, nS, nA, policy, gamma=0.9, tol=1e-3):
    """Evaluate the value function from a given policy.

    Parameters
    ----------
    P: dictionary
        It is from gym.core.Environment
        P[state][action] is tuples with (probability, nextstate, reward, terminal)
    nS: int
        number of states
    nA: int
        number of actions
    gamma: float
        Discount factor. Number in range [0, 1)
    policy: np.array
        The policy to evaluate. Maps states to actions.
    max_iteration: int
        The maximum number of iterations to run before stopping. Feel free to change it.
    tol: float
        Determines when value function has converged.
    Returns
    -------
    value function: np.ndarray
        The value function from the given policy.
    """
    ############################
    # YOUR IMPLEMENTATION HERE #
    ############################
    V = np.zeros(nS)
    new_V = V.copy()
    while True:
        V = new_V.copy()
        new_V = np.zeros(nS, dtype=float)
        for state in range(nS):
            for probability, nextstate, reward, terminal in P[state][policy[state]]:
                new_V[state] += probability * (reward + gamma * V[nextstate])
        if is_convergent(V, new_V, tol):
            break
    return new_V


def policy_improvement(P, nS, nA, value_from_policy, policy, gamma=0.9):
    """Given the value function from policy improve the policy.

    Parameters
    ----------
    P, nS, nA, gamma:
        defined at beginning of file
    value_from_policy: np.ndarray
        The value calculated from the policy
    policy: np.array
        The previous policy.

    Returns
    -------
    new_policy: np.ndarray[nS]
        An array of integers. Each integer is the optimal action to take
        in that state according to the environment dynamics and the
        given value function.
    """

    ############################
    # YOUR IMPLEMENTATION HERE #
    q_function = np.zeros([nS, nA])
    for state in range(nS):
        for action in range(nA):
            for probability, nextstate, reward, terminal in P[state][action]:
                q_function[state][action] += (probability * (reward + gamma * value_from_policy[nextstate]))
    new_policy = np.argmax(q_function, axis=1)
    ############################
    return new_policy


def policy_iteration(P, nS, nA, gamma=0.9, tol=10e-3):
    """Runs policy iteration.

    You should call the policy_evaluation() and policy_improvement() methods to
    implement this method.

    Parameters
    ----------
    P, nS, nA, gamma:
        defined at beginning of file
    tol: float
        tol parameter used in policy_evaluation()
    Returns:
    ----------
    value_function: np.ndarray[nS]
    policy: np.ndarray[nS]
    """
    value_function = np.zeros(nS)
    policy = np.zeros(nS, dtype=int)
    ############################
    # YOUR IMPLEMENTATION HERE #
    policy_stable = False
    while not policy_stable:
        value_function = policy_evaluation(P, nS, nA, policy, gamma, tol)
        new_policy = policy_improvement(P, nS, nA, value_function, policy, gamma)
        if np.array_equal(policy, new_policy):
            policy_stable = True
        else:
            policy = new_policy.copy()
    ############################
    return value_function, policy


def is_policy_equal(p, new_p):
    return np.array_equal(p, new_p)


def value_iteration(P, nS, nA, gamma=0.9, tol=1e-3):
    """
    Learn value function and policy by using value iteration method for a given
    gamma and environment.

    Parameters:
    ----------
    P, nS, nA, gamma:
        defined at beginning of file
    tol: float
        Terminate value iteration when
            max |value_function(s) - prev_value_function(s)| < tol
    Returns:
    ----------
    value_function: np.ndarray[nS]
    policy: np.ndarray[nS]
    """

    value_function = np.zeros(nS)
    policy = np.zeros(nS, dtype=int)
    ############################
    # YOUR IMPLEMENTATION HERE #
    new_V = value_function.copy()
    while True:
        for state in range(nS):
            for action in range(nA):
                q = 0
                for probability, nextstate, reward, terminal in P[state][action]:
                    q += probability * (reward + gamma * value_function[nextstate])
                if q > new_V[state]:
                    new_V[state] = q
        if is_convergent(value_function, new_V, tol):
            break
        else:
            value_function = new_V.copy()

    policy = policy_improvement(P, nS, nA, new_V, policy, gamma)
    ############################
    return value_function, policy


def render_single(env, policy, max_steps=100):
    """
      This function does not need to be modified
      Renders policy once on environment. Watch your agent play!

      Parameters
      ----------
      env: gym.core.Environment
        Environment to play on. Must have nS, nA, and P as
        attributes.
      Policy: np.array of shape [env.nS]
        The action to take at a given state
    """

    episode_reward = 0
    ob = env.reset()
    for t in range(max_steps):
        env.render()
        time.sleep(0.25)
        a = policy[ob]
        ob, rew, done, _ = env.step(a)
        episode_reward += rew
        if done:
            break
    env.render();
    if not done:
        print("The agent didn't reach a terminal state in {} steps.".format(max_steps))
    else:
        print("Episode reward: %f" % episode_reward)


def run_in_env(env):
    print("Environment:", env)
    env = gym.make(env)
    print("\n" + "-" * 25 + "\nBeginning Policy Iteration\n" + "-" * 25)
    start_time = time.time()
    V_pi, p_pi = policy_iteration(env.P, env.nS, env.nA, gamma=0.9, tol=1e-3)
    print("  Policy iteration took", time.time() - start_time, "seconds")
    print('  Optimal Value Function: %r' % V_pi)
    print('  Optimal Policy:         %r' % p_pi)
    # render_single(env, p_pi, 100)

    print("\n" + "-" * 25 + "\nBeginning Value Iteration\n" + "-" * 25)
    start_time = time.time()
    V_vi, p_vi = value_iteration(env.P, env.nS, env.nA, gamma=0.9, tol=1e-3)
    print("  Value iteration took", time.time() - start_time, "seconds")
    print('  Optimal Value Function: %r' % V_vi)
    print('  Optimal Policy:         %r' % p_vi)
    # render_single(env, p_vi, 100)


# Edit below to run policy and value iteration on different environments and
# visualize the resulting policies in action!
# You may change the parameters in the functions below
if __name__ == "__main__":
    # comment/uncomment these lines to switch between deterministic/stochastic environments
    # env = gym.make("Deterministic-4x4-FrozenLake-v0")
    run_in_env(env="Deterministic-4x4-FrozenLake-v0")
    run_in_env(env="Stochastic-4x4-FrozenLake-v0")
