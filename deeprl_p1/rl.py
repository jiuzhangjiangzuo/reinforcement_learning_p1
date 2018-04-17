# coding: utf-8
from __future__ import division, absolute_import
from __future__ import print_function, unicode_literals

import numpy as np
import time

def evaluate_policy(env, gamma, policy, max_iterations=int(1e3), tol=1e-3):
    """Evaluate the value of a policy.

    See page 87 (pg 105 pdf) of the Sutton and Barto Second Edition
    book.

    http://webdocs.cs.ualberta.ca/~sutton/book/bookdraft2016sep.pdf

    Parameters
    ----------
    env: gym.core.Environment
      The environment to compute value iteration for. Must have nS,
      nA, and P as attributes.
    gamma: float
      Discount factor, must be in range [0, 1)
    policy: np.array
      The policy to evaluate. Maps states to actions.
    max_iterations: int
      The maximum number of iterations to run before stopping.
    tol: float
      Determines when value function has converged.

    Returns
    -------
    np.ndarray, int
      The value for the given policy and the number of iterations till
      the value function converged.
    """
    v = np.zeros(env.nS)
    iterations = 0

    eval_converge = False
    while not eval_converge:
        iterations += 1
        eval_converge = True
        # iterate through each state
        for s in range(env.nS):
            a = policy[s]
            expected_value = 0.0
            for prob, nextstate, reward, is_terminal in env.P[s][a]:
                if is_terminal == True:
                    expected_value +=  prob * (reward + gamma * 0)
                else:
                    expected_value +=  prob * (reward + gamma * v[nextstate])

            # update state value function
            old_v = v[s]
            v[s] = expected_value
            if (abs(v[s] - old_v) > tol):
                eval_converge = False

    return v, iterations


def value_function_to_policy(env, gamma, value_function):
    """Output action numbers for each state in value_function.

    Parameters
    ----------
    env: gym.core.Environment
      Environment to compute policy for. Must have nS, nA, and P as
      attributes.
    gamma: float
      Discount factor. Number in range [0, 1)
    value_function: np.ndarray
      Value of each state.

    Returns
    -------
    np.ndarray
      An array of integers. Each integer is the optimal action to take
      in that state according to the environment dynamics and the
      given value function.
    """
    policy = np.zeros(env.nS, dtype='int')
    for s in range(env.nS):
        max_value = None
        # if take this action, calculate the expected reward
        for a in range(env.nA):
            expected_value = 0.0
            for prob, nextstate, reward, is_terminal in env.P[s][a]:
                if is_terminal:
                    expected_value +=  prob * (reward + gamma * 0)
                else:
                    expected_value +=  prob * (reward + gamma * value_function[nextstate])
            # Record the maximum value and corresponding action
            if max_value is None or max_value < expected_value:
                max_value = expected_value
                policy[s] = a

    return policy


def improve_policy(env, gamma, value_func, policy):
    """Given a policy and value function improve the policy.

    See page 87 (pg 105 pdf) of the Sutton and Barto Second Edition
    book.

    http://webdocs.cs.ualberta.ca/~sutton/book/bookdraft2016sep.pdf

        Parameters
    ----------
    env: gym.core.Environment
      The environment to compute value iteration for. Must have nS,
      nA, and P as attributes.
    gamma: float
      Discount factor, must be in range [0, 1)
    value_func: np.ndarray
      Value function for the given policy.
    policy: dict or np.array
      The policy to improve. Maps states to actions.

    Returns
    -------
    bool, np.ndarray
      Returns true if policy changed. Also returns the new policy.
    """
    old_policy = policy
    new_policy = value_function_to_policy(env, gamma, value_func)

    policy_stable = True
    for s in range(env.nS):
        if old_policy[s] != new_policy[s]:
            policy_stable = False
    return policy_stable, new_policy


def policy_iteration(env, gamma, max_iterations=int(1e3), tol=1e-3):
    """Runs policy iteration.

    See page 87 (pg 105 pdf) of the Sutton and Barto Second Edition
    book.

    http://webdocs.cs.ualberta.ca/~sutton/book/bookdraft2016sep.pdf

    You should use the improve_policy and evaluate_policy methods to
    implement this method.

    Parameters
    ----------
    env: gym.core.Environment
      The environment to compute value iteration for. Must have nS,
      nA, and P as attributes.
    gamma: float
      Discount factor, must be in range [0, 1)
    max_iterations: int
      The maximum number of iterations to run before stopping.
    tol: float
      Determines when value function has converged.

    Returns
    -------
    (np.ndarray, np.ndarray, int, int)
       Returns optimal policy, value function, number of policy
       improvement iterations, and number of value iterations.
    """
    policy = np.zeros(env.nS, dtype='int')
    value_func = np.zeros(env.nS)
    improve_iteration = 0
    evalue_iteration = 0
    policy_stable = False

    for i in range(max_iterations):
        value_func, e_iter = evaluate_policy(env, gamma, policy, max_iterations, tol)
        policy_stable, policy = improve_policy(env, gamma, value_func, policy)
        improve_iteration += 1
        evalue_iteration += e_iter
        if policy_stable:
            break
    return policy, value_func, improve_iteration, evalue_iteration


def value_iteration(env, gamma, max_iterations=int(1e3), tol=1e-3):
    """Runs value iteration for a given gamma and environment.

    See page 90 (pg 108 pdf) of the Sutton and Barto Second Edition
    book.

    http://webdocs.cs.ualberta.ca/~sutton/book/bookdraft2016sep.pdf

    Parameters
    ----------
    env: gym.core.Environment
      The environment to compute value iteration for. Must have nS,
      nA, and P as attributes.
    gamma: float
      Discount factor, must be in range [0, 1)
    max_iterations: int
      The maximum number of iterations to run before stopping.
    tol: float
      Determines when value function has converged.

    Returns
    -------
    np.ndarray, iteration
      The value function and the number of iterations it took to converge.
    """
    V = np.zeros(env.nS)
    iteration_cnt = 0
    for i in range(max_iterations):
        delta = 0
        for s in range(env.nS):
            v = V[s]
            max_value = None
            for a in range(env.nA):
                expectation = 0
                for prob, nextstate, reward, is_terminal in env.P[s][a]:
                    if is_terminal:
                        expectation += prob * (reward + gamma * 0)
                    else:
                        expectation += prob * (reward + gamma * V[nextstate])
                max_value = expectation if max_value is None else max(max_value, expectation)
            V[s] = max_value
            delta = max(delta, abs(v - V[s]))
        iteration_cnt += 1
        if delta < tol:
            break

    V[env.nS-1] = 0
    return V, iteration_cnt


def print_policy(policy, action_names):
    """Print the policy in human-readable format.

    Parameters
    ----------
    policy: np.ndarray
      Array of state to action number mappings
    action_names: dict
      Mapping of action numbers to characters representing the action.
    """
    str_policy = policy.astype('str')
    for action_num, action_name in action_names.items():
        np.place(str_policy, policy == action_num, action_name)

    print(str_policy)
    return str_policy
