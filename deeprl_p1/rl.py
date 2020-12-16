# coding: utf-8
from __future__ import division, absolute_import
from __future__ import print_function, unicode_literals

import numpy as np
import time

def evaluate_policy(env, gamma, policy, value_func, max_iterations=int(1e3), tol=1e-3):
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
    value_func: np.array
      The value function array
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
    iterations = 0
    v = value_func

    for i in range(max_iterations):
        v_old = v.copy()
        iterations += 1
        delta = 0
        # iterate through each state
        for s in range(env.nS):
            a = policy[s]
            expected_value = 0.0
            for prob, nextstate, reward, is_terminal in env.P[s][a]:
                if is_terminal == True:
                    expected_value +=  prob * (reward + gamma * 0)
                else:
                    expected_value +=  prob * (reward + gamma * v_old[nextstate])

            # update state value function
            v[s] = expected_value
            delta = max(delta, abs(v[s] - v_old[s]))

        # converage
        if (delta < tol):
            break

    return v, iterations


def improve_policy(env, gamma, value_function, policy):
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
    policy_stable = True
    for s in range(env.nS):
        old_action = policy[s]
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
        if old_action != policy[s]:
            policy_stable = False

    return policy_stable, policy



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
        value_func, e_iter = evaluate_policy(env, gamma, policy, value_func, max_iterations, tol)
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
    value functions (numpy array), policy (numpy array), iteration (int)
    """
    V = np.zeros(env.nS)
    policy = np.zeros(env.nS, dtype='int')

    iteration_cnt = 0
    for i in range(max_iterations):
        delta = 0
        V_old = V.copy()
        for s in range(env.nS):
            max_value = None
            for a in range(env.nA):
                expectation = 0
                for prob, nextstate, reward, is_terminal in env.P[s][a]:
                    if is_terminal:
                        expectation += prob * (reward + gamma * 0)
                    else:
                        expectation += prob * (reward + gamma * V_old[nextstate])
                #max_value = expectation if max_value is None else max(max_value, expectation)

                if max_value is None or max_value < expectation:
                    max_value = expectation
                    policy[s] = a

            V[s] = max_value
            delta = max(delta, abs(V_old[s] - V[s]))
        iteration_cnt += 1
        if delta < tol:
            break

    return V, policy, iteration_cnt

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

    return str_policy
