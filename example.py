#!/usr/bin/env python
# coding: utf-8

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from builtins import input

import deeprl_p1.lake_envs as lake_env
from deeprl_p1.rl import *
import gym
import time



def run_random_policy(env):
    """Run a random policy for the given environment.

    Logs the total reward and the number of steps until the terminal
    state was reached.

    Parameters
    ----------
    env: gym.envs.Environment
      Instance of an OpenAI gym.

    Returns
    -------
    (float, int)
      First number is the total undiscounted reward received. The
      second number is the total number of actions taken before the
      episode finished.
    """
    initial_state = env.reset()
    env.render()
    time.sleep(1)  # just pauses so you can see the output

    total_reward = 0
    num_steps = 0
    while True:
        nextstate, reward, is_terminal, debug_info = env.step(
            env.action_space.sample())
        env.render()
        print('reward:{}'.format(reward))
        total_reward += reward
        num_steps += 1

        if is_terminal:
            break

        time.sleep(1)
    return total_reward, num_steps


def run_optimal_policy(env, opp, gamma=0.9):
    """Run an optimal policy for the given environment.

    Logs the total reward and the number of steps until the terminal
    state was reached.

    Parameters
    ----------
    env: gym.envs.Environment
      Instance of an OpenAI gym.

    opp: np.ndarray
      Optimal Policy

    Returns
    -------
    (float, int)
      First number is the total undiscounted reward received. The
      second number is the total number of actions taken before the
      episode finished.
    """
    initial_state = env.reset()
    env.render()
    #time.sleep(1)  # just pauses so you can see the output

    total_reward = 0
    num_steps = 0
    s = initial_state


    while True:
        next_act = opp[s]
        nextstate, reward, is_terminal, debug_info = env.step(next_act)
        env.render()

        total_reward += pow(gamma, num_steps) * reward
        num_steps += 1

        if is_terminal:
            break

        #time.sleep(1)

        s = nextstate

    return total_reward, num_steps


def print_env_info(env):
    print('Environment has %d states and %d actions.' % (env.nS, env.nA))


def print_model_info(env, state, action):
    transition_table_row = env.P[state][action]
    print(
        ('According to transition function, '
         'taking action %s(%d) in state %d leads to'
         ' %d possible outcomes') % (lake_env.action_names[action],
                                     action, state, len(transition_table_row)))
    for prob, nextstate, reward, is_terminal in transition_table_row:
        state_type = 'terminal' if is_terminal else 'non-terminal'
        print(
            '\tTransitioning to %s state %d with probability %f and reward %f'
            % (state_type, nextstate, prob, reward))


def plot_policy(policy, length):
    grid = print_policy(policy, {lake_env.DOWN:'D', lake_env.LEFT:'L', lake_env.UP:'U', lake_env.RIGHT:'R'})
    for i in range(length):
        line = ""
        for j in range(length):
            line += grid[i*length+j]
        print(line)


def cal_value_iteration(env, gamma=0.9):
    value_func, policy, iteration_cnt = value_iteration(env, gamma=gamma)
    print("Value Iternation:%d" % iteration_cnt)
    print("Show me the policy:")
    plot_policy(policy, 4)
    print("")
    run_optimal_policy(env, policy)


def cal_policy_iteration(env, gamma=0.9):
    policy, value_func, improve_iteration, evalue_iteration = policy_iteration(env, gamma=gamma)
    print("Policy iteration:%d" % improve_iteration)
    print("Show me the policy:")
    plot_policy(policy, 4)
    print("")
    run_optimal_policy(env, policy)

def compare_performance(env, gamma=0.9):
    value_func, policy, iteration_cnt = value_iteration(env, gamma=gamma)
    print("Value Iternation:%d" % iteration_cnt)
    policy, value_func, improve_iteration, evalue_iteration = policy_iteration(env, gamma=gamma)
    print("Policy iteration:%d" % improve_iteration)

def main():
    env = gym.make('Deterministic-4x4-FrozenLake-v0')
    #env = gym.make('Stochastic-4x4-FrozenLake-v0')

    #cal_value_iteration(env)
    #cal_policy_iteration(env)
    compare_performance(env)


if __name__ == '__main__':
    main()
