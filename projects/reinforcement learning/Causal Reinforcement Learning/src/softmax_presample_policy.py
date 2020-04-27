#!/usr/bin/env python

import argparse
import itertools as itt
from collections import Counter
from copy import deepcopy

import gym
import pyro
import torch
from gym.wrappers import TimeLimit
from pyro.distributions import Categorical, Delta
from pyro.infer import EmpiricalMarginal, Importance
from tabulate import tabulate

from envs import FrozenLakeWrapper, make_mdp


def trajectory_model(env, policy):
    """trajectory_model

    A probabilistic program for MDP environment trajectories using a presampled
    policy.

    :param env: OpenAI Gym FrozenLake environment
    :param policy: predetermined policy function
    """
    env = deepcopy(env)

    # running return and discount factor
    return_, discount = 0.0, 1.0
    for _ in itt.count():
        action = policy(env.state)
        _, reward, done, _ = env.step(action)

        # running return and discount factor
        return_ += discount * reward
        discount *= args.gamma

        if done:
            break

    return_ = pyro.sample(f'G', Delta(return_))

    return return_


def softmax_agent_model(env):
    """softmax_agent_model

    Softmax agent model;  Performs inference to estimate $Q^\pi(s, a)$, then
    uses pyro.factor to modify the trace log-likelihood.

    :param env: OpenAI Gym environment
    """
    policy_probs = torch.ones(env.state_space.n, env.action_space.n)
    policy_vector = pyro.sample('policy_vector', Categorical(policy_probs))

    inference = Importance(trajectory_model, num_samples=args.num_samples)
    posterior = inference.run(env, lambda state: policy_vector[state])
    Q = EmpiricalMarginal(posterior, 'G').mean

    pyro.factor('factor_Q', args.alpha * Q)

    return policy_vector


def policy(env, log=False):
    """policy

    :param env: OpenAI Gym environment
    :param log: boolean; if True, print log info
    """
    inference = Importance(softmax_agent_model, num_samples=args.num_samples)
    posterior = inference.run(env)
    marginal = EmpiricalMarginal(posterior, 'policy_vector')

    if log:
        policy_samples = marginal.sample((args.num_samples,))
        action_samples = policy_samples[:, env.state]
        counts = Counter(action_samples.tolist())
        hist = [counts[i] / args.num_samples for i in range(env.action_space.n)]
        print('policy:')
        print(tabulate([hist], headers=env.actions, tablefmt='fancy_grid'))

    policy_vector = marginal.sample()
    return policy_vector[env.state]


def main():
    env = make_mdp(args.mdp, episodic=True)
    env = TimeLimit(env, 10)

    env.reset()
    for t in itt.count():
        print('---')
        print(f't: {t}')
        print('state:')
        env.render()

        action = policy(env, log=True)
        _, reward, done, _ = env.step(action)
        print(f'reward: {reward}')

        if done:
            print('final state:')
            env.render()
            print(f'Episode finished after {t+1} timesteps')
            break

    env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mdp', help='path to MDP file')
    parser.add_argument('--alpha', type=float, default=5_000.0)
    parser.add_argument('--gamma', type=float, default=0.95)
    parser.add_argument('--num-samples', type=int, default=20)
    args = parser.parse_args()

    print(f'args: {args}')
    main()
