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


def trajectory_model_frozenlake(t, env, action=None):
    """trajectory_model_frozenlake

    A probabilistic program for the frozenlake environment trajectories using
    the softmax agent.

    $\\pi(action_0; state_0) \\propto \\exp(\\alpha Q(state_0, action_0)$

    :param t: Current time-step
    :param env: OpenAI Gym FrozenLake environment
    :param action: (optional) presampled action
    """
    env = deepcopy(env)

    # running return and discount factor
    return_, discount = 0.0, 1.0
    for tt in itt.count(t):
        if action is None:
            action = softmax_agent_model(
                tt, env, trajectory_model=trajectory_model_frozenlake
            )

        _, reward, done, _ = env.step(torch.as_tensor(action).item())

        # running return and discount factor
        return_ += discount * reward
        discount *= args.gamma

        if done:
            break

    pyro.sample(f'G_{t}', Delta(torch.as_tensor(return_)))

    return return_


def trajectory_model_mdp(t, env, action=None):
    """trajectory_model_mdp

    A probabilistic program for MDP environment trajectories using the softmax
    agent.

    $\\pi(action_0; state_0) \\propto \\exp(\\alpha Q(state_0, action_0)$

    :param t: Current time-step
    :param env: OpenAI Gym FrozenLake environment
    :param action: (optional) presampled action
    """
    env = deepcopy(env)

    # running return and discount factor
    return_, discount = 0.0, 1.0
    for tt in itt.count(t):
        if action is None:
            action = softmax_agent_model(
                tt, env, trajectory_model=trajectory_model_mdp
            )

        _, reward, done, _ = env.step(action)

        # running return and discount factor
        return_ += discount * reward
        discount *= args.gamma

        if done:
            break

    return_ = pyro.sample(f'G_{t}', Delta(return_))

    return return_


def softmax_agent_model(t, env, *, trajectory_model):
    """softmax_agent_model

    Softmax agent model;  Performs inference to estimate $Q^\pi(s, a)$, then
    uses pyro.factor to modify the trace log-likelihood.

    :param t: time-step
    :param env: OpenAI Gym environment
    :param trajectory_model: trajectory probabilistic program
    """
    action_probs = torch.ones(env.action_space.n)
    action = pyro.sample(f'A_{t}', Categorical(action_probs))

    inference = Importance(trajectory_model, num_samples=args.num_samples)
    posterior = inference.run(t, env, action)
    Q = EmpiricalMarginal(posterior, f'G_{t}').mean

    pyro.factor(f'softmax_{t}', args.alpha * Q)

    return action


def policy(t, env, *, trajectory_model, log=False):
    """policy

    :param t: time-step
    :param env: OpenAI Gym environment
    :param trajectory_model: trajectory probabilistic program
    :param log: boolean; if True, print log info
    """
    inference = Importance(softmax_agent_model, num_samples=args.num_samples)
    posterior = inference.run(t, env, trajectory_model=trajectory_model)
    marginal = EmpiricalMarginal(posterior, f'A_{t}')

    if log:
        samples = marginal.sample((args.num_samples,))
        counts = Counter(samples.tolist())
        hist = [counts[i] / args.num_samples for i in range(env.action_space.n)]
        print('policy:')
        print(tabulate([hist], headers=env.actions, tablefmt='fancy_grid'))

    return marginal.sample()


def main():
    if args.mdp == 'frozenlake':
        env = gym.make('FrozenLake-v0', is_slippery=False)
        env = FrozenLakeWrapper(env)

        trajectory_model = trajectory_model_frozenlake

        # makes sure integer action is sent to frozenlake environment
        def action_cast(action):
            return action.item()

    else:
        env = make_mdp(args.mdp, episodic=True)
        env = TimeLimit(env, 10)

        trajectory_model = trajectory_model_mdp

        # makes sure tensor action is sent to MDP environment
        def action_cast(action):
            return action

    env.reset()
    for t in itt.count():
        print('---')
        print(f't: {t}')
        print('state:')
        env.render()

        action = policy(t, env, trajectory_model=trajectory_model, log=True)
        _, reward, done, _ = env.step(action_cast(action))
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
