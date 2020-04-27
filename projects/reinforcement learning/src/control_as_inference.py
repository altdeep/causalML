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

import agent_models
from envs import FrozenLakeWrapper, make_mdp


def trajectory_model_frozenlake(env, *, agent_model, factor_G=False):
    """trajectory_model_frozenlake

    A probabilistic program for the frozenlake environment trajectories.  The
    sample return can be used to affect the trace likelihood.

    :param env: OpenAI Gym FrozenLake environment
    :param agent_model: agent's probabilistic program
    :param factor_G: boolean; if True then apply $\\alpha G$ likelihood factor
    """
    env = deepcopy(env)

    # running return and discount factor
    return_, discount = 0.0, 1.0
    for t in itt.count():
        action = agent_model(f'A_{t}', env, env.s)
        _, reward, done, _ = env.step(action.item())

        # running return and discount factor
        return_ += discount * reward
        discount *= args.gamma

        if done:
            break

    pyro.sample('G', Delta(torch.as_tensor(return_)))

    if factor_G:
        pyro.factor('factor_G', args.alpha * return_)

    return return_


def trajectory_model_mdp(env, *, agent_model, factor_G=False):
    """trajectory_model_mdp

    A probabilistic program for MDP environment trajectories.  The sample return
    can be used to affect the trace likelihood.

    :param env: OpenAI Gym environment
    :param agent_model: agent's probabilistic program
    :param factor_G: boolean; if True then apply $\\alpha G$ likelihood factor
    """
    env = deepcopy(env)

    # running return and discount factor
    return_, discount = 0.0, 1.0

    # with keep_state=True only the time-step used to name sites is being reset
    state = env.reset(keep_state=True)
    for t in itt.count():
        action = agent_model(f'A_{t}', env, state)
        state, reward, done, _ = env.step(action)

        # running return and discount factor
        return_ += discount * reward
        discount *= args.gamma

        if done:
            break

    pyro.sample('G', Delta(return_))

    if factor_G:
        pyro.factor('factor_G', args.alpha * return_)

    return return_


def policy_control_as_inference_like(
    env, *, trajectory_model, agent_model, log=False
):
    """policy_control_as_inference_like

    Implements a control-as-inference-like policy which "maximizes"
    $\\Pr(A_0 \\mid S_0, high G)$.

    Not actually standard CaI, because we don't really condition on G;  rather,
    we use $\\alpha G$ as a likelihood factor on sample traces.

    :param env: OpenAI Gym environment
    :param trajectory_model: trajectory probabilistic program
    :param agent_model: agent's probabilistic program
    :param log: boolean; if True, print log info
    """
    inference = Importance(trajectory_model, num_samples=args.num_samples)
    posterior = inference.run(env, agent_model=agent_model, factor_G=True)
    marginal = EmpiricalMarginal(posterior, 'A_0')

    if log:
        samples = marginal.sample((args.num_samples,))
        counts = Counter(samples.tolist())
        hist = [counts[i] / args.num_samples for i in range(env.action_space.n)]
        print('policy:')
        print(tabulate([hist], headers=env.actions, tablefmt='fancy_grid'))

    return marginal.sample()


def infer_Q(env, action, *, trajectory_model, agent_model, log=False):
    """infer_Q

    Infer Q(state, action) via pyro's importance sampling.

    :param env: OpenAI Gym environment
    :param action: integer action
    :param trajectory_model: trajectory probabilistic program
    :param agent_model: agent's probabilistic program
    :param log: boolean; if True, print log info
    """
    posterior = Importance(
        pyro.do(trajectory_model, {'A_0': torch.as_tensor(action)}),
        num_samples=args.num_samples,
    ).run(env, agent_model=agent_model)
    Q = EmpiricalMarginal(posterior, 'G').mean

    if log:
        print(f'Q({env.actions[action]}) = {Q.item()}')

    return Q


def softmax_like(env, *, trajectory_model, agent_model, log=False):
    """softmax_like

    :param env: OpenAI Gym environment
    :param trajectory_model: trajectory probabilistic program
    :param agent_model: agent's probabilistic program
    :param log: boolean; if True, print log info
    """

    Qs = torch.as_tensor(
        [
            infer_Q(
                env,
                action,
                trajectory_model=trajectory_model,
                agent_model=agent_model,
                log=log,
            )
            for action in range(env.action_space.n)
        ]
    )
    action_logits = args.alpha * Qs
    action_dist = Categorical(logits=action_logits)

    if log:
        print('policy:')
        print(
            tabulate(
                [action_dist.probs.tolist()],
                headers=env.actions,
                tablefmt='fancy_grid',
            )
        )

    return action_dist.sample()


def main():
    assert args.policy in ('control-as-inference-like', 'softmax-like')

    if args.policy == 'control-as-inference-like':
        policy = policy_control_as_inference_like
    elif args.policy == 'softmax-like':
        policy = softmax_like

    if args.mdp == 'frozenlake':
        env = gym.make('FrozenLake-v0', is_slippery=False)
        env = FrozenLakeWrapper(env)

        trajectory_model = trajectory_model_frozenlake
        agent_model = agent_models.get_agent_model('FrozenLake-v0')

        # makes sure integer action is sent to frozenlake environment
        def action_cast(action):
            return action.item()

    else:
        env = make_mdp(args.mdp, episodic=True)
        env = TimeLimit(env, 100)

        trajectory_model = trajectory_model_mdp
        agent_model = agent_models.get_agent_model(args.mdp)

        # makes sure tensor action is sent to MDP environment
        def action_cast(action):
            return action

    env.reset()
    for t in itt.count():
        print('---')
        print(f't: {t}')
        print('state:')
        env.render()

        action = policy(
            env,
            trajectory_model=trajectory_model,
            agent_model=agent_model,
            log=True,
        )
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
    parser.add_argument('mdp', help='`frozenlake` string or path to MDP file')
    parser.add_argument(
        '--policy',
        choices=['control-as-inference-like', 'softmax-like'],
        default='control-as-inference-like',
        help='Choose one of two control strategies',
    )
    parser.add_argument('--alpha', type=float, default=100.0)
    parser.add_argument('--gamma', type=float, default=0.95)
    parser.add_argument('--num-samples', type=int, default=2_000)
    args = parser.parse_args()

    print(f'args: {args}')
    main()
