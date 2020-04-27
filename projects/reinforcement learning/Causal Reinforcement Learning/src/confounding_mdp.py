#!/usr/bin/env python

import argparse
import itertools as itt
from copy import deepcopy

import gym
import pandas as pd
import pyro
import torch
from gym.wrappers import TimeLimit
from pyro.distributions import Delta
from pyro.infer import EmpiricalMarginal, Importance

import agent_models
from envs import make_cmdp


def trajectory_model(env, *, agent_model):
    """trajectory_model

    A probabilistic program which simulates a trajectory by sampling random
    actions.  The sample return can be used to affect the trace likelihood such
    that the agent policy becomes

    $\\pi(action_0; state_0) \\propto \\exp(\\alpha return_0)$

    :param env: OpenAI Gym environment
    :param agent_model: agent's probabilistic program
    """
    env = deepcopy(env)

    # initializing the running return and discount factor
    return_, discount = 0.0, 1.0

    # with keep_state=True only the time-step used to name sites is being reset
    state, confounder = env.reset(keep_state=True)
    for t in itt.count():
        action = agent_model(f'A_{t}', env, (state, confounder))
        state, reward, done, _ = env.step(action)

        # updating the running return and discount factor
        return_ += discount * reward
        discount *= args.gamma

        if done:
            break

    pyro.sample('G', Delta(return_))

    return return_


def infer_Q(env, action, infer_type='intervention', *, agent_model):
    """infer_Q

    Infer Q(state, action) via pyro's importance sampling, via conditioning or
    intervention.

    :param env: OpenAI Gym environment
    :param action: integer action
    :param infer_type: type of inference; none, condition, or intervention
    :param agent_model: agent's probabilistic program
    """
    if infer_type not in ('intervention', 'condition', 'none'):
        raise ValueError('Invalid inference type {infer_type}')

    if infer_type == 'intervention':
        model = pyro.do(trajectory_model, {'A_0': torch.tensor(action)})
    elif infer_type == 'condition':
        model = pyro.condition(trajectory_model, {'A_0': torch.tensor(action)})
    else:  # infer_type == 'none'
        model = trajectory_model

    posterior = Importance(model, num_samples=args.num_samples).run(
        env, agent_model=agent_model
    )
    return EmpiricalMarginal(posterior, 'G').mean


def main():
    env = make_cmdp(args.cmdp, episodic=True)
    env = TimeLimit(env, 10)

    agent_model_name = args.cmdp.split('/')[-1]
    agent_model = agent_models.get_agent_model(agent_model_name)

    values_df_index = 'E[G]', 'E[G | A=a]', 'E[G | do(A=a)]'
    values_df_columns = env.model.actions

    _, state = env.reset()
    for t in itt.count():
        print()
        print(f't: {t}')
        env.render()

        Qs_none = [
            infer_Q(env, action, 'none', agent_model=agent_model).item()
            for action in range(env.action_space.n)
        ]
        Qs_condition = [
            infer_Q(env, action, 'condition', agent_model=agent_model).item()
            for action in range(env.action_space.n)
        ]
        Qs_intervention = [
            infer_Q(env, action, 'intervention', agent_model=agent_model).item()
            for action in range(env.action_space.n)
        ]

        values_df = pd.DataFrame(
            [Qs_none, Qs_condition, Qs_intervention],
            values_df_index,
            values_df_columns,
        )
        print(values_df)

        action = torch.tensor(Qs_intervention).argmax()
        state, _, done, _ = env.step(action)

        if done:
            print()
            print(f'final state: {state}')
            print(f'Episode finished after {t+1} timesteps')
            break

    env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('cmdp', help='CMDP file')
    parser.add_argument(
        '--gamma', type=float, default=0.95, help='discount factor'
    )
    parser.add_argument(
        '--num-samples',
        type=int,
        default=10,
        help='number of samples to be used for importance sampling',
    )
    args = parser.parse_args()

    print(f'args: {args}')
    main()
