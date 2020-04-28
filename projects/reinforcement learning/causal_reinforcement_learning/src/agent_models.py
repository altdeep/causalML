import pyro
import torch
from pyro.distributions import Categorical

__registry = {}


def __register(name):
    def decorator(func):
        __registry[name] = func
        return func

    return decorator


def get_agent_model(name):
    try:
        return __registry[name]
    except KeyError:
        return uniform


def uniform(site, env, state):
    action_probs = torch.ones(env.action_space.n)
    return pyro.sample(site, Categorical(action_probs))


@__register('circle.cmdp')
def circle(site, env, state):
    state, confounder = state

    policy = torch.tensor(
        [
            [
                [0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0, 1.0],
                [0.0, 1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
            ],
            [
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [1.0, 1.0, 1.0, 1.0],
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0, 0.0],
            ],
        ]
    )

    action_probs = policy[confounder, state]
    return pyro.sample(site, Categorical(action_probs))
