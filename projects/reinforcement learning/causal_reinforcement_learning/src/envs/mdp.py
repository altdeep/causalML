import types

from gym_pyro import PyroMDP

from . import renders


def make_mdp(path, *args, **kwargs):
    """make_mdp

    Creates a PyroMDP instance based on the given MDP file, injecting the
    respective custom renderer if found in `renders.py`.

    :param path: path to MDP file
    :param *args: arguments to PyroMDP
    :param **kwargs: keyword arguments to PyroMDP
    """
    with open(path) as f:
        env = PyroMDP(f.read(), *args, **kwargs)  # pylint: disable=missing-kwoa

    basename = path.split('/')[-1]
    try:
        render = renders.get_render(basename)
    except KeyError:
        pass
    else:
        env.render = types.MethodType(render, env)

    return env
