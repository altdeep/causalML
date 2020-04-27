import types

from gym_pyro import PyroPOMDP

from . import renders


def make_pomdp(path, *args, **kwargs):
    """make_pomdp

    Creates a PyroPOMDP instance based on the given POMDP file, injecting the
    respective custom renderer if found in `renders.py`.

    :param path: path to POMDP file
    :param *args: arguments to PyroPOMDP
    :param **kwargs: keyword arguments to PyroPOMDP
    """
    with open(path) as f:
        env = PyroPOMDP(  # pylint: disable=missing-kwoa
            f.read(), *args, **kwargs
        )

    basename = path.split('/')[-1]
    try:
        render = renders.get_render(basename)
    except KeyError:
        pass
    else:
        env.render = types.MethodType(render, env)

    return env
