import types

from gym_pyro import PyroCMDP

from . import renders


def make_cmdp(path, *args, **kwargs):
    """make_cmdp

    Creates a PyroCMDP instance based on the given CMDP file, injecting the
    respective custom renderer if found in `renders.py`.

    :param path: path to CMDP file
    :param *args: arguments to PyroCMDP
    :param **kwargs: keyword arguments to PyroCMDP
    """
    with open(path) as f:
        env = PyroCMDP(  # pylint: disable=missing-kwoa
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
