import contextlib
import io
import sys

from gym.utils import colorize

__registry = {}


def get_render(name):
    """get_render

    Returns a custom-made renderer for the environment called `name`.  Used to
    inject better graphical representations of the environment state.

    :param name: name of render method
    """
    return __registry[name]


def __register(name):
    def decorator(func):
        __registry[name] = func
        return func

    return decorator


@__register('gridworld.mdp')
def render_gridworld(  # pylint: disable=inconsistent-return-statements
    self, mode='human'
):
    if mode not in ('human', 'ansi'):
        raise ValueError('Only `human` and `ansi` modes are supported')

    # stream where to send the string representation of the env
    outfile = sys.stdout if mode == 'human' else io.StringIO()

    if self.action_prev is not None:
        ai = self.action_prev.item()
        print(f'action: {self.model.actions[ai]}', file=outfile)

    if self.state < 5:
        i = self.state.item() // 4
        j = self.state.item() % 4
    else:
        i = (self.state.item() + 1) // 4
        j = (self.state.item() + 1) % 4

    desc = [['.', '.', '.', '+'], ['.', ' ', '.', '-'], ['.', '.', '.', '.']]
    desc[i][j] = colorize(desc[i][j], 'red', highlight=True)

    desc = '\n'.join(''.join(line) for line in desc)
    print(desc, file=outfile)

    if mode == 'ansi':
        with contextlib.closing(outfile):
            return outfile.getvalue()
