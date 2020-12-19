from typing import Sequence, Union

import numpy as np
from pyro.distributions import TorchDistribution


def _is_single(idx):
    return np.ndim(idx) == 0


class MultivariateDistribution(TorchDistribution):
    def __init__(self, *args, var_names=None, **kwargs):
        super().__init__(*args, **kwargs)  # Necessary to allow multiple inheritance
        self.variable_names = None
        self._var_indices = None
        self.rename(var_names)

    @property
    def num_variables(self) -> int:
        raise NotImplementedError

    @property
    def variable_shapes(self) -> Sequence[int]:
        raise NotImplementedError

    def rename(self, new_var_names):
        if new_var_names is not None and len(new_var_names) != self.num_variables:
            raise ValueError(f"Number of names ({len(new_var_names)}) must match "
                             f"number of variables ({self.num_variables})")
        if new_var_names is None:
            self.variable_names = self._var_indices = None
        else:
            self.variable_names = list(new_var_names)
            self._var_indices = {name: i for i, name in enumerate(new_var_names)}

    def _check_index(self, index):
        if index < 0 or index >= self.num_variables:
            raise ValueError(f"Variable index ({index}) must be between 0 and "
                             f"number of variables ({self.num_variables})")

    def _get_checked_index(self, index_or_name):
        if isinstance(index_or_name, str) and self.variable_names:
            return self._var_indices[index_or_name]  # Assumed valid from initialisation
        self._check_index(index_or_name)
        return index_or_name

    def _marginalise_single(self, marg_index) -> TorchDistribution:
        raise NotImplementedError

    def _marginalise_multi(self, marg_indices) -> 'MultivariateDistribution':
        raise NotImplementedError

    def marginalise(self, which) -> Union[TorchDistribution, 'MultivariateDistribution']:
        if _is_single(which):
            which = self._get_checked_index(which)
            return self._marginalise_single(which)
        else:
            which = [self._get_checked_index(i) for i in which]
            marg = self._marginalise_multi(which)
            if self.variable_names:
                marg.rename([self.variable_names[i] for i in which])
            return marg

    def _condition(self, marg_indices, cond_indices, cond_values, squeeze) \
            -> Union[TorchDistribution, 'MultivariateDistribution']:
        raise NotImplementedError

    def condition(self, cond_dict, squeeze=False) -> Union[TorchDistribution, 'MultivariateDistribution']:
        if len(cond_dict) == 0:
            return self
        cond_dict = {self._get_checked_index(key): value for key, value in cond_dict.items()}
        marg_indices = [i for i in range(self.num_variables) if i not in cond_dict]
        if squeeze and len(marg_indices) > 1:
            raise RuntimeError(f"Only univariate distributions can be squeezed "
                               f"(num_variables={len(marg_indices)})")
        cond_indices = list(cond_dict.keys())
        cond_values = list(cond_dict.values())
        cond = self._condition(marg_indices, cond_indices, cond_values, squeeze)
        if self.variable_names:
            cond.rename([self.variable_names[i] for i in marg_indices])
        return cond

    def squeeze(self) -> TorchDistribution:
        if self.num_variables != 1:
            raise RuntimeError(f"Only univariate distributions can be squeezed "
                               f"(num_variables={self.num_variables})")
        return self.marginalise(0)

    def __call__(self, *marg_keys, squeeze=True, **cond_dict) \
            -> Union[TorchDistribution, 'MultivariateDistribution']:
        if len(cond_dict) == 0:
            if squeeze and len(marg_keys) > 1:
                raise RuntimeError(f"Only univariate distributions can be squeezed "
                                   f"(num_variables={len(marg_keys)})")
            return self.marginalise(marg_keys)
        elif len(marg_keys) == 0:
            return self.condition(cond_dict, squeeze)
        else:
            joined_keys = list(marg_keys) + list(cond_dict.keys())
            partial_dist = self.marginalise(joined_keys)
            return partial_dist.condition(cond_dict, squeeze)

    def __repr__(self):
        s = super().__repr__()[:-1]
        if self.variable_names is not None:
            s += f", variable_names: {self.variable_names}"
        return s + ')'
