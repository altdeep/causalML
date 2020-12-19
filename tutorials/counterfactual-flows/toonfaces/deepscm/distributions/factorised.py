from typing import Mapping, Sequence, Union

import torch
from pyro.distributions import TorchDistribution
from torch.distributions import register_kl
from torch.distributions.constraints import Constraint

from .multivariate import MultivariateDistribution


def _iterate_parts(value, ndims: Sequence[int]):
    for ndim in ndims:
        yield value[..., :ndim]
        value = value[..., ndim:]


class _FactorisedSupport(Constraint):
    def __init__(self, supports: Sequence[Constraint], ndims: Sequence[int]):
        self.supports = supports
        self.ndims = ndims

    def check(self, value):
        return all(support.check(part)
                   for support, part in zip(self.supports, _iterate_parts(value, self.ndims)))


class Factorised(MultivariateDistribution):
    arg_constraints = {}

    def __init__(self, factors: Union[Sequence[TorchDistribution], Mapping[str, TorchDistribution]],
                 validate_args=None, var_names=None):
        if isinstance(factors, Mapping):
            if var_names is not None:
                raise ValueError("var_names should not be given alongside a factor dictionary")
            var_names = list(factors.keys())
            factors = list(factors.values())
        self.factors = factors
        batch_shape = factors[0].batch_shape
        event_shape = torch.Size([sum(factor.event_shape[0] for factor in self.factors)])
        self._ndims = [factor.event_shape[0] if len(factor.event_shape) > 0 else 1
                       for factor in self.factors]
        super().__init__(batch_shape, event_shape, validate_args, var_names=var_names)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(Factorised, _instance)
        batch_shape = torch.Size(batch_shape)
        new.factors = [factor.expand(batch_shape) for factor in self.factors]
        new._ndims = self._ndims
        super(Factorised, new).__init__(batch_shape, self.event_shape, validate_args=False,
                                        var_names=self.variable_names)
        new._validate_args = self._validate_args
        return new

    @property
    def has_rsample(self):
        return any(factor.has_rsample for factor in self.factors)

    @property
    def support(self):
        return _FactorisedSupport([factor.support for factor in self.factors], self._ndims)

    def rsample(self, sample_shape=torch.Size()):
        return torch.cat([factor.rsample(sample_shape) for factor in self.factors], dim=-1)

    @property
    def num_variables(self):
        return len(self.factors)

    @property
    def variable_shapes(self):
        return [factor.event_shape[0] for factor in self.factors]

    def _marginalise_single(self, factor_index: int) -> TorchDistribution:
        return self.factors[factor_index]

    def _marginalise_multi(self, factor_indices: Sequence[int]) -> 'Factorised':
        return Factorised([self.factors[i] for i in factor_indices],
                          validate_args=self._validate_args)

    def _condition(self, marg_indices, cond_indices, cond_values, squeeze):
        cond_dist = self.marginalise(marg_indices[0] if squeeze else marg_indices)
        cond_batch_shape = torch.Size([cond_values[0].shape[0]]) + cond_dist.batch_shape
        return cond_dist.expand(cond_batch_shape)

    def log_prob(self, value):
        return sum(factor.log_prob(part)
                   for factor, part in zip(self.factors, self.partition_dimensions(value)))

    def entropy(self):
        return sum(factor.entropy() for factor in self.factors)

    def partition_dimensions(self, data):
        return _iterate_parts(data, self._ndims)

    @property
    def mean(self):
        return torch.cat([factor.mean for factor in self.factors], dim=-1)

    @property
    def variance(self):
        return sum(factor.variance for factor in self.factors)

    def __repr__(self):
        factors = self.factors
        if self.variable_names is not None:
            factors = dict(zip(self.variable_names, factors))
        return self.__class__.__name__ + f"({factors})"


@register_kl(Factorised, Factorised)
def _kl_factorised_factorised(p: Factorised, q: Factorised):
    return sum(kl_divergence(p_factor, q_factor)
               for p_factor, q_factor in zip(p.factors, q.factors))


if __name__ == '__main__':
    from pyro.distributions import Dirichlet, MultivariateNormal
    from torch.distributions import kl_divergence
    from distributions.mixture import Mixture

    B, D1, D2 = 5, 3, 4
    N = 1000

    dist1 = MultivariateNormal(torch.zeros(D1), torch.eye(D1)).expand((B,))
    dist2 = Dirichlet(torch.ones(D2)).expand((B,))
    print(dist1.batch_shape, dist1.event_shape)
    print(dist2.batch_shape, dist2.event_shape)
    fact = Factorised([dist1, dist2])
    print(fact.batch_shape, fact.event_shape)
    samples = fact.rsample((N,))
    print(samples[0])
    print(samples.shape)
    logp = fact.log_prob(samples)
    print(logp.shape)
    entropy = fact.entropy()
    print(entropy.shape)
    print(entropy, -logp.mean())
    print()

    print(kl_divergence(fact, fact))
    mixture = Mixture(torch.ones(B), fact)
    samples = mixture.rsample((N,))
    logp = mixture.log_prob(samples)
    print(samples.shape)
    print(logp.shape)
