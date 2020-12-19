import warnings
from functools import total_ordering
from typing import Sequence, Tuple

import torch
from torch import Tensor
from pyro.distributions import Categorical as PyroCategorical, MultivariateNormal as PyroMultivariateNormal
from torch.distributions import Distribution, Categorical, MultivariateNormal

from .factorised import Factorised
from .multivariate import MultivariateDistribution
from .natural_mvn import NaturalMultivariateNormal
from deepscm.util import posdef_solve

_PROD_REGISTRY = {}


def register_product(type_p, type_q):
    """
    Decorator to register a pairwise function with :meth:`product`.
    Usage::

        # TODO: Fix and explain the expected signature for product functions
        @register_product(Normal, Normal)
        def prod_normal_normal(p, q):
            # insert implementation here

    When `product(p, q)` is called, the lookup will attempt to match either `(type_p, type_q)` or
    `(type_q, type_p)`.

    Args:
        type_p (type): A subclass of :class:`~torch.distributions.Distribution`.
        type_q (type): A subclass of :class:`~torch.distributions.Distribution`.
    """
    if not isinstance(type_p, type) and issubclass(type_p, Distribution):
        raise TypeError(f"Expected type_p to be a Distribution subclass but got {type_p.__name__}")
    if not isinstance(type_q, type) and issubclass(type_q, Distribution):
        raise TypeError(f"Expected type_q to be a Distribution subclass but got {type_q.__name__}")

    def decorator(fun):
        _PROD_REGISTRY[type_p, type_q] = fun
        return fun

    return decorator


# Copied from https://github.com/pytorch/pytorch/blob/e0b90b87/torch/distributions/kl.py#L76-L92
@total_ordering
class _Match(object):
    __slots__ = ['types']

    def __init__(self, *types):
        self.types = types

    def __eq__(self, other):
        return self.types == other.types

    def __le__(self, other):
        for x, y in zip(self.types, other.types):
            if not issubclass(x, y):
                return False
            if x is not y:
                break
        return True


# Adapted from https://github.com/pytorch/pytorch/blob/e0b90b87/torch/distributions/kl.py#L95-L112
def _dispatch_prod(type_p, type_q):
    """
    Find the most specific approximate match, assuming single inheritance.
    """
    matches = [(super_p, super_q) for super_p, super_q in _PROD_REGISTRY
               if issubclass(type_p, super_p) and issubclass(type_q, super_q)]
    if not matches:
        raise NotImplementedError(f"No product implemented for {type_p.__name__} vs. {type_q.__name__}")
    # Check that the left- and right- lexicographic orders agree.
    left_p, left_q = min(_Match(*m) for m in matches).types
    right_q, right_p = min(_Match(*reversed(m)) for m in matches).types
    left_fun = _PROD_REGISTRY[left_p, left_q]
    right_fun = _PROD_REGISTRY[right_p, right_q]
    if left_fun is not right_fun:
        warnings.warn(f"Ambiguous product({type_p.__name__}, {type_q.__name__}). "
                      f"Please register_product({left_p.__name__}, {right_q.__name__})",
                      RuntimeWarning)
    return left_fun


def product(p: Distribution, q: Distribution, expand=True) -> Tuple[Distribution, Tensor]:
    """Computes the product of two distributions.

    Computes in closed form the distribution :math:`r(x)` and log-normalising constant
    :math:`\log Z` such that :math:`r(x) = p(x) \cdot q(x) / Z`.

    Args:
        p (Distribution): A distribution.
        q (Distribution): Another distribution.
        expand (bool): Whether to perform the outer product of the batch shapes. If ``True``,
            the output batch shape will be ``p.batch_shape + q.batch_shape``. Otherwise, it will be
            equal to the input batch shapes, which must be the same.

    Returns:
        (Distribution, Tensor): The resulting distribution :math:`r(x)` and the log-normalising
            constant :math:`\log Z`.

    Raises:
        NotImplementedError: If the distribution types have not been registered via
            :meth:`register_product`.
    """
    if p.event_shape != q.event_shape:
        raise ValueError("Cannot compute the product of distributions with different event shapes")
    if isinstance(p, MultivariateDistribution) and isinstance(q, MultivariateDistribution):
        if p.num_variables != q.num_variables:
            raise ValueError(f"Cannot compute the product of multivariate distributions with "
                             f"different numbers of variables: "
                             f"{p.num_variables} vs. {q.num_variables}")
        for pn, qn in zip(p.variable_names, q.variable_names):
            if pn != qn:
                raise ValueError(f"Cannot compute the product of multivariate distributions with "
                                 f"different variable names: "
                                 f"{p.variable_names} vs. {q.variable_names}")

    p_type, q_type = type(p), type(q)
    p_shape, q_shape = _broadcast_shapes(p.batch_shape, q.batch_shape, expand)
    try:
        prod_fcn = _dispatch_prod(p_type, q_type)
        pq, pq_logprob = prod_fcn(p, q, p_shape, q_shape)
    except NotImplementedError:
        prod_fcn = _dispatch_prod(q_type, p_type)
        pq, pq_logprob = prod_fcn(q, p, q_shape, p_shape)

    if isinstance(pq, MultivariateDistribution):
        var_names = None
        if isinstance(p, MultivariateDistribution):
            var_names = p.variable_names
        elif isinstance(q, MultivariateDistribution):
            var_names = q.variable_names
        pq.rename(var_names)

    return pq, pq_logprob


def _broadcast_shapes(p_shape, q_shape, expand):
    if not expand:
        if p_shape != q_shape:
            raise ValueError(f"Equal batch shapes expected, got {p_shape} vs. {q_shape}")
        return p_shape, q_shape
    p_empty_shape = torch.Size([1] * len(p_shape))
    q_empty_shape = torch.Size([1] * len(q_shape))
    p_new_shape = p_shape + q_empty_shape
    q_new_shape = p_empty_shape + q_shape
    return p_new_shape, q_new_shape


def _reshape_batch(tensors: Sequence[torch.Tensor], old_shape: torch.Size, new_shape: torch.Size):
    if old_shape == new_shape:
        return tensors
    return [t.reshape(new_shape + t.shape[len(old_shape):]) for t in tensors]


@register_product(Categorical, Categorical)
def _prod_categorical_categorical(p: Categorical, q: Categorical, p_shape, q_shape):
    p_logits, = _reshape_batch([p.logits], p.batch_shape, p_shape)
    q_logits, = _reshape_batch([q.logits], q.batch_shape, q_shape)

    pq_logits = p_logits + q_logits

    if isinstance(p, PyroCategorical) or isinstance(q, PyroCategorical):
        pq = PyroCategorical(logits=pq_logits)
    else:
        pq = Categorical(logits=pq_logits)
    pq_lognorm = pq_logits.logsumexp(-1) - p_logits.logsumexp(-1) - q_logits.logsumexp(-1)

    return pq, pq_lognorm


@register_product(MultivariateNormal, MultivariateNormal)
def _prod_mvn_mvn(p: MultivariateNormal, q: MultivariateNormal, p_shape, q_shape):
    p_mean, p_prec, p_cov = _reshape_batch([p.mean, p.precision_matrix, p.covariance_matrix],
                                           p.batch_shape, p_shape)
    q_mean, q_prec, q_cov = _reshape_batch([q.mean, q.precision_matrix, q.covariance_matrix],
                                           q.batch_shape, q_shape)

    pq_prec = p_prec + q_prec
    pq_mean = posdef_solve(p_prec @ p_mean[..., None] + q_prec @ q_mean[..., None],
                           pq_prec)[0].squeeze(-1)

    if isinstance(p, PyroMultivariateNormal) or isinstance(q, PyroMultivariateNormal):
        pq = PyroMultivariateNormal(pq_mean, precision_matrix=pq_prec)
    else:
        pq = MultivariateNormal(pq_mean, precision_matrix=pq_prec)
    pq_lognorm = PyroMultivariateNormal(q_mean, p_cov + q_cov).log_prob(p_mean)

    return pq, pq_lognorm


@register_product(NaturalMultivariateNormal, NaturalMultivariateNormal)
def _prod_nmvn_nmvn(p: NaturalMultivariateNormal, q: NaturalMultivariateNormal, p_shape, q_shape):
    p_eta1, p_eta2 = _reshape_batch([p.nat_param1, p.nat_param2], p.batch_shape, p_shape)
    q_eta1, q_eta2 = _reshape_batch([q.nat_param1, q.nat_param2], q.batch_shape, q_shape)

    pq_eta1 = p_eta1 + q_eta1
    pq_eta2 = p_eta2 + q_eta2

    pq = NaturalMultivariateNormal(pq_eta1, pq_eta2)
    pq_lognorm = pq.log_normalizer - p.log_normalizer.reshape(p_shape) \
        - q.log_normalizer.reshape(q_shape)

    return pq, pq_lognorm


@register_product(NaturalMultivariateNormal, MultivariateNormal)
def _prod_nmvn_mvn(p: NaturalMultivariateNormal, q: MultivariateNormal, p_shape, q_shape):
    return _prod_nmvn_nmvn(p, NaturalMultivariateNormal.from_standard(q), p_shape, q_shape)


@register_product(Factorised, Factorised)
def _prod_factorised_factorised(p: Factorised, q: Factorised, p_shape, q_shape):
    if len(p.factors) != len(q.factors):
        raise ValueError("Factorised distributions must have the same number of factors")
    expand = p_shape != p.batch_shape or q_shape != q.batch_shape
    pq_factors_lognorms = [product(p_factor, q_factor, expand)
                           for p_factor, q_factor in zip(p.factors, q.factors)]
    pq_factors, pq_lognorms = zip(*pq_factors_lognorms)
    pq = Factorised(list(pq_factors))
    pq_lognorm = sum(pq_lognorms)
    return pq, pq_lognorm
