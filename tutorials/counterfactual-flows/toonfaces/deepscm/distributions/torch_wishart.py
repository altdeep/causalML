import math

import torch
from torch.distributions import constraints
from torch.distributions.exp_family import ExponentialFamily
from torch.distributions.utils import lazy_property

_LOG_2 = math.log(2)


def mvdigamma(x: torch.Tensor, p: int) -> torch.Tensor:
    i = torch.arange(p, dtype=x.dtype, device=x.device)
    return torch.digamma(x.unsqueeze(-1) - .5 * i).sum(-1)


def mvtrigamma(x: torch.Tensor, p: int) -> torch.Tensor:
    i = torch.arange(p, dtype=x.dtype, device=x.device)
    return torch.polygamma(1, x.unsqueeze(-1) - .5 * i).sum(-1)


def _triangular_logdet(tri: torch.Tensor) -> torch.Tensor:
    return tri.diagonal(dim1=-2, dim2=-1).log().sum(-1)


def _posdef_logdet(A: torch.Tensor) -> torch.Tensor:
    tril = torch.cholesky(A, upper=False)
    return 2. * _triangular_logdet(tril)


def _batched_cholesky_inverse(tri, upper=False) -> torch.Tensor:
    eye = torch.eye(tri.shape[-1], dtype=tri.dtype, device=tri.device).expand_as(tri)
    return torch.cholesky_solve(eye, tri, upper=upper)


def _standard_wishart_tril(df: torch.Tensor, dim: int, shape: torch.Size):
    """
    References
    ----------
    - Sawyer, S. (2007). Wishart Distributions and Inverse-Wishart Sampling.
      https://www.math.wustl.edu/~sawyer/hmhandouts/Wishart.pdf
    - Anderson, T. W. (2003). An Introduction to Multivariate Statistical Analysis (3rd ed.).
      John Wiley & Sons, Inc.
    - Odell, P. L. & Feiveson, A. H. (1966). A Numerical Procedure to Generate a Sample
      Covariance Matrix. Journal of the American Statistical Association, 61(313):199-203.
    - Ku, Y.-C. & Blomfield, P. (2010). Generating Random Wishart Matrices with Fractional
      Degrees of Freedom in OX.
    """
    dtype, device = df.dtype, df.device
    i = torch.arange(dim, dtype=dtype, device=device)
    concentration = .5 * (df.unsqueeze(-1) - i).expand(shape + (dim,))
    V = 2. * torch._standard_gamma(concentration)
    N = torch.randn(*shape, dim * (dim - 1) // 2, dtype=dtype, device=device)
    T = torch.diag_embed(V.sqrt())  # T is lower-triangular
    i, j = torch.tril_indices(dim, dim, offset=-1)
    T[..., i, j] = N
    return T


class Wishart(ExponentialFamily):
    arg_constraints = {'scale': constraints.positive_definite,
                       'scale_tril': constraints.lower_cholesky}
    support = constraints.positive_definite
    has_rsample = True
    _mean_carrier_measure = 0

    def __init__(self, df, scale=None, scale_tril=None, validate_args=None):
        self.df = torch.as_tensor(df)
        if (scale is None) + (scale_tril is None) != 1:
            raise ValueError("Exactly one of scale or scale_tril must be specified.")
        if scale is not None:
            self.scale = torch.as_tensor(scale)
        if scale_tril is not None:
            self.scale_tril = torch.as_tensor(scale_tril)
        batch_shape = self.df.shape
        event_shape = self.scale_tril.shape[-2:]
        self.arg_constraints['df'] = constraints.greater_than(event_shape[-1] - 1)
        super(Wishart, self).__init__(batch_shape, event_shape, validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(Wishart, _instance)
        batch_shape = torch.Size(batch_shape)
        new.df = self.df.expand(batch_shape)
        new.scale_tril = self.scale_tril.expand(batch_shape + self.event_shape)
        super(Wishart, new).__init__(batch_shape, self.event_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new

    @lazy_property
    def scale(self):
        return self.scale_tril @ self.scale_tril.transpose(-2, -1)

    @lazy_property
    def scale_tril(self):
        return torch.cholesky(self.scale, upper=False)

    @property
    def mean(self):
        return self.df[..., None, None] * self.scale

    def rsample(self, sample_shape=torch.Size()):
        shape = torch.Size(sample_shape) + self.batch_shape
        T = _standard_wishart_tril(self.df, self.event_shape[-1], shape)
        M = self.scale_tril @ T
        W = M @ M.transpose(-2, -1)
        return W

    def _log_normalizer(self, eta1, eta2):
        D = self.event_shape[-1]
        a = eta1 + .5 * (D + 1)
        return torch.mvlgamma(a, D) - a * _posdef_logdet(-eta2)

    @lazy_property
    def log_normalizer(self):
        D = self.event_shape[-1]
        return torch.mvlgamma(.5 * self.df, D) + .5 * self.df \
            * (D * _LOG_2 + 2. * _triangular_logdet(self.scale_tril))

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        D = self.event_shape[-1]
        logdet = _posdef_logdet(value)
        # TODO: Replace with cholesky_solve once its derivative is implemented
        # prod = torch.cholesky_solve(value, self.scale_tril, upper=False)
        prod_ = torch.triangular_solve(value, self.scale_tril, upper=False)[0]
        prod = torch.triangular_solve(prod_, self.scale_tril, upper=False, transpose=True)[0]
        trace = prod.diagonal(dim1=-2, dim2=-1).sum(-1)
        return .5 * (self.df - D - 1) * logdet - .5 * trace - self.log_normalizer

    def _expected_logdet(self):
        D = self.event_shape[-1]
        return mvdigamma(.5 * self.df, D) + D * _LOG_2 + 2. * _triangular_logdet(self.scale_tril)

    def _variance_logdet(self):
        return mvtrigamma(.5 * self.df, self.event_shape[-1])

    def entropy(self):
        D = self.event_shape[-1]
        E_logdet = self._expected_logdet()
        return self.log_normalizer - .5 * (self.df - D - 1) * E_logdet + .5 * self.df * D

    @property
    def _natural_params(self):
        return .5 * (self.df - self.event_shape[-1] - 1), \
               -.5 * _batched_cholesky_inverse(self.scale_tril)


class InverseWishart(Wishart):
    arg_constraints = {'scale': constraints.positive_definite,
                       'scale_tril': constraints.lower_cholesky}
    support = constraints.positive_definite
    has_rsample = True
    _mean_carrier_measure = 0

    def __init__(self, df, scale=None, scale_tril=None, validate_args=None):
        super(InverseWishart, self).__init__(df, scale=scale, scale_tril=scale_tril,
                                             validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(InverseWishart, _instance)
        return super(InverseWishart, self).expand(batch_shape, new)

    @property
    def mean(self):
        return self.scale / (self.df[..., None, None] - self.event_shape[-1] - 1)

    def rsample(self, sample_shape=torch.Size()):
        shape = torch.Size(sample_shape) + self.batch_shape
        T = _standard_wishart_tril(self.df, self.event_shape[-1], shape)
        eye = torch.eye(T.shape[-1], dtype=T.dtype, device=T.device).expand_as(T)
        # TODO: Replace with cholesky_solve once its derivative is implemented
        # inv_scale = torch.cholesky_solve(eye, self.scale_tril, upper=False)
        inv_scale_ = torch.triangular_solve(eye, self.scale_tril, upper=False)[0]
        inv_scale = torch.triangular_solve(inv_scale_, self.scale_tril, upper=False, transpose=True)[0]
        assert torch.allclose(inv_scale @ self.scale, eye)
        inv_scale_tril = torch.cholesky(inv_scale, upper=False)
        assert torch.allclose((inv_scale_tril @ inv_scale_tril.transpose(-2, -1)) @ self.scale, eye)
        M = inv_scale_tril @ T
        M_inv = torch.triangular_solve(eye, M, upper=False)[0]
        W = M_inv.transpose(-2, -1) @ M_inv
        assert constraints.lower_cholesky.check(T).all()
        assert constraints.lower_cholesky.check(M).all()
        assert constraints.lower_cholesky.check(M_inv).all()
        assert constraints.positive_definite.check(W).all()
        return W

    def _log_normalizer(self, eta1, eta2):
        D = self.event_shape[-1]
        a = -(eta1 + .5 * (D + 1))
        return torch.mvlgamma(a, D) - a * _posdef_logdet(-eta2)

    @lazy_property
    def log_normalizer(self):
        D = self.event_shape[-1]
        return torch.mvlgamma(.5 * self.df, D) + .5 * self.df \
            * (D * _LOG_2 - 2. * _triangular_logdet(self.scale_tril))

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        D = self.event_shape[-1]
        value_tril = torch.cholesky(value, upper=False)
        logdet = 2. * _triangular_logdet(value_tril)
        # TODO: Replace with cholesky_solve once its derivative is implemented
        # prod = torch.cholesky_solve(self.scale, value_tril, upper=False)
        prod_ = torch.triangular_solve(self.scale, value_tril, upper=False)[0]
        prod = torch.triangular_solve(prod_, value_tril, upper=False, transpose=True)[0]
        trace = prod.diagonal(dim1=-2, dim2=-1).sum(-1)
        return -.5 * (self.df + D + 1) * logdet - .5 * trace - self.log_normalizer

    def _expected_inverse(self):
        return self.df[..., None, None] * _batched_cholesky_inverse(self.scale_tril)

    def _expected_logdet(self):
        D = self.event_shape[-1]
        return -mvdigamma(.5 * self.df, D) - D * _LOG_2 + 2. * _triangular_logdet(self.scale_tril)

    def _variance_logdet(self):
        raise NotImplementedError
        # return mvtrigamma(.5 * self.df, self.event_shape[-1])

    def entropy(self):
        D = self.event_shape[-1]
        E_logdet = self._expected_logdet()
        return self.log_normalizer + .5 * (self.df + D + 1) * E_logdet + .5 * self.df * D

    @property
    def _natural_params(self):
        return -.5 * (self.df + self.event_shape[-1] + 1), \
               -.5 * self.scale
