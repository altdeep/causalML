import math

import torch
from pyro.distributions import TorchDistribution
from torch.distributions import constraints
from torch.distributions.exp_family import ExponentialFamily
from torch.distributions.utils import lazy_property

from deepscm import util


def mvdigamma(x: torch.Tensor, p: int) -> torch.Tensor:
    i = torch.arange(p, dtype=x.dtype, device=x.device)
    return torch.digamma(x.unsqueeze(-1) - .5 * i).sum(-1)


def mvtrigamma(x: torch.Tensor, p: int) -> torch.Tensor:
    i = torch.arange(p, dtype=x.dtype, device=x.device)
    return torch.polygamma(1, x.unsqueeze(-1) - .5 * i).sum(-1)


class Wishart(TorchDistribution, ExponentialFamily):
    arg_constraints = {'scale': constraints.positive_definite}
    support = constraints.positive_definite
    has_rsample = True
    _mean_carrier_measure = 0

    @property
    def mean(self):
        return 2. * self.concentration[..., None, None] \
               * (self.scale_tril @ self.scale_tril.transpose(-2, -1))

    def __init__(self, concentration, scale, validate_args=None):
        self.concentration = torch.as_tensor(concentration)
        self.scale = torch.as_tensor(scale)
        batch_shape = self.concentration.shape
        event_shape = self.scale.shape[-2:]
        self.arg_constraints['concentration'] = constraints.greater_than(.5 * (event_shape[-1] - 1))
        super(Wishart, self).__init__(batch_shape, event_shape, validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(Wishart, _instance)
        batch_shape = torch.Size(batch_shape)
        new.concentration = self.concentration.expand(batch_shape)
        new.scale = self.scale.expand(batch_shape + self.event_shape)
        super(Wishart, new).__init__(batch_shape, self.event_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new

    @lazy_property
    def scale_tril(self):
        return util.inverse_cholesky(2. * self.scale)

    def rsample(self, sample_shape=torch.Size()):
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
        shape = torch.Size(sample_shape) + self.batch_shape
        dtype, device = self.concentration.dtype, self.concentration.device
        D = self.event_shape[-1]
        df = 2. * self.concentration  # type: torch.Tensor
        i = torch.arange(D, dtype=dtype, device=device)
        concentration = .5 * (df.unsqueeze(-1) - i).expand(shape + (D,))
        V = 2. * torch._standard_gamma(concentration)
        N = torch.randn(*shape, D * (D - 1) // 2, dtype=dtype, device=device)
        T = torch.diag_embed(V.sqrt())  # T is lower-triangular
        i, j = torch.tril_indices(D, D, offset=-1)
        T[..., i, j] = N
        M = self.scale_tril @ T
        W = M @ M.transpose(-2, -1)
        return W

    def _log_normalizer(self, eta1, eta2):
        D = self.event_shape[-1]
        a = -eta1 - .5 * (D + 1)
        return torch.mvlgamma(a, D) - a * util.posdef_logdet(-eta2)[0]

    @lazy_property
    def log_normalizer(self):
        D = self.event_shape[-1]
        return torch.mvlgamma(self.concentration, D) + self.concentration \
            * (D * math.log(2.) + 2. * util.triangular_logdet(self.scale_tril))

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        D = self.event_shape[-1]
        logdet = util.posdef_logdet(value)[0]
        prod = (self.scale * value).sum(dim=(-2, -1))
        return (self.concentration - .5 * (D + 1)) * logdet - prod - self.log_normalizer

    def expected_logdet(self):
        D = self.event_shape[-1]
        return mvdigamma(self.concentration, D) + D * math.log(2.) \
            + 2. * util.triangular_logdet(self.scale_tril)

    def variance_logdet(self):
        return mvtrigamma(self.concentration, self.event_shape[-1])

    def entropy(self):
        D = self.event_shape[-1]
        E_logdet = self.expected_logdet()
        return self.log_normalizer - (self.concentration - .5 * (D + 1)) * E_logdet \
            + self.concentration * D

    @property
    def _natural_params(self):
        return self.concentration - .5 * (self.event_shape[-1] + 1), -self.scale
