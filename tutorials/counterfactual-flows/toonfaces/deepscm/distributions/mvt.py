"""
Multivariate t-distribution
"""
import math

import torch
from pyro.distributions import Chi2, TorchDistribution
from torch._six import inf, nan
from torch.distributions import constraints
from torch.distributions.utils import _standard_normal, broadcast_all, lazy_property

from deepscm.util import triangular_logdet, matvec


class MultivariateStudentT(TorchDistribution):
    r"""
    Creates a multivariate Student's t-distribution parameterized by degree of
    freedom :attr:`df`, mean :attr:`loc` and scale :attr:`scale`.
    Example::
        >>> m = MultivariateStudentT(torch.tensor([2.0]))
        >>> m.sample()  # Multivariate Student's t-distribution with degrees of freedom=2
        tensor([ 0.1046])
    Args:
        df (float or Tensor): degrees of freedom
        loc (float or Tensor): mean of the distribution
        scale (float or Tensor): scale of the distribution
    """
    arg_constraints = {'df': constraints.positive, 'loc': constraints.real,
                       'scale': constraints.positive_definite}
    support = constraints.real
    has_rsample = True

    def __init__(self, df, loc=0., scale=1., validate_args=None):
        self.df, self.loc, self.scale = broadcast_all(df, loc, scale)
        self._chi2 = Chi2(self.df)
        batch_shape = self.df.size()
        super(MultivariateStudentT, self).__init__(batch_shape, validate_args=validate_args)

    @property
    def mean(self):
        m = self.loc.clone()
        m[self.df <= 1] = nan
        return m

    @property
    def variance(self):
        m = self.df.clone()
        m[self.df > 2] = self.scale[self.df > 2] * self.df[self.df > 2] / (self.df[self.df > 2] - 2)
        m[(self.df <= 2) & (self.df > 1)] = inf
        m[self.df <= 1] = nan
        return m

    def rsample(self, sample_shape=torch.Size()):
        #   X ~ Normal(0, I)
        #   Z ~ Chi2(df)
        #   Y = X / sqrt(Z / df) ~ MultivariateStudentT(df)
        shape = self._extended_shape(sample_shape)
        X = _standard_normal(shape, dtype=self.df.dtype, device=self.df.device)
        Z = self._chi2.rsample(sample_shape)
        Y = X * torch.rsqrt(Z / self.df).unsqueeze(-1)
        return self.loc + matvec(self.scale_tril, Y)

    @lazy_property
    def scale_tril(self) -> torch.Tensor:
        return torch.cholesky(self.scale, upper=False)

    def _log_normalizer(self):
        return torch.lgamma(.5 * self.df) - torch.lgamma(.5 * (self.df + self.loc.shape[-1])) \
               + .5 * torch.log(math.pi * self.df) + triangular_logdet(self.scale_tril)

    def log_prob(self, x):
        D = self.loc.shape[-1]
        y = torch.triangular_solve((x - self.loc).unsqueeze(-1), self.scale_tril, upper=False)[0].squeeze(-1)
        maha = (y * y).sum(-1)
        return -.5 * (self.df + D) * torch.log1p(maha / self.df) - self._log_normalizer()

    def entropy(self):
        return .5 * (self.df + 1) * (torch.digamma(.5 * (self.df + self.loc.shape[-1]))
                                     - torch.digamma(.5 * self.df)) + self._log_normalizer()

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(MultivariateStudentT, _instance)
        batch_shape = torch.Size(batch_shape)
        new.df = self.df.expand(batch_shape)
        new.loc = self.loc.expand(batch_shape)
        new.scale = self.scale.expand(batch_shape)
        new._chi2 = self._chi2.expand(batch_shape)
        super(MultivariateStudentT, new).__init__(batch_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new
