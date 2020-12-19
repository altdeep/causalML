import math

import torch
import torch.distributions as td
from torch.distributions import constraints
from torch.distributions.utils import lazy_property

from .multivariate import MultivariateDistribution
from .mvn import MultivariateNormal
from deepscm.util import inverse_cholesky, mahalanobis, matvec, triangular_logdet

_LOG_2PI = math.log(2. * math.pi)


class _NegativeDefinite(constraints.Constraint):
    """
    Constrain to negative-definite matrices.

    Adapted from: https://github.com/pytorch/pytorch/blob/master/torch/distributions/constraints.py
    """
    def check(self, value):
        return value.symeig(eigenvectors=False)[0][..., -1] < 0.0


class NaturalMultivariateNormal(MultivariateDistribution, td.ExponentialFamily):
    arg_constraints = {'nat_param1': constraints.real_vector,
                       'nat_param2': _NegativeDefinite()}
    support = constraints.real
    has_rsample = True

    def __init__(self, nat_param1: torch.Tensor, nat_param2: torch.Tensor, validate_args=None,
                 var_names=None):
        batch_shape = torch.Size(nat_param1.shape[:-1])
        event_shape = torch.Size((nat_param1.shape[-1],))
        self.nat_param1 = nat_param1
        self.nat_param2 = nat_param2
        super().__init__(batch_shape, event_shape, validate_args=validate_args, var_names=var_names)

    @property
    def _natural_params(self):
        return self.nat_param1, self.nat_param2

    def _log_normalizer(self, nat_param1, nat_param2):
        maha, prec_tril = mahalanobis(-2. * nat_param2, nat_param1)
        logdet = triangular_logdet(prec_tril)
        return .5 * maha - logdet + .5 * self.event_shape[0] * _LOG_2PI

    @lazy_property
    def log_normalizer(self):
        return self._log_normalizer(self.nat_param1, self.nat_param2)

    @property
    def _mean_carrier_measure(self):
        return 0.

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        if self._validate_args:
            self._validate_sample(value)
        prod1 = (value * self.nat_param1).sum(-1)
        prod2 = (value * matvec(self.nat_param2, value)).sum(-1)
        return prod1 + prod2 - self.log_normalizer

    def entropy(self) -> torch.Tensor:
        logdet = triangular_logdet(self.scale_tril)
        return .5 * self.event_shape[0] * (1. + _LOG_2PI) + logdet

    @lazy_property
    def scale_tril(self) -> torch.Tensor:
        return inverse_cholesky(self.precision_matrix)

    @lazy_property
    def precision_matrix(self) -> torch.Tensor:
        return -2. * self.nat_param2

    @lazy_property
    def covariance_matrix(self) -> torch.Tensor:
        return self.scale_tril @ self.scale_tril.transpose(-1, -2)

    @lazy_property
    def mean(self) -> torch.Tensor:
        return matvec(self.covariance_matrix, self.nat_param1)

    def rsample(self, sample_shape=torch.Size()):
        dtype, device = self.nat_param1.dtype, self.nat_param1.device
        noise = torch.randn(*self._extended_shape(sample_shape), dtype=dtype, device=device)
        return self.mean + matvec(self.scale_tril, noise)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(NaturalMultivariateNormal, _instance)
        batch_shape = torch.Size(batch_shape)
        eta1_shape = batch_shape + self.event_shape
        eta2_shape = batch_shape + self.event_shape + self.event_shape
        new.nat_param1 = self.nat_param1.expand(eta1_shape)
        new.nat_param2 = self.nat_param2.expand(eta2_shape)
        super(NaturalMultivariateNormal, new).__init__(batch_shape, self.event_shape,
                                                       validate_args=False)
        new._validate_args = self._validate_args
        return new

    def to_standard(self) -> MultivariateNormal:
        return MultivariateNormal(self.mean, scale_tril=self.scale_tril,
                                  validate_args=self._validate_args)

    @staticmethod
    def from_standard(mvn: td.MultivariateNormal) -> 'NaturalMultivariateNormal':
        precision = mvn.precision_matrix
        nat_param2 = -.5 * precision
        nat_param1 = matvec(precision, mvn.mean)
        return NaturalMultivariateNormal(nat_param1, nat_param2, validate_args=mvn._validate_args)

    @property
    def num_variables(self):
        return self.event_shape[0]

    @property
    def variable_shapes(self):
        return [1] * self.num_variables

    def _marginalise_single(self, index):
        mvn = self.to_standard()
        return mvn.marginalise(index)

    def _marginalise_multi(self, indices):
        mvn = self.to_standard()
        marg_mvn = mvn.marginalise(indices)
        return NaturalMultivariateNormal.from_standard(marg_mvn)

    def _condition(self, y_dims, x_dims, x, squeeze):
        mvn = self.to_standard()
        cond_mvn = mvn._condition(y_dims, x_dims, x, squeeze)
        if squeeze:
            return cond_mvn
        return NaturalMultivariateNormal.from_standard(cond_mvn)


def eval_grid(xx, yy, fcn):
    xy = torch.stack([xx.flatten(), yy.flatten()], dim=1)
    return fcn(xy).reshape_as(xx)


def plot_dist(xx, yy, dist: td.Distribution, data=None):
    xlim = [xx.min(), xx.max()]
    ylim = [yy.min(), yy.max()]
    with torch.no_grad():
        if len(dist.batch_shape) > 0:
            zz = eval_grid(xx, yy, lambda xy: dist.log_prob(xy)[..., 0])
        else:
            zz = eval_grid(xx, yy, dist.log_prob)
    plt.imshow(zz.exp().T, interpolation='bilinear', origin='lower', extent=[*xlim, *ylim])
    # plt.contourf(xx, yy, zz.exp(), cmap='viridis')
    if data is not None:
        plt.scatter(*data.T, c='k', s=4)
    plt.xlim(xlim)
    plt.ylim(ylim)


if __name__ == '__main__':
    N, D = 1000, 2
    mean = torch.ones(D)
    stdx, stdy, corr = 1., 1., -.7
    cov = torch.tensor([[stdx ** 2, corr * stdx * stdy],
                        [corr * stdx * stdy, stdy ** 2]])
    # cov = torch.ones(D) + .01 * torch.eye(D)

    mean.requires_grad_()
    mvn = td.MultivariateNormal(mean.expand(5, -1), cov, validate_args=True)
    nmvn = NaturalMultivariateNormal.from_standard(mvn)
    mvn_ = nmvn.to_standard()

    print(mvn.entropy())
    print(nmvn.entropy())
    print(mvn_.entropy())
    X = nmvn.rsample((N,))
    # print(X.shape)
    print(mvn.rsample((N,)).shape)
    print(nmvn.log_prob(X).shape)
    print(torch.allclose(mvn.log_prob(X), nmvn.log_prob(X)))
    print(mvn_.rsample((N,)).shape)
    # torch.autograd.grad(X.mean(), mean)

    import matplotlib.pyplot as plt

    # # xlim, ylim = zip(X.min(0)[0], X.max(0)[0])
    # xlim = [mean[0].item() - 2. * stdx, mean[0].item() + 2. * stdx]
    # ylim = [mean[1].item() - 2. * stdy, mean[1].item() + 2. * stdy]
    # x = torch.linspace(*xlim, 200)
    # y = torch.linspace(*ylim, 200)
    # xx, yy = torch.meshgrid(x, y)
    #
    # plot_dist(xx, yy, mvn, mvn.sample((N,)))
    # plt.title("Original")
    # plt.show()
    # plot_dist(xx, yy, nmvn, nmvn.sample((N,)))
    # plt.title("Natural")
    # plt.show()
    # plot_dist(xx, yy, mvn_, mvn_.sample((N,)))
    # plt.title("Reconstructed")
    # plt.show()

    # #
    # mvn_logp = mvn.log_prob(samples)
    # nmvn_logp = nmvn.log_prob(samples)
    # mvn__logp = mvn_.log_prob(samples)
    #
    # diff1 = mvn_logp - nmvn_logp
    # diff2 = mvn_logp - mvn__logp
    # print("Avg diff log: {:.4g}".format(diff1.mean()))
    # print("Avg diff log: {:.4g}".format(diff2.mean()))
    # print()
    # print("Avg abs diff log: {:.4g}".format(diff1.abs().mean()))
    # print("Avg abs diff log: {:.4g}".format(diff2.abs().mean()))
    # print("Avg rel diff log: {:.4f} %".format(100. * diff1.abs().mean().expm1()))
    # print("Avg rel diff log: {:.4f} %".format(100. * diff2.abs().mean().expm1()))
    # print()
    # print("Max abs diff log: {:.4g}".format(diff1.abs().max()))
    # print("Max abs diff log: {:.4g}".format(diff2.abs().max()))
    # print("Max rel diff: {:.4f} %".format(100. * diff1.abs().max().expm1()))
    # print("Max rel diff: {:.4f} %".format(100. * diff2.abs().max().expm1()))
