from typing import Generic, TypeVar, Union

import torch
from pyro.distributions import Categorical, MultivariateNormal, TorchDistribution

from .multivariate import MultivariateDistribution
from .natural_mvn import NaturalMultivariateNormal
from .products import product
from deepscm.util import posdef_solve

T = TypeVar('T', bound=TorchDistribution)


class Mixture(TorchDistribution, Generic[T]):
    def __new__(cls, proportions, components, *args, **kwargs):
        # Automatically construct a MultivariateMixture for multivariate components
        if isinstance(components, MultivariateDistribution):
            return super().__new__(MultivariateMixture, proportions, components, *args, **kwargs)
        return super().__new__(Mixture, proportions, components, *args, **kwargs)

    def __init__(self, proportions: Union[Categorical, torch.Tensor], components: T):
        if isinstance(proportions, torch.Tensor):
            proportions = Categorical(proportions)
        if proportions._num_events != components.batch_shape[-1]:
            raise ValueError(f"Length of proportions vector ({proportions._num_events}) "
                             f"must match number of components ({components.batch_shape[-1]}).")
        self.mixing = proportions
        self.components = components
        super().__init__(components.batch_shape[:-1], components.event_shape)

    @property
    def num_components(self) -> int:
        return self.components.batch_shape[-1]

    def rsample(self, sample_shape=torch.Size()):
        assignments = self.mixing.sample(sample_shape)
        samples = self.components.rsample(sample_shape)
        batch_shape = self.batch_shape
        full_shape = [*sample_shape, *batch_shape, 1, *self.event_shape]
        thin_shape = [*sample_shape, *batch_shape, 1] + [1] * len(self.event_shape)
        sdim = len(sample_shape) + len(batch_shape)
        assignments = assignments.view(thin_shape).expand(full_shape)
        return samples.gather(sdim, assignments, sparse_grad=False).squeeze(sdim)

    def _broadcast(self, sample: torch.Tensor) -> torch.Tensor:
        component_axis = sample.dim() - len(self.event_shape)
        return sample.unsqueeze(component_axis)

    @property
    def mean(self):
        probs = self.mixing.probs  # (..., K)
        means = self.components.mean  # (..., K) + event_shape
        ndim = len(self.event_shape)
        empty_event_shape = torch.Size([1] * ndim)
        probs = probs.reshape(probs.shape + empty_event_shape)
        mean = (probs * means).sum(-ndim - 1)
        return mean

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        log_liks = self.components.log_prob(self._broadcast(value))
        return torch.logsumexp(self.mixing.logits + log_liks, dim=-1)

    def posterior(self, potentials: T) -> 'Mixture':
        post_components, post_lognorm = product(potentials, self.components, expand=True)
        post_logits = self.mixing.logits + post_lognorm
        post_mixing = Categorical(logits=post_logits)
        return Mixture(post_mixing, post_components)

    def __repr__(self):
        return self.__class__.__name__ + f"(mixing: {self.mixing}, components: {self.components})"


class MultivariateMixture(MultivariateDistribution, Mixture[MultivariateDistribution]):
    def __init__(self, proportions, components: MultivariateDistribution):
        super().__init__(proportions, components, var_names=components.variable_names)

    @property
    def num_variables(self):
        return self.components.num_variables

    @property
    def variable_shapes(self):
        return self.components.variable_shapes

    def rename(self, new_var_names):
        super().rename(new_var_names)
        self.components.rename(new_var_names)

    def marginalise(self, which):
        marg_components = self.components.marginalise(which)
        return Mixture(self.mixing, marg_components)

    def _condition(self, marg_indices, cond_indices, cond_values, squeeze):
        cond_values = [self._broadcast(value) for value in cond_values]
        marg_components = self.components.marginalise(cond_indices)
        marg_values = torch.cat(cond_values, -1)
        cond_logits = self.mixing.logits + marg_components.log_prob(marg_values)
        cond_mixing = Categorical(logits=cond_logits)
        cond_dict = dict(zip(cond_indices, cond_values))
        cond_components = self.components.condition(cond_dict, squeeze)
        return Mixture(cond_mixing, cond_components)


class MultivariateNormalMixture(Mixture[MultivariateNormal]):
    def posterior(self, potentials: MultivariateNormal) -> 'MultivariateNormalMixture':
        means = potentials.mean.unsqueeze(1)  # (N, 1, D)
        precs = potentials.precision_matrix.unsqueeze(1)  # (N, 1, D, D)
        covs = potentials.covariance_matrix.unsqueeze(1)  # (N, 1, D, D)

        prior_means = self.components.mean.unsqueeze(0)  # (1, K, D)
        prior_precs = self.components.precision_matrix.unsqueeze(0)  # (1, K, D, D)
        prior_covs = self.components.covariance_matrix.unsqueeze(0)  # (1, K, D, D)

        post_precs = precs + prior_precs
        post_means = posdef_solve(precs @ means[..., None] + prior_precs @ prior_means[..., None],
                                  post_precs)[0].squeeze(-1)
        post_components = MultivariateNormal(post_means, precision_matrix=post_precs)

        post_lognorm = MultivariateNormal(prior_means, covs + prior_covs).log_prob(means)
        post_logits = self.mixing.logits + post_lognorm

        return MultivariateNormalMixture(Categorical(logits=post_logits), post_components)


class NaturalMultivariateNormalMixture(Mixture[NaturalMultivariateNormal]):
    def posterior(self, potentials: Union[NaturalMultivariateNormal, MultivariateNormal]) \
            -> 'NaturalMultivariateNormalMixture':
        if isinstance(potentials, MultivariateNormal):
            potentials = NaturalMultivariateNormal.from_standard(potentials)

        eta1 = potentials.nat_param1.unsqueeze(1)  # (N, 1, D)
        eta2 = potentials.nat_param2.unsqueeze(1)  # (N, 1, D, D)

        prior_eta1 = self.components.nat_param1.unsqueeze(0)  # (1, K, D)
        prior_eta2 = self.components.nat_param2.unsqueeze(0)  # (1, K, D, D)

        post_eta1 = eta1 + prior_eta1
        post_eta2 = eta2 + prior_eta2
        post_components = NaturalMultivariateNormal(post_eta1, post_eta2)

        post_lognorm = post_components.log_normalizer - self.components.log_normalizer
        post_logits = self.mixing.logits + post_lognorm

        return NaturalMultivariateNormalMixture(Categorical(logits=post_logits), post_components)


def eval_grid(xx, yy, fcn):
    xy = torch.stack([xx.flatten(), yy.flatten()], dim=1)
    return fcn(xy).reshape_as(xx)


if __name__ == '__main__':
    from pyro.distributions import Dirichlet

    N, K, D = 200, 4, 2
    props = Dirichlet(5*torch.ones(K)).sample()
    mean = torch.arange(K).float().view(K, 1).expand(K, D)
    var = .1 * torch.eye(D).expand(K, -1, -1)
    mixing = Categorical(props)
    components = MultivariateNormal(mean, var)
    print("mixing", mixing.batch_shape, mixing.event_shape)
    print("components", components.batch_shape, components.event_shape)
    # mixture = MultivariateNormalMixture(mixing, components)
    # mixture = NaturalMultivariateNormalMixture(mixing, NaturalMultivariateNormal.from_standard(components))
    mixture = Mixture(mixing, NaturalMultivariateNormal.from_standard(components))
    mixture.rename(['x', 'y'])
    print("mixture names", mixture.variable_names)
    print("mixture", mixture.batch_shape, mixture.event_shape)
    probe = MultivariateNormal(mean[:3]+1*torch.tensor([1., -1.]), .2 * var[:3])
    post_mixture = mixture.posterior(probe)
    print("post_mixture names", post_mixture.variable_names)
    samples = mixture.sample([N])
    n = 1
    post_samples = post_mixture.sample([N])[:, n]
    print("post_mixture", post_mixture.batch_shape, post_mixture.event_shape)
    print("sample", samples.shape)

    x = torch.linspace(-2, K - 1 + 2, 200)
    y = torch.linspace(-2, K - 1 + 2, 200)
    xx, yy = torch.meshgrid(x, y)
    xy = torch.stack([xx, yy], -1)
    zz = mixture.log_prob(xy)
    post_zz = post_mixture.log_prob(xy[..., None, :])[:, :, 1]
    probe_zz = probe.log_prob(xy[..., None, :])[:, :, 1]

    import matplotlib.pyplot as plt
    plt.imshow(zz.exp().T, interpolation='bilinear', origin='lower',
               extent=[x[0], x[-1], y[0], y[-1]])
    plt.scatter(*samples.T, c='k', s=4)
    plt.contour(xx, yy, probe_zz.exp(), cmap='inferno')
    plt.plot(*mixture.mean, 'wo')
    plt.show()

    plt.imshow(post_zz.exp().T, interpolation='bilinear', origin='lower',
               extent=[x[0], x[-1], y[0], y[-1]])
    plt.scatter(*post_samples.T, c='k', s=2)
    plt.contour(xx, yy, zz.exp(), cmap='inferno')
    plt.contour(xx, yy, probe_zz.exp(), cmap='inferno')
    plt.plot(*post_mixture.mean[1], 'wo')
    plt.show()
    # mixture.posterior(td.Normal(mean, std))
