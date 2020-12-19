import torch
import torch.distributions as td
from torch.distributions import constraints

from .mvt import MultivariateStudentT
from .natural_mvn import NaturalMultivariateNormal
from .wishart import Wishart
from deepscm.util import cholseky_inverse, mahalanobis, matvec, outer, posdef_inverse, triangular_logdet


class _Symmetric(constraints.Constraint):
    """
    Constrain to symmetric matrices.
    """
    def check(self, value):
        return torch.isclose(value, value.transpose(-1, -2)).flatten(-2).all(-1)


def mvdigamma(x: torch.Tensor, p: int) -> torch.Tensor:
    i = torch.arange(p, dtype=x.dtype, device=x.device)
    return torch.digamma(x.unsqueeze(-1) - .5 * i).sum(-1)


def _validate_std_params(mean, nu, a, B):
    D = mean.shape[-1]
    if (a <= .5 * (D - 1)).any():
        raise ValueError(f"Parameter 'a' is too small ({a:.2f} <= {.5 * (D - 1):.2f})")
    if not constraints.positive_definite.check(B).all():
        raise ValueError("Parameter 'B' is not positive definite")
    if (nu <= 0).any():
        raise ValueError(f"Parameter 'nu' must be positive ({nu:.2f} < 0)")


class NaturalNormalWishart(td.Distribution):
    arg_constraints = {'lambda1': constraints.real_vector,
                       'lambda2': _Symmetric(),
                       'nu': constraints.positive}
    has_rsample = False

    def __init__(self, dof, lambda1, lambda2, nu, validate_args=None):
        self.dof = dof
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.nu = nu
        batch_shape, event_shape = lambda1.shape[:-1], lambda1.shape[-1:]
        D = event_shape[0]
        self.arg_constraints['dof'] = constraints.greater_than(2. * D + 1.)
        super().__init__(batch_shape, event_shape, validate_args=validate_args)

    @staticmethod
    def from_standard(mean, nu, a, B, validate_args=None) -> 'NaturalNormalWishart':
        D = mean.shape[-1]
        if validate_args:
            _validate_std_params(mean, nu, a, B)
        dof = 2. * a + D + 2.
        lambda1 = nu.unsqueeze(-1) * mean
        lambda2 = 2. * B + outer(lambda1, mean)
        return NaturalNormalWishart(dof, lambda1, lambda2, nu, validate_args=validate_args)

    def to_standard(self):
        D = self.event_shape[0]
        mean = self.lambda1 / self.nu.unsqueeze(-1)
        a = .5 * (self.dof - D) - 1.
        B = .5 * (self.lambda2 - outer(self.lambda1, mean))
        if self._validate_args:
            _validate_std_params(mean, self.nu, a, B)
        return mean, self.nu, a, B

    def rsample(self, sample_shape=torch.Size()) -> NaturalMultivariateNormal:
        # TODO: Test NW sampling
        prior_mean, nu, a, B = self.to_standard()
        prec = Wishart(a, B).rsample(sample_shape)
        mean = td.MultivariateNormal(prior_mean, precision_matrix=nu * prec).rsample(sample_shape)

        eta2 = -.5 * prec
        eta1 = matvec(prec, mean)
        return NaturalMultivariateNormal(eta1, eta2, validate_args=self._validate_args)

    @property
    def mean(self) -> NaturalMultivariateNormal:
        mean, nu, a, B = self.to_standard()
        expec_eta2 = -.5 * a * posdef_inverse(B)[0]
        expec_eta1 = -2. * matvec(expec_eta2, mean)
        return NaturalMultivariateNormal(expec_eta1, expec_eta2, validate_args=self._validate_args)

    def expected_stats(self):
        mean, nu, a, B = self.to_standard()
        D = mean.shape[-1]
        maha, B_tril = mahalanobis(B, mean)
        logdet_B = 2. * triangular_logdet(B_tril)
        expec_lambda2 = -.5 * a[..., None, None] * cholseky_inverse(B_tril)
        expec_lambda1 = -2. * matvec(expec_lambda2, mean)
        expec_log_norm = .5 * (a * maha + D / nu + mvdigamma(a, D) - logdet_B)
        return expec_lambda1, expec_lambda2, expec_log_norm

    def _suff_stats(self, data: torch.Tensor, weights: torch.Tensor = None):
        sample_shape = data.shape[:-len(self.event_shape)]
        expected_shape = sample_shape + self.batch_shape
        if weights is None:
            weights = torch.tensor(1., device=data.device).expand(expected_shape)
        elif weights.shape != expected_shape:
            raise ValueError(f"Expected weights shape {expected_shape}, got {weights.shape}")
        sample_size = sample_shape.numel()
        batch_size = self.batch_shape.numel()
        event_size = self.event_shape.numel()
        weights = weights.view(sample_size, batch_size)  # (N, K)
        data = data.view(sample_size, event_size)  # (N, D)
        N = weights.sum(0).squeeze(0)
        t1 = torch.einsum('nk,nd->kd', weights, data).squeeze(0)
        t2 = torch.einsum('nk,nd,ne->kde', weights, data, data).squeeze(0)
        return N, t1, t2

    def posterior(self, data: torch.Tensor, weights: torch.Tensor = None):
        N, t1, t2 = self._suff_stats(data, weights)
        nu_ = self.nu + N
        dof_ = self.dof + N
        lambda1_ = self.lambda1 + t1
        lambda2_ = self.lambda2 + t2
        return NaturalNormalWishart(dof_, lambda1_, lambda2_, nu_)

    def predictive(self):
        mean, nu, a, B = self.to_standard()
        D = mean.shape[-1]
        dof = 2. * a - D + 1.
        scale = 2. * (nu + 1.) / (nu * dof) * B
        return MultivariateStudentT(dof, mean, scale, validate_args=self._validate_args)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(NaturalNormalWishart, _instance)
        batch_shape = torch.Size(batch_shape)
        scalar_shape = batch_shape
        lambda1_shape = batch_shape + self.event_shape
        lambda2_shape = batch_shape + self.event_shape + self.event_shape
        new.dof = self.dof.expand(scalar_shape)
        new.nu = self.nu.expand(scalar_shape)
        new.lambda1 = self.lambda1.expand(lambda1_shape)
        new.lambda2 = self.lambda2.expand(lambda2_shape)
        super(NaturalNormalWishart, new).__init__(batch_shape, self.event_shape,
                                                  validate_args=False)
        new._validate_args = self._validate_args
        return new


if __name__ == '__main__':
    N, K, D = 1000, 3, 2
    mean = torch.zeros(D)
    nu = torch.tensor(1.)
    a = torch.tensor(float(D) - 1.)
    B = torch.eye(D)
    niw = NaturalNormalWishart.from_standard(mean, nu, a, B)
    data = torch.randn(N, D)
    post_niw = niw.posterior(data)
    print(post_niw.to_standard())
    mix = td.Dirichlet(torch.ones(K))
    weights = mix.sample((N,))
    expanded_niw = niw.expand((K,))
    post_niw = expanded_niw.posterior(data)
    print(post_niw.to_standard())
    post_niw = expanded_niw.posterior(data, weights)
    print(post_niw.to_standard())

    samples = niw.rsample((K,))
    print(samples)
    print(samples.batch_shape, samples.event_shape)
