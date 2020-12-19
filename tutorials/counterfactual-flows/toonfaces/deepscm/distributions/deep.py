import torch
from pyro.distributions import (
    Bernoulli, Beta, Gamma, Independent, MultivariateNormal,
    Normal, TorchDistribution, LowRankMultivariateNormal
)
from torch import nn

from .params import MixtureParams


class DeepConditional(nn.Module):
    def predict(self, x: torch.Tensor) -> TorchDistribution:
        raise NotImplementedError


class _DeepIndepNormal(DeepConditional):
    def __init__(self, backbone: nn.Module, mean_head: nn.Module, logvar_head: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.mean_head = mean_head
        self.logvar_head = logvar_head

    def forward(self, x):
        h = self.backbone(x)
        mean = self.mean_head(h)
        logvar = self.logvar_head(h)
        return mean, logvar

    def predict(self, x) -> Independent:
        mean, logvar = self(x)
        std = (.5 * logvar).exp()
        event_ndim = len(mean.shape[1:])  # keep only batch dimension
        return Normal(mean, std).to_event(event_ndim)


class DeepIndepNormal(_DeepIndepNormal):
    def __init__(self, backbone: nn.Module, hidden_dim: int, out_dim: int):
        super().__init__(
            backbone=backbone,
            mean_head=nn.Linear(hidden_dim, out_dim),
            logvar_head=nn.Linear(hidden_dim, out_dim)
        )


class Conv2dIndepNormal(_DeepIndepNormal):
    def __init__(self, backbone: nn.Module, hidden_channels: int, out_channels: int = 1):
        super().__init__(
            backbone=backbone,
            mean_head=nn.Conv2d(hidden_channels, out_channels=out_channels, kernel_size=1),
            logvar_head=nn.Conv2d(hidden_channels, out_channels=out_channels, kernel_size=1)
        )


class Conv3dIndepNormal(_DeepIndepNormal):
    def __init__(self, backbone: nn.Module, hidden_channels: int, out_channels: int = 1):
        super().__init__(
            backbone=backbone,
            mean_head=nn.Conv3d(hidden_channels, out_channels=out_channels, kernel_size=1),
            logvar_head=nn.Conv3d(hidden_channels, out_channels=out_channels, kernel_size=1)
        )


def _assemble_tril(diag: torch.Tensor, lower_vec: torch.Tensor) -> torch.Tensor:
    dim = diag.shape[-1]
    L = torch.diag_embed(diag)  # L is lower-triangular
    i, j = torch.tril_indices(dim, dim, offset=-1)
    L[..., i, j] = lower_vec
    return L


class DeepMultivariateNormal(DeepConditional):
    def __init__(self, backbone: nn.Module, hidden_dim: int, latent_dim: int):
        super().__init__()
        self.backbone = backbone
        cov_lower_dim = (latent_dim * (latent_dim - 1)) // 2
        self.mean_head = nn.Linear(hidden_dim, latent_dim)
        self.lower_head = nn.Linear(hidden_dim, cov_lower_dim)
        self.logdiag_head = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        h = self.backbone(x)
        mean = self.mean_head(h)
        diag = self.logdiag_head(h).exp()
        lower = self.lower_head(h)
        scale_tril = _assemble_tril(diag, lower)
        return mean, scale_tril

    def predict(self, x) -> MultivariateNormal:
        mean, scale_tril = self(x)
        return MultivariateNormal(mean, scale_tril=scale_tril)


class DeepLowRankMultivariateNormal(DeepConditional):
    def __init__(self, backbone: nn.Module, hidden_dim: int, latent_dim: int, rank: int):
        super().__init__()
        self.backbone = backbone

        self.latent_dim = latent_dim
        self.rank = rank

        self.mean_head = nn.Linear(hidden_dim, latent_dim)
        self.factor_head = nn.Linear(hidden_dim, latent_dim * rank)
        self.logdiag_head = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        h = self.backbone(x)
        mean = self.mean_head(h)
        diag = self.logdiag_head(h).exp()
        factors = self.factor_head(h).view(x.shape[0], self.latent_dim, self.rank)

        return mean, diag, factors

    def predict(self, x) -> LowRankMultivariateNormal:
        mean, diag, factors = self(x)
        return LowRankMultivariateNormal(mean, factors, diag)


class MixtureSIN(DeepConditional):
    def __init__(self, encoder: DeepConditional, mixture_params: MixtureParams):
        super().__init__()
        self.encoder = encoder
        self.mixture_params = mixture_params

    def predict(self, data) -> TorchDistribution:
        potentials = self.encoder.predict(data)
        mixture = self.mixture_params.get_distribution()
        posteriors = mixture.posterior(potentials)  # q(latents | data)
        return posteriors


class _DeepIndepGamma(DeepConditional):
    def __init__(self, backbone: nn.Module, rate_head: nn.Module, conc_head: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.rate_head = nn.Sequential(rate_head, nn.Softplus())
        self.conc_head = nn.Sequential(conc_head, nn.Softplus())

    def forward(self, x):
        h = self.backbone(x)
        rate = self.rate_head(h)
        conc = self.conc_head(h)
        return rate, conc

    def predict(self, x) -> Independent:
        rate, conc = self(x)
        event_ndim = len(rate.shape[1:])  # keep only batch dimension
        return Gamma(rate, conc).to_event(event_ndim)


class DeepIndepGamma(_DeepIndepGamma):
    def __init__(self, backbone: nn.Module, hidden_dim: int, out_dim: int):
        super().__init__(
            backbone=backbone,
            rate_head=nn.Linear(hidden_dim, out_dim),
            conc_head=nn.Linear(hidden_dim, out_dim)
        )


class _DeepIndepBeta(DeepConditional):
    def __init__(self, backbone: nn.Module, alpha_head: nn.Module, beta_head: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.alpha_head = nn.Sequential(alpha_head, nn.Softplus())
        self.beta_head = nn.Sequential(beta_head, nn.Softplus())

    def forward(self, x):
        h = self.backbone(x)
        alpha = self.alpha_head(h)
        beta = self.beta_head(h)
        return alpha, beta

    def predict(self, x) -> Independent:
        alpha, beta = self(x)
        event_ndim = len(alpha.shape[1:])  # keep only batch dimension
        return Beta(alpha, beta).to_event(event_ndim)


class DeepIndepBeta(_DeepIndepBeta):
    def __init__(self, backbone: nn.Module, hidden_dim: int, out_dim: int):
        super().__init__(
            backbone=backbone,
            alpha_head=nn.Linear(hidden_dim, out_dim),
            beta_head=nn.Linear(hidden_dim, out_dim)
        )


class Conv2dIndepBeta(_DeepIndepBeta):
    def __init__(self, backbone: nn.Module, hidden_channels: int = 1, out_channels: int = 1):
        super().__init__(
            backbone=backbone,
            alpha_head=nn.Conv2d(hidden_channels, out_channels=out_channels, kernel_size=1),
            beta_head=nn.Conv2d(hidden_channels, out_channels=out_channels, kernel_size=1)
        )


class Conv3dIndepBeta(_DeepIndepBeta):
    def __init__(self, backbone: nn.Module, hidden_channels: int = 1, out_channels: int = 1):
        super().__init__(
            backbone=backbone,
            alpha_head=nn.Conv3d(hidden_channels, out_channels=out_channels, kernel_size=1),
            beta_head=nn.Conv3d(hidden_channels, out_channels=out_channels, kernel_size=1)
        )


class DeepBernoulli(DeepConditional):
    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone

    def forward(self, z):
        logits = self.backbone(z)
        return logits

    def predict(self, z) -> Independent:
        logits = self(z)
        event_ndim = len(logits.shape[1:])  # keep only batch dimension
        return Bernoulli(logits=logits).to_event(event_ndim)


if __name__ == '__main__':
    import torch
    from arch import mnist

    hidden_dim = 10
    latent_dim = 10
    encoder = DeepIndepNormal(mnist.Encoder(hidden_dim), hidden_dim, latent_dim)
    x = torch.randn(5, 1, 28, 28)
    post = encoder.predict(x)
    print(post.batch_shape, post.event_shape)

    # decoder = Conv2dIndepNormal(mnist.Decoder(latent_dim), 1, 1)
    decoder = DeepBernoulli(mnist.Decoder(latent_dim))
    latents = post.rsample()
    print(latents.shape)
    recon = decoder.predict(latents)
    print(recon.batch_shape, recon.event_shape)

    # from distributions import params
    #
    # num_clusters = 4
    # sin = MixtureSIN(
    #     DeepMultivariateNormal(mnist.Encoder(hidden_dim), hidden_dim, latent_dim),
    #     params.MixtureParams(
    #         params.CategoricalParams(num_clusters),
    #         params.MultivariateNormalParams(latent_dim, (num_clusters,))
    #     )
    # )
    #
    # print(sin)
    #
    # post = sin.predict(x)
    # print(post)
