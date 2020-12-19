# adapted from https://github.com/bayesiains/nsf/blob/7433f3fab7861ee57244655170f40e95194fa139/nde/transforms/normalization.py#L134
import torch
from torch import nn
from torch.distributions import constraints
from pyro.distributions.torch_transform import TransformModule


class ActNorm(TransformModule):
    codomain = constraints.real
    bijective = True
    event_dim = 3

    def __init__(self, features):
        """
        Transform that performs activation normalization. Works for 2D and 4D inputs. For 4D
        inputs (images) normalization is performed per-channel, assuming BxCxHxW input shape.

        Reference:
        > D. Kingma et. al., Glow: Generative flow with invertible 1x1 convolutions, NeurIPS 2018.
        """
        super().__init__()

        self.initialized = False
        self.log_scale = nn.Parameter(torch.zeros(features))
        self.shift = nn.Parameter(torch.zeros(features))

    @property
    def scale(self):
        return torch.exp(self.log_scale)

    def _broadcastable_scale_shift(self, inputs):
        if inputs.dim() == 4:
            return self.scale.view(1, -1, 1, 1), self.shift.view(1, -1, 1, 1)
        else:
            return self.scale.view(1, -1), self.shift.view(1, -1)

    def _call(self, x):
        if x.dim() not in [2, 4]:
            raise ValueError("Expecting inputs to be a 2D or a 4D tensor.")

        if self.training and not self.initialized:
            self._initialize(x)

        scale, shift = self._broadcastable_scale_shift(x)
        outputs = scale * x + shift

        return outputs

    def _inverse(self, y):
        if y.dim() not in [2, 4]:
            raise ValueError("Expecting inputs to be a 2D or a 4D tensor.")

        scale, shift = self._broadcastable_scale_shift(y)
        outputs = (y - shift) / scale

        return outputs

    def log_abs_det_jacobian(self, x, y):
        """
        Calculates the elementwise determinant of the log Jacobian, i.e.
        log(abs([dy_0/dx_0, ..., dy_{N-1}/dx_{N-1}])). Note that this type of
        transform is not autoregressive, so the log Jacobian is not the sum of the
        previous expression. However, it turns out it's always 0 (since the
        determinant is -1 or +1), and so returning a vector of zeros works.
        """

        ones = torch.ones(x.shape[0], device=x.device)
        if x.dim() == 4:
            _, _, h, w = x.shape
            log_abs_det_jacobian = h * w * torch.sum(self.log_scale) * ones
        else:
            log_abs_det_jacobian = torch.sum(self.log_scale) * ones

        return log_abs_det_jacobian

    def _initialize(self, inputs):
        """Data-dependent initialization, s.t. post-actnorm activations have zero mean and unit
        variance. """
        if inputs.dim() == 4:
            num_channels = inputs.shape[1]
            inputs = inputs.permute(0, 2, 3, 1).reshape(-1, num_channels)

        with torch.no_grad():
            std = inputs.std(dim=0)
            mu = (inputs / std).mean(dim=0)
            self.log_scale.data = -torch.log(std)
            self.shift.data = -mu

        self.initialized = True
