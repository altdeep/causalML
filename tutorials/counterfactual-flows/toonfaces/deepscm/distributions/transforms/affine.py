from pyro.distributions.conditional import ConditionalTransformModule
from pyro.distributions.torch_transform import TransformModule
from pyro.distributions import transforms as pyro_transforms
from torch.distributions import transforms

import torch


class LearnedAffineTransform(TransformModule, transforms.AffineTransform):
    def __init__(self, loc=None, scale=None, **kwargs):

        super().__init__(loc=loc, scale=scale, **kwargs)

        if loc is None:
            self.loc = torch.nn.Parameter(torch.zeros([1, ]))
        if scale is None:
            self.scale = torch.nn.Parameter(torch.ones([1, ]))

    def _broadcast(self, val):
        dim_extension = tuple(1 for _ in range(val.dim() - 1))
        loc = self.loc.view(-1, *dim_extension)
        scale = self.scale.view(-1, *dim_extension)

        return loc, scale

    def _call(self, x):
        loc, scale = self._broadcast(x)

        return loc + scale * x

    def _inverse(self, y):
        loc, scale = self._broadcast(y)
        return (y - loc) / scale


class ConditionalAffineTransform(ConditionalTransformModule):
    def __init__(self, context_nn, event_dim=0, **kwargs):
        super().__init__(**kwargs)

        self.event_dim = event_dim
        self.context_nn = context_nn

    def condition(self, context):
        loc, log_scale = self.context_nn(context)
        scale = torch.exp(log_scale)

        ac = transforms.AffineTransform(loc, scale, event_dim=self.event_dim)
        return ac


class LowerCholeskyAffine(pyro_transforms.LowerCholeskyAffine):
    def log_abs_det_jacobian(self, x, y):
        """
        Calculates the elementwise determinant of the log Jacobian, i.e.
        log(abs(dy/dx)).
        """
        return torch.ones(x.size()[:-1], dtype=x.dtype, layout=x.layout, device=x.device) * \
            self.scale_tril.diagonal(dim1=-2, dim2=-1).log().sum(-1).sum(-1)
