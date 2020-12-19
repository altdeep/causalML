# adapted from https://github.com/bayesiains/nsf/blob/7433f3fab7861ee57244655170f40e95194fa139/nde/transforms/reshape.py#L7
import torch
from torch.distributions.utils import lazy_property
from torch.distributions import constraints
from torch.distributions.transforms import Transform
import numpy as np


class PrintTransform(Transform):
    codomain = constraints.real
    bijective = True
    event_dim = 3
    volume_preserving = True

    def __init__(self, pre, debug=False):
        super().__init__()
        self.pre = pre
        self.debug = debug

    def _call(self, inputs):
        if self.debug:
            print("[call]", self.pre, inputs.shape)
        return inputs

    def _inverse(self, inputs):
        if self.debug:
            print("[inverse]", self.pre, inputs.shape)
        return inputs

    def log_abs_det_jacobian(self, x, y):
        #print(x.shape, y.shape)
        return 1

    def get_output_shape(self, c, h, w):
        return (c, h, w)


class SqueezeTransform(Transform):
    """A transformation defined for image data that trades spatial dimensions for channel
    dimensions, i.e. "squeezes" the inputs along the channel dimensions.
    Implementation adapted from https://github.com/pclucas14/pytorch-glow and
    https://github.com/chaiyujin/glow-pytorch.
    Reference:
    > L. Dinh et al., Density estimation using Real NVP, ICLR 2017.
    """

    codomain = constraints.real
    bijective = True
    event_dim = 3
    volume_preserving = True

    def __init__(self, factor=2):
        super().__init__(cache_size=1)

        self.factor = factor

    def _call(self, inputs):
        """
        :param x: the input into the bijection
        :type x: torch.Tensor

        Invokes the bijection x=>y; in the prototypical context of a
        :class:`~pyro.distributions.TransformedDistribution` `x` is a sample from
        the base distribution (or the output of a previous transform)
        """
        if inputs.dim() < 3:
            raise ValueError(f'Expecting inputs with at least 3 dimensions, got {inputs.shape} - {inputs.dim()}')

        *batch_dims, c, h, w = inputs.size()
        num_batch = len(batch_dims)

        if h % self.factor != 0 or w % self.factor != 0:
            breakpoint()
            raise ValueError('Input image size not compatible with the factor.')

        inputs = inputs.view(*batch_dims, c, h // self.factor, self.factor, w // self.factor,
                             self.factor)
        permute = np.array((0, 2, 4, 1, 3)) + num_batch
        inputs = inputs.permute(*np.arange(num_batch), *permute).contiguous()
        inputs = inputs.view(*batch_dims, c * self.factor * self.factor, h // self.factor,
                             w // self.factor)

        return inputs

    def _inverse(self, inputs):
        """
        :param y: the output of the bijection
        :type y: torch.Tensor

        Inverts y => x.
        """
        if inputs.dim() < 3:
            raise ValueError(f'Expecting inputs with at least 3 dimensions, got {inputs.shape}')

        *batch_dims, c, h, w = inputs.size()
        num_batch = len(batch_dims)

        if c < 4 or c % 4 != 0:
            raise ValueError('Invalid number of channel dimensions.')

        inputs = inputs.view(*batch_dims, c // self.factor ** 2, self.factor, self.factor, h, w)
        permute = np.array((0, 3, 1, 4, 2)) + num_batch
        inputs = inputs.permute(*np.arange(num_batch), *permute).contiguous()
        inputs = inputs.view(*batch_dims, c // self.factor ** 2, h * self.factor, w * self.factor)

        return inputs

    def log_abs_det_jacobian(self, x, y):
        """
        Calculates the elementwise determinant of the log Jacobian, i.e.
        log(abs([dy_0/dx_0, ..., dy_{N-1}/dx_{N-1}])). Note that this type of
        transform is not autoregressive, so the log Jacobian is not the sum of the
        previous expression. However, it turns out it's always 0 (since the
        determinant is -1 or +1), and so returning a vector of zeros works.
        """

        log_abs_det_jacobian = torch.zeros(x.size()[:-3], dtype=x.dtype, layout=x.layout, device=x.device)
        return log_abs_det_jacobian

    def get_output_shape(self, c, h, w):
        return (c * self.factor * self.factor,
                h // self.factor,
                w // self.factor)


class ReshapeTransform(Transform):
    codomain = constraints.real
    bijective = True
    volume_preserving = True

    def __init__(self, input_shape, output_shape):
        super().__init__()
        self.event_dim = len(input_shape)
        self.inv_event_dim = len(output_shape)
        self.input_shape = input_shape
        self.output_shape = output_shape

    def _call(self, inputs):
        """
        :param x: the input into the bijection
        :type x: torch.Tensor

        Invokes the bijection x=>y; in the prototypical context of a
        :class:`~pyro.distributions.TransformedDistribution` `x` is a sample from
        the base distribution (or the output of a previous transform)
        """
        batch_dims = inputs.shape[:-self.event_dim]
        inp_shape = inputs.shape[-self.event_dim:]
        if inp_shape != self.input_shape:
            raise RuntimeError('Unexpected inputs shape ({}, but expecting {})'
                               .format(inp_shape, self.input_shape))
        return inputs.reshape(*batch_dims, *self.output_shape)

    def _inverse(self, inputs):
        """
        :param y: the output of the bijection
        :type y: torch.Tensor

        Inverts y => x.
        """
        batch_dims = inputs.shape[:-self.inv_event_dim]
        inp_shape = inputs.shape[-self.inv_event_dim:]
        if inp_shape != self.output_shape:
            raise RuntimeError('Unexpected inputs shape ({}, but expecting {})'
                               .format(inp_shape, self.output_shape))
        return inputs.reshape(*batch_dims, *self.input_shape)

    def log_abs_det_jacobian(self, x, y):
        """
        Calculates the elementwise determinant of the log Jacobian, i.e.
        log(abs([dy_0/dx_0, ..., dy_{N-1}/dx_{N-1}])). Note that this type of
        transform is not autoregressive, so the log Jacobian is not the sum of the
        previous expression. However, it turns out it's always 0 (since the
        determinant is -1 or +1), and so returning a vector of zeros works.
        """

        log_abs_det_jacobian = torch.zeros(x.size()[:-self.event_dim], dtype=x.dtype, layout=x.layout, device=x.device)
        return log_abs_det_jacobian


class TransposeTransform(Transform):
    """
    A bijection that reorders the input dimensions, that is, multiplies the input by
    a permutation matrix. This is useful in between
    :class:`~pyro.distributions.transforms.AffineAutoregressive` transforms to
    increase the flexibility of the resulting distribution and stabilize learning.
    Whilst not being an autoregressive transform, the log absolute determinate of
    the Jacobian is easily calculable as 0. Note that reordering the input dimension
    between two layers of
    :class:`~pyro.distributions.transforms.AffineAutoregressive` is not equivalent
    to reordering the dimension inside the MADE networks that those IAFs use; using
    a :class:`~pyro.distributions.transforms.Permute` transform results in a
    distribution with more flexibility.

    Example usage:

    >>> from pyro.nn import AutoRegressiveNN
    >>> from pyro.distributions.transforms import AffineAutoregressive, Permute
    >>> base_dist = dist.Normal(torch.zeros(10), torch.ones(10))
    >>> iaf1 = AffineAutoregressive(AutoRegressiveNN(10, [40]))
    >>> ff = Permute(torch.randperm(10, dtype=torch.long))
    >>> iaf2 = AffineAutoregressive(AutoRegressiveNN(10, [40]))
    >>> flow_dist = dist.TransformedDistribution(base_dist, [iaf1, ff, iaf2])
    >>> flow_dist.sample()  # doctest: +SKIP

    :param permutation: a permutation ordering that is applied to the inputs.
    :type permutation: torch.LongTensor

    """

    codomain = constraints.real
    bijective = True
    volume_preserving = True

    def __init__(self, permutation):
        super().__init__(cache_size=1)

        self.event_dim = len(permutation)
        self.permutation = permutation

    @lazy_property
    def inv_permutation(self):
        result = torch.empty_like(self.permutation, dtype=torch.long)
        result[self.permutation] = torch.arange(self.permutation.size(0),
                                                dtype=torch.long,
                                                device=self.permutation.device)
        return result

    def _call(self, x):
        """
        :param x: the input into the bijection
        :type x: torch.Tensor

        Invokes the bijection x=>y; in the prototypical context of a
        :class:`~pyro.distributions.TransformedDistribution` `x` is a sample from
        the base distribution (or the output of a previous transform)
        """

        *batch_dims, c, h, w = x.size()
        num_batch = len(batch_dims)

        return x.permute(*np.arange(num_batch), *(self.permutation + num_batch)).contiguous()

    def _inverse(self, y):
        """
        :param y: the output of the bijection
        :type y: torch.Tensor

        Inverts y => x.
        """

        *batch_dims, c, h, w = y.size()
        num_batch = len(batch_dims)

        return y.permute(*np.arange(num_batch), *(self.inv_permutation + num_batch)).contiguous()

    def log_abs_det_jacobian(self, x, y):
        """
        Calculates the elementwise determinant of the log Jacobian, i.e.
        log(abs([dy_0/dx_0, ..., dy_{N-1}/dx_{N-1}])). Note that this type of
        transform is not autoregressive, so the log Jacobian is not the sum of the
        previous expression. However, it turns out it's always 0 (since the
        determinant is -1 or +1), and so returning a vector of zeros works.
        """

        log_abs_det_jacobian = torch.zeros(x.size()[:-self.event_dim], dtype=x.dtype, layout=x.layout, device=x.device)
        return log_abs_det_jacobian
