import torch
from torch import nn
import numpy as np

from collections.abc import Iterable


class BasicFlowConvNet(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, param_dims, context_dims: int = None, param_nonlinearities=None):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels

        self.param_dims = param_dims
        self.count_params = len(param_dims)
        self.output_dims = sum(param_dims)

        self.context_dims = context_dims
        self.param_nonlinearities = param_nonlinearities

        self.seq1 = nn.Sequential(
            nn.Conv2d(in_channels + context_dims if context_dims is not None else in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, self.output_dims, kernel_size=3, padding=1)
        )

        ends = torch.cumsum(torch.tensor(param_dims), dim=0)
        starts = torch.cat((torch.zeros(1).type_as(ends), ends[:-1]))
        self.param_slices = [slice(s.item(), e.item()) for s, e in zip(starts, ends)]

        def weights_init(m):
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, 0., 1e-4)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.)

        self.apply(weights_init)

    def forward(self, inputs, context=None):
        # pyro affine coupling splits on the last dimension and not the channel dimension
        # -> we want to permute the dimensions to move the last dimension into the channel dimension
        # and then transpose back

        if not ((self.context_dims is None) == (context is None)):
            raise ValueError('Given context does not match context dims: context: {} and context_dims:{}'.format(context, self.context_dims))

        *batch_dims, h, w, c = inputs.size()
        num_batch = len(batch_dims)

        permutation = np.array((2, 0, 1)) + num_batch
        outputs = inputs.permute(*np.arange(num_batch), *permutation).contiguous()

        if context is not None:
            # assuming scalar inputs [B, C]
            context = context.view(*context.shape, 1, 1).expand(-1, -1, *outputs.shape[2:])
            outputs = torch.cat([outputs, context], 1)


        # print(self.in_channels, self.context_dims,self.hidden_channels)
        # import pdb; pdb.set_trace()
        outputs = self.seq1(outputs.double())

        permutation = np.array((1, 2, 0)) + num_batch
        outputs = outputs.permute(*np.arange(num_batch), *permutation).contiguous()

        if self.count_params > 1:
            outputs = tuple(outputs[..., s] for s in self.param_slices)

        if self.param_nonlinearities is not None:
            if isinstance(self.param_nonlinearities, Iterable):
                outputs = tuple(n(o) for o, n in zip(outputs, self.param_nonlinearities))
            else:
                outputs = tuple(self.param_nonlinearities(o) for o in outputs)

        return outputs


