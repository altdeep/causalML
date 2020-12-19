
import torch
from torch import nn
import numpy as np

from collections.abc import Iterable

class SEMRealNVP(nn.Module):
    def __init__(self,
                 # these are architecture-specific
                 num_scales: int = 4, flows_per_scale: int = 2, hidden_channels: int = 256, use_actnorm: bool = False
                 # these are data- and model- dependent
                 img_height: int, img_width: int, img_channels:int, num_pgm_features:int, use_affine_ex:bool
                 ):
        super().__init__()
        # these are data- and model- dependent
        self.img_height = img_height             # img_height = Toonfaces.height
        self.img_width = img_width               # img_width = Toonfaces.width
        self.img_channels = img_channels         # img_channels = Toonfaces.channels
        self.num_pgm_features = num_pgm_features # num_pgm_features = len(THE_FEATURES.keys())
        self.use_affine_ex = use_affine_ex

        # these are architecture-specific
        self.num_scales = num_scales
        self.flows_per_scale = flows_per_scale
        self.hidden_channels = hidden_channels
        self.use_actnorm = use_actnorm

    def _build_image_flow(self):
        self.trans_modules = ComposeTransformModule([])
        self.x_transforms = []
        self.x_transforms += [self._get_preprocess_transforms()]
        c = self.img_channels
        for n in range(self.num_scales):
            self.x_transforms.append(SqueezeTransform())
            c *= 4

            for i in range(self.flows_per_scale): # 0,1,2,3
                if self.use_actnorm:
                    actnorm = ActNorm(c)
                    self.trans_modules.append(actnorm)
                    self.x_transforms.append(actnorm)

                gcp = GeneralizedChannelPermute(channels=c)
                self.x_transforms.append(PrintTransform("before GCP in flows-per-scale" + str(i)))
                self.trans_modules.append(gcp)
                self.x_transforms.append(gcp)

                self.x_transforms.append(PrintTransform("before TransposeTransform" + str(i)))
                self.x_transforms.append(TransposeTransform(torch.tensor((1, 2, 0))))
                self.x_transforms.append(PrintTransform("before ConditionalAffineCoupling" + str(i)))

                ac = ConditionalAffineCoupling(c // 2, BasicFlowConvNet(c // 2, self.hidden_channels, (c // 2, c // 2), num_pgm_features))
                self.trans_modules.append(ac)
                self.x_transforms.append(ac)

                self.x_transforms.append(PrintTransform("last_in_flows-per-scale" + str(i)))
                self.x_transforms.append(TransposeTransform(torch.tensor((2, 0, 1))))

            self.x_transforms.append(PrintTransform("last_in_num-scale" + str(n)))
            gcp = GeneralizedChannelPermute(channels=c)
            self.trans_modules.append(gcp)
            self.x_transforms.append(gcp)

        self.x_transforms.append(PrintTransform("second last thing ever"))
        self.x_transforms += [
            ReshapeTransform((img_channels * (4**self.num_scales), img_height // 2**self.num_scales, img_width // 2**self.num_scales), (img_channels, img_height, img_width)) # was (1, 32, 32)
        ]
        self.x_transforms.append(PrintTransform("last thing ever"))

        if self.use_affine_ex:
            affine_net = DenseNN(num_pgm_features, [img_height // 2, img_width // 2], param_dims=[1, 1]) # was 2 and [16, 16]
            affine_trans = ConditionalAffineTransform(context_nn=affine_net, event_dim=3)

            self.trans_modules.append(affine_trans)
            self.x_transforms.append(affine_trans)

