import torch
import pyro

from pyro.nn import pyro_method

from pyro.distributions import Normal, TransformedDistribution
from pyro.distributions.torch_transform import ComposeTransformModule
from pyro.distributions.transforms import (
    Spline, ExpTransform, ComposeTransform, AffineCoupling,
    GeneralizedChannelPermute, SigmoidTransform
)
from deepscm.distributions.transforms.reshape import ReshapeTransform, SqueezeTransform, TransposeTransform
from deepscm.distributions.transforms.affine import LearnedAffineTransform
from deepscm.arch.mnist import BasicFlowConvNet
from deepscm.distributions.transforms.normalisation import ActNorm

from .base_nf_experiment import BaseFlowSEM, MODEL_REGISTRY


class IndependentFlowSEM(BaseFlowSEM):
    def __init__(self, use_affine_ex: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.use_affine_ex = use_affine_ex

        # decoder parts

        # Flow for modelling t Gamma
        self.thickness_flow_components = ComposeTransformModule([Spline(1)])
        self.thickness_flow_constraint_transforms = ComposeTransform([self.thickness_flow_lognorm, ExpTransform()])
        self.thickness_flow_transforms = ComposeTransform([self.thickness_flow_components, self.thickness_flow_constraint_transforms])

        # affine flow for s normal
        self.intensity_flow_components = ComposeTransformModule([LearnedAffineTransform(), Spline(1)])
        self.intensity_flow_constraint_transforms = ComposeTransform([SigmoidTransform(), self.intensity_flow_norm])
        self.intensity_flow_transforms = [self.intensity_flow_components, self.intensity_flow_constraint_transforms]

        # realnvp or so for x
        self._build_image_flow()

    def _build_image_flow(self):

        self.trans_modules = ComposeTransformModule([])

        self.x_transforms = []

        self.x_transforms += [self._get_preprocess_transforms()]

        c = 1
        for _ in range(self.num_scales):
            self.x_transforms.append(SqueezeTransform())
            c *= 4

            for _ in range(self.flows_per_scale):
                if self.use_actnorm:
                    actnorm = ActNorm(c)
                    self.trans_modules.append(actnorm)
                    self.x_transforms.append(actnorm)

                gcp = GeneralizedChannelPermute(channels=c)
                self.trans_modules.append(gcp)
                self.x_transforms.append(gcp)

                self.x_transforms.append(TransposeTransform(torch.tensor((1, 2, 0))))

                ac = AffineCoupling(c // 2, BasicFlowConvNet(c // 2, self.hidden_channels, (c // 2, c // 2)))
                self.trans_modules.append(ac)
                self.x_transforms.append(ac)

                self.x_transforms.append(TransposeTransform(torch.tensor((2, 0, 1))))

            gcp = GeneralizedChannelPermute(channels=c)
            self.trans_modules.append(gcp)
            self.x_transforms.append(gcp)

        self.x_transforms += [
            ReshapeTransform((4**self.num_scales, 32 // 2**self.num_scales, 32 // 2**self.num_scales), (1, 32, 32))
        ]

    @pyro_method
    def pgm_model(self):
        thickness_base_dist = Normal(self.thickness_base_loc, self.thickness_base_scale).to_event(1)
        thickness_dist = TransformedDistribution(thickness_base_dist, self.thickness_flow_transforms)

        thickness = pyro.sample('thickness', thickness_dist)
        # pseudo call to thickness_flow_transforms to register with pyro
        _ = self.thickness_flow_components

        intensity_base_dist = Normal(self.intensity_base_loc, self.intensity_base_scale).to_event(1)
        intensity_dist = TransformedDistribution(intensity_base_dist, self.intensity_flow_transforms)

        intensity = pyro.sample('intensity', intensity_dist)
        # pseudo call to intensity_flow_transforms to register with pyro
        _ = self.intensity_flow_components

        return thickness, intensity

    @pyro_method
    def model(self):
        thickness, intensity = self.pgm_model()

        x_base_dist = Normal(self.x_base_loc, self.x_base_scale).to_event(3)
        x_dist = TransformedDistribution(x_base_dist, ComposeTransform(self.x_transforms).inv)

        x = pyro.sample('x', x_dist)

        return x, thickness, intensity

    @pyro_method
    def infer_thickness_base(self, thickness):
        return self.thickness_flow_transforms.inv(thickness)

    @pyro_method
    def infer_intensity_base(self, intensity):
        return self.intensity_flow_transforms.inv(intensity)

    @pyro_method
    def infer_x_base(self, thickness, intensity, x):
        return ComposeTransform(self.x_transforms)(x)


MODEL_REGISTRY[IndependentFlowSEM.__name__] = IndependentFlowSEM
