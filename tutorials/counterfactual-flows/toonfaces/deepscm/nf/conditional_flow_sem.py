import torch
import pyro
from torch import nn

from pyro.nn import pyro_method
from pyro.primitives import deterministic

# EDIT: import Gumbel and Delta distributions
from pyro.distributions import Normal, Gumbel, Delta, TransformedDistribution
from pyro.distributions.torch_transform import ComposeTransformModule
from pyro.distributions.conditional import ConditionalTransformedDistribution
from pyro.distributions.transforms import (
    Spline, ExpTransform, ComposeTransform, ConditionalAffineCoupling,
    GeneralizedChannelPermute, SigmoidTransform
)
from pyro.distributions.relaxed_straight_through import RelaxedOneHotCategoricalStraightThrough
from deepscm.distributions.transforms.reshape import ReshapeTransform, SqueezeTransform, TransposeTransform, PrintTransform
from deepscm.distributions.transforms.affine import ConditionalAffineTransform
from deepscm.distributions.transforms.normalisation import ActNorm

from deepscm.arch.toonfaces import BasicFlowConvNet
from deepscm.datasets.toonfaces import Toonfaces
from pyro.nn import DenseNN

from .base_nf_experiment import BaseFlowSEM, MODEL_REGISTRY
from deepscm.experiments.toonfaces.types import THE_FEATURES
from deepscm.util import infer_gumbel_noise


class ConditionalFlowSEM(BaseFlowSEM):
    def __init__(self, use_affine_ex: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.use_affine_ex = use_affine_ex

        # EDIT: register parameters for the model, the shapes of which are specific to the assumed SCM
        # FIXME: in extensible versions of this code, we will need to provide a flexible way to store conditional dependencies
        # between variables in the SCM and then build these parameters based on the that stored information, vs set them manually.
        # Such an extension must also track the dimension dependency for forward sampling and abduction purposes.

        # has a dag
        self.register_parameter('face_color_logits', nn.Parameter(torch.zeros([11, ]))) # p(face_color) where there are 11 values for face_color
        self.register_parameter('hair_logits', nn.Parameter(torch.zeros([111, ]))) # p(hair) where there are 111 values for hair
        self.register_parameter('hair_color_logits', nn.Parameter(torch.zeros([11, 111, 10, ]))) # p(hair_color|face_color, hair) where there are 10 values of hair_color
        self.register_parameter('facial_hair_logits', nn.Parameter(torch.zeros([11, 111, 15, ]))) # p(facial_hair|face_color, hair) where there are 15 values of facial_hair

        # self.register_parameter('face_color_logits', nn.Parameter(torch.zeros([11, ]))) # p(face_color) where there are 11 values for face_color
        # self.register_parameter('hair_logits', nn.Parameter(torch.zeros([111, ]))) # p(hair) where there are 111 values for hair
        # self.register_parameter('hair_color_logits', nn.Parameter(torch.zeros([10, ]))) # p(hair_color|face_color, hair) where there are 10 values of hair_color
        # self.register_parameter('facial_hair_logits', nn.Parameter(torch.zeros([15, ]))) # p(facial_hair|face_color, hair) where there are 15 values of facial_hair

        # decoder parts

        # # Flow for modelling t Gamma
        # self.thickness_flow_components = ComposeTransformModule([Spline(1)])
        # self.thickness_flow_constraint_transforms = ComposeTransform([self.thickness_flow_lognorm, ExpTransform()])
        # self.thickness_flow_transforms = ComposeTransform([self.thickness_flow_components, self.thickness_flow_constraint_transforms])

        # # affine flow for s normal
        # intensity_net = DenseNN(1, [1], param_dims=[1, 1], nonlinearity=torch.nn.Identity())
        # self.intensity_flow_components = ConditionalAffineTransform(context_nn=intensity_net, event_dim=0)
        # self.intensity_flow_constraint_transforms = ComposeTransform([SigmoidTransform(), self.intensity_flow_norm])
        # self.intensity_flow_transforms = [self.intensity_flow_components, self.intensity_flow_constraint_transforms]
        # # build flow as s_affine_w * t * e_s + b -> depends on t though

    @pyro_method
    def pgm_model(self):
        '''
          "facial_hair", 15   <<< gender
          "face_color", 11     <<< race
          "hair", 111         <<< gender
          "hair_color", 10     <<< age
        '''

        # sample from the noise distributions
        # hair_base_dist = RelaxedOneHotCategoricalStraightThrough(self.hair_base_loc, self.hair_base_scale).to_event(1)
        hair_base_dist = Gumbel(self.hair_base_loc, self.hair_base_scale).to_event(1)
        face_color_base_dist = Gumbel(self.face_color_base_loc, self.face_color_base_scale).to_event(1)
        facial_hair_base_dist = Gumbel(self.facial_hair_base_loc, self.facial_hair_base_scale).to_event(1)
        hair_color_base_dist = Gumbel(self.hair_color_base_loc, self.hair_color_base_scale).to_event(1)

        hair_base = pyro.sample('hair_base', hair_base_dist)
        face_color_base = pyro.sample('face_color_base', face_color_base_dist)
        facial_hair_base = pyro.sample('facial_hair_base', facial_hair_base_dist)
        hair_color_base = pyro.sample('hair_color_base', hair_color_base_dist)

        # get the categorical values, starting with the roots
        hair = deterministic('hair', torch.argmax(hair_base + self.hair_logits, dim=-1).double())
        face_color = deterministic('face_color', torch.argmax(face_color_base + self.face_color_logits, dim=-1).double())
        facial_hair = deterministic('facial_hair', torch.argmax(facial_hair_base + self.facial_hair_logits[face_color.long(), hair.long(), ...], dim=-1).double())
        hair_color = deterministic('hair_color', torch.argmax(hair_color_base + self.hair_color_logits[face_color.long(), hair.long(), ...], dim=-1).double())

        # note: must return values in the order that they appear in THE_FEATURES
        return facial_hair, face_color, hair, hair_color

    @pyro_method
    def model(self):
        facial_hair, face_color, hair, hair_color = self.pgm_model()
        # breakpoint()
        context = torch.cat([feat.unsqueeze(-1) for feat in [hair, face_color, facial_hair, hair_color]], -1)

        #                                    ,-----This might be Sample x Height x Width
        # base noise of image: ______________|________________
        x_base_dist = Normal(self.x_base_loc, self.x_base_scale).to_event(3)
        cond_x_transforms = ComposeTransform(ConditionalTransformedDistribution(x_base_dist, self.x_transforms).condition(context).transforms).inv
        cond_x_dist = TransformedDistribution(x_base_dist, cond_x_transforms)
        x = pyro.sample('x', cond_x_dist)
        x = x.double()

        return x, facial_hair, face_color, hair, hair_color

    @pyro_method
    def infer_categorical_base(self, categorical_value, probs=None, logits=None):
        '''
        Given value of categorical variable and its logits, infer Gumbel noise
        that could have generated it.

        :param categorical_value: value of the categorical variable that had been observed.
        :param probs: the probs of the categorical distribution that generated categorical_value.
        :param logits: the logits of the categorical distribution that generated categorical_value.
        '''
        if probs is None and logits is None:
            raise ValueError('Must pass value for probs or logits.')
        return infer_gumbel_noise(categorical_value, probs=probs, logits=logits)

    # @pyro_method
    # def infer_thickness_base(self, thickness):
    #     return self.thickness_flow_transforms.inv(thickness)

    # @pyro_method
    # def infer_intensity_base(self, thickness, intensity):
    #     intensity_base_dist = Normal(self.intensity_base_loc, self.intensity_base_scale)

    #     thickness_ = self.thickness_flow_constraint_transforms.inv(thickness)
    #     cond_intensity_transforms = ComposeTransform(
    #         ConditionalTransformedDistribution(intensity_base_dist, self.intensity_flow_transforms).condition(thickness_).transforms)
    #     return cond_intensity_transforms.inv(intensity)

    @pyro_method
    def infer_x_base(self, context, x):
        '''
        Infer the base noise value, N_x, conditioned on context.

        :param context: a torch.tensor of the variables in the DAG concatenated together.
        :param x: the image
        '''
        x_base_dist = Normal(self.x_base_loc, self.x_base_scale)
        cond_x_transforms = ComposeTransform(ConditionalTransformedDistribution(x_base_dist, self.x_transforms).condition(context).transforms)
        return cond_x_transforms(x)

    @classmethod
    def add_arguments(cls, parser):
        parser = super().add_arguments(parser)

        parser.add_argument(
            '--use_affine_ex', default=False, action='store_true', help="whether to use conditional affine transformation on e_x (default: %(default)s)")

        return parser


