from typing import Mapping
import numpy as np
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
import pyro
from pyro.nn import pyro_method
from deepscm import util as U
from deepscm.experiments.toonfaces.types import THE_FEATURES
from deepscm.datasets.toonfaces import Toonfaces
from deepscm.experiments.toonfaces.base_experiment import BaseCovariateExperiment, BaseSEM, EXPERIMENT_REGISTRY, MODEL_REGISTRY  # noqa: F401

class BaseFlowSEM(BaseSEM):
    def __init__(self, num_scales: int = 4, flows_per_scale: int = 2, hidden_channels: int = 256,
                 use_actnorm: bool = False, **kwargs):
        super().__init__(**kwargs)

        self.num_scales = num_scales
        self.flows_per_scale = flows_per_scale
        self.hidden_channels = hidden_channels
        self.use_actnorm = use_actnorm

        # priors
        # assuming that these will be the parameters of the gumblemax
        # FIXME: requires grad is false -- this will likely cause problems
        for (name, max_size) in THE_FEATURES.items():
            shape = [max_size, ]
            # these buffer variabls are the 0, 1 values in our Gumbel(0, 1) samples and will not be trained
            # by the optimizer, hence why we register them as buffers and not parameters
            self.register_buffer('{}_base_loc'.format(name), torch.zeros(shape, requires_grad=False))
            self.register_buffer('{}_base_scale'.format(name), torch.ones(shape, requires_grad=False))
            # NOTE: in subclasses of BaseFlowSEM, we register the parameters for the logits of the categorical
            # distributions. These parameters have shapes that depend on the structure of specific DAG,
            # hence why they're in the specific SEM subclasses

        # NOTE: this is noise around images, shape is image shape
        self.register_buffer('x_base_loc', torch.zeros([Toonfaces.channels, Toonfaces.height, Toonfaces.width], requires_grad=False))
        self.register_buffer('x_base_scale', torch.ones([Toonfaces.channels, Toonfaces.height, Toonfaces.width], requires_grad=False))

    @pyro_method
    def infer(self, **obs):
        return self.infer_exogenous(**obs)

    # EXPERIMENTAL
    @pyro_method
    def counterfactual(self,
        obs: Mapping, condition: Mapping=None,
        lookup_infer=False, img_dir: str=None,
        att_to_img_dict: dict=None, att_to_img_path: str=None):
        _required_data = ['x'] + list(THE_FEATURES.keys())
        assert set(obs.keys()) == set(_required_data)

        exogeneous = self.infer(**obs)
        counter = pyro.poutine.do(pyro.poutine.condition(self.sample_scm, data=exogeneous), data=condition)(obs['x'].shape[0])
        if lookup_infer:
            breakpoint()
            assert img_dir is not None
            conditioning_set = {k: v for k, v in zip(list(THE_FEATURES.keys()), counter[1:])}
            if att_to_img_dict is None:
                if att_to_img_path is None:
                    raise ValueError('Must pass att_to_img path if lookup_infer and att_to_img_path are None')
                att_to_img_dict = U.load_from_pickle(att_to_img_path)
            img = U.simulate_conditional_img(conditioning_set, att_to_img_dict, img_dir=img_dir)
            img = np.asarray(img)
            conditioning_set['x'] = torch.tensor(img)
            return conditioning_set
        return {k: v for k, v in zip(_required_data, counter)}

    @classmethod
    def add_arguments(cls, parser):
        parser = super().add_arguments(parser)

        parser.add_argument('--num_scales', default=4, type=int, help="number of scales (default: %(default)s)")
        parser.add_argument('--flows_per_scale', default=10, type=int, help="number of flows per scale (default: %(default)s)")
        parser.add_argument('--hidden_channels', default=256, type=int, help="number of hidden channels in convnet (default: %(default)s)")
        parser.add_argument('--use_actnorm', default=False, action='store_true', help="whether to use activation norm (default: %(default)s)")

        return parser


class NormalisingFlowsExperiment(BaseCovariateExperiment):
    def __init__(self, hparams, pyro_model: BaseSEM):
        # hparams.latent_dim = 32 * 32

        super().__init__(hparams, pyro_model)

    def configure_optimizers(self):

        # thickness_params = self.pyro_model.thickness_flow_components.parameters()
        # intensity_params = self.pyro_model.intensity_flow_components.parameters()

        x_params = self.pyro_model.trans_modules.parameters()
        face_color_logits = self.pyro_model.face_color_logits
        hair_logits = self.pyro_model.hair_logits
        hair_color_logits = self.pyro_model.hair_color_logits
        facial_hair_logits = self.pyro_model.facial_hair_logits

        optimizer = torch.optim.Adam([
            # params=self.pyro_model.parameters(),
            {'params': x_params, 'lr': self.hparams.img_lr},
            {'params': face_color_logits, 'lr': self.hparams.pgm_lr},
            {'params': hair_logits, 'lr': self.hparams.pgm_lr},
            {'params': hair_color_logits, 'lr': self.hparams.pgm_lr},
            {'params': facial_hair_logits, 'lr': self.hparams.pgm_lr},
            ], lr=self.hparams.general_lr, eps=1e-5,
            amsgrad=self.hparams.use_amsgrad, weight_decay=self.hparams.l2)

        scheduler = CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-6)
        return [optimizer], [scheduler] # why do we have to return these as lists?


    def prepare_data(self):
        super().prepare_data()

        # self.z_range = self.z_range.reshape((9, 1, 32, 32))

    def get_logprobs(self, **obs):
        # _required_data = ('x', 'thickness', 'intensity')
        # EDIT
        _required_data = ('x', 'hair', 'face_color', 'hair_color', 'facial_hair')
        assert set(obs.keys()) == set(_required_data)

        cond_model = pyro.condition(self.pyro_model.sample, data=obs)
        model_trace = pyro.poutine.trace(cond_model).get_trace(obs['x'].shape[0])
        model_trace.compute_log_prob()

        log_probs = {}
        nats_per_dim = {}

        # get the values of the root nodes
        hair_value = model_trace.nodes['hair']['value'].long()
        face_color_value = model_trace.nodes['face_color']['value'].long()
        # comment me out
        # print(f'hair_value.size(): {hair_value.size()}')

        # loop through the model_trace nodes to get the image site, x
        for name, site in model_trace.nodes.items():
            if site["type"] == "sample" and site["is_observed"] and name == 'x':
                log_probs[name] = site["log_prob"].mean()
                log_prob_shape = site["log_prob"].shape
                value_shape = site["value"].shape
                if len(log_prob_shape) < len(value_shape):
                    dims = np.prod(value_shape[len(log_prob_shape):])
                else:
                    dims = 1.
                nats_per_dim[name] = -site["log_prob"].mean() / dims

                self.logger.experiment.add_scalar('x/base/log_prob', site["log_prob"].mean())

                if self.hparams.validate:
                    print(f'at site {name} with dim {dims} and nats: {nats_per_dim[name]} and logprob: {log_probs[name]}')
                    if torch.any(torch.isnan(nats_per_dim[name])):
                        raise ValueError('got nan')
            elif name in THE_FEATURES.keys():
                # breakpoint()
                value_shape = site['value'].shape
                logits = getattr(self.pyro_model, name + '_logits')
                if name == 'hair_color' or name == 'facial_hair':
                    # note that to get the logits for these variables, we have to index their CPTs
                    logits = logits.repeat(value_shape[0], *[1 for i in range(len(logits.shape))])
                    logits = torch.index_select(logits, dim=1, index=face_color_value)
                    logits = torch.index_select(logits, dim=2, index=hair_value)

                # FIXME: this code may be buggy
                # not sure what to expect from logits.shape vs what standard site['log_prob'].shape would return
                log_prob_shape = logits.shape
                if len(log_prob_shape) < len(value_shape):
                    dims = np.prod(value_shape[len(log_prob_shape):])
                else:
                    dims = 1.
                nats_per_dim[name] = -logits.mean() / dims
                self.logger.experiment.add_scalar(f'x/{name}/logits', -logits.mean())
                if self.hparams.validate:
                    print(f'at site {name} with dim {dims} and nats: {nats_per_dim[name]} and logprob: {logits.mean()}')
                    if torch.any(torch.isnan(nats_per_dim[name])):
                        raise ValueError('got nan - see line 141 of base_nf_experiment')
        return log_probs, nats_per_dim

    def prep_batch(self, batch):
        x = batch['image'].float()
        attrs = batch['attrs']
        hair = torch.tensor(attrs['hair'])
        hair_color = torch.tensor(attrs['hair_color'])
        facial_hair = torch.tensor(attrs['facial_hair'])
        face_color = torch.tensor(attrs['face_color'])

        # from IPython.core.debugger import Tracer; Tracer()()

        x = torch.nn.functional.pad(x, (0, 0, 2, 2, 2, 2)) # BHWC-> B{2H2}{2W2}C
        x += torch.rand_like(x)

        x = x.reshape(-1, Toonfaces.channels, Toonfaces.height, Toonfaces.width)

        return {'x': x, 'hair': hair, 'hair_color': hair_color,
            'facial_hair': facial_hair, 'face_color': face_color}

    def training_step(self, batch, batch_idx):
        batch = self.prep_batch(batch)

        log_probs, nats_per_dim = self.get_logprobs(**batch)
        loss = torch.stack(tuple(nats_per_dim.values())).sum()

        if torch.isnan(loss):
            self.logger.experiment.add_text('nan', f'nand at {self.current_epoch}')
            raise ValueError('loss went to nan')

        lls = {(f'train/log p({k})'): v for k, v in log_probs.items()}
        nats_per_dim = {('train/' + k + '_nats_per_dim'): v for k, v in nats_per_dim.items()}

        tensorboard_logs = {'train/loss': loss, **nats_per_dim, **lls}

        return {'loss': loss, 'log': tensorboard_logs, **lls}

    def validation_step(self, batch, batch_idx):
        batch = self.prep_batch(batch)

        log_probs, nats_per_dim = self.get_logprobs(**batch)
        loss = torch.stack(tuple(nats_per_dim.values())).sum()

        lls = {(f'train/log p({k})'): v for k, v in log_probs.items()}
        nats_per_dim = {(k + '_nats_per_dim'): v for k, v in nats_per_dim.items()}

        return {'loss': loss, **lls, **nats_per_dim}

    def test_step(self, batch, batch_idx):
        batch = self.prep_batch(batch)

        log_probs, nats_per_dim = self.get_logprobs(**batch)
        loss = torch.stack(tuple(nats_per_dim.values())).sum()

        lls = {(f'train/log p({k})'): v for k, v in log_probs.items()}
        nats_per_dim = {(k + '_nats_per_dim'): v for k, v in nats_per_dim.items()}

        return {'loss': loss, **lls, **nats_per_dim}


