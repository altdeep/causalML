import pyro

from pyro.nn import PyroModule, pyro_method

from pyro.distributions import TransformedDistribution
from pyro.infer.reparam.transform import TransformReparam
from torch.distributions import Independent

from deepscm.datasets.toonfaces import Toonfaces # note, returns key, img, attributes
from pyro.distributions.transforms import ComposeTransform, SigmoidTransform, AffineTransform
from deepscm.experiments.toonfaces.types import THE_FEATURES
from deepscm.util import infer_gumbel_noise

import torchvision.utils
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
import torch
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

# don't need for toonfaces
# from deepscm.morphodata import measure
import os
from functools import partial
import multiprocessing
import operator
from functools import reduce

def flatten(xs):
    return reduce(operator.iconcat, xs, [])


EXPERIMENT_REGISTRY = {}
MODEL_REGISTRY = {}


class BaseSEM(PyroModule):
    def __init__(self, preprocessing: str = 'realnvp'):
        super().__init__()

        self.preprocessing = preprocessing

        # priors
        # assuming that these will be the parameters of the gumblemax
        # FIXME: requires grad is false -- this will likely cause problems
        for (name, max_size) in THE_FEATURES.items():
            shape = [max_size, ]
            label = lambda suf: '{}_lognorm{}'.format(name, suf)
            self.register_buffer(label('_loc'), torch.zeros(shape, requires_grad=False))
            self.register_buffer(label('_scale'), torch.ones(shape, requires_grad=False))
            loc = getattr(self, label('_loc'))
            scale = getattr(self, label('_scale'))
            setattr(self, label(''), AffineTransform(loc=loc, scale=scale)) # NOTE: we call this the 'transform'

    def __setattr__(self, name, value):
        super().__setattr__(name, value)
        name_scheme = lambda pre, suf: '{}_lognorm{}'.format(pre, suf)
        feature_keys = THE_FEATURES.keys()
        feature_key_to_reserved = lambda key: [name_scheme(key, "_loc"), name_scheme(key, "_scale")]
        reserved_from_register_buffer = flatten([feature_key_to_reserved(key) for key in feature_keys])

        if name in reserved_from_register_buffer:
            # when a user sets a reserved scheme, we must fix the AffineTransform params
            # FIXME: original code always references non-tensor-based scalars via ___.item()
            loc = getattr(self, name_scheme(name, '_loc'))
            scale = getattr(self, name_scheme(name, '_scale'))
            transform = getattr(self, name_scheme(name, ''))
            setattr(transform, "loc", loc)
            setattr(transform, "scale", scale)

    def _get_preprocess_transforms(self):
        alpha = 0.05
        num_bits = 8

        if self.preprocessing == 'glow':
            # Map to [-0.5,0.5]
            a1 = AffineTransform(-0.5, (1. / 2 ** num_bits))
            preprocess_transform = ComposeTransform([a1])
        elif self.preprocessing == 'realnvp':
            # Map to [0,1]
            a1 = AffineTransform(0., (1. / 2 ** num_bits))

            # Map into unconstrained space as done in RealNVP
            a2 = AffineTransform(alpha, (1 - alpha))

            s = SigmoidTransform()

            preprocess_transform = ComposeTransform([a1, a2, s.inv])

        return preprocess_transform

    @pyro_method
    def pgm_model(self):
        raise NotImplementedError()

    @pyro_method
    def model(self):
        raise NotImplementedError()

    @pyro_method
    def pgm_scm(self, *args, **kwargs):
        def config(msg):
            if isinstance(msg['fn'], TransformedDistribution):
                return TransformReparam()
            else:
                return None

        return pyro.poutine.reparam(self.pgm_model, config=config)(*args, **kwargs)

    @pyro_method
    def scm(self, *args, **kwargs):
        def config(msg):
            if isinstance(msg['fn'], TransformedDistribution):
                return TransformReparam()
            else:
                return None

        return pyro.poutine.reparam(self.model, config=config)(*args, **kwargs)

    @pyro_method
    def sample(self, n_samples=1):
        with pyro.plate('observations', n_samples):
            samples = self.model()

        return (*samples,)

    @pyro_method
    def sample_scm(self, n_samples=1):
        with pyro.plate('observations', n_samples):
            samples = self.scm()

        return (*samples,)

    # TODO: this is not used. Is this a problem???? (e_t refers to thickness)
    @pyro_method
    def infer_e_t(self, *args, **kwargs):
        raise NotImplementedError()

    # TODO: this is not used. Is this a problem???? (e_s refers to s.......stroke?)
    @pyro_method
    def infer_e_s(self, *args, **kwargs):
        raise NotImplementedError()

    # TODO: this is not used. Is this a problem???? (e_x refers to the image)
    @pyro_method
    def infer_e_x(self, *args, **kwargs):
        raise NotImplementedError()

    # @pyro_method
    # def infer_exogeneous(self, **obs: dict):
    #     '''
    #     Infer values of the exogenous variables in the SCM, given the observed values
    #     for those variables, obs.

    #     :param obs: dict of observed values. 
    #     :return output: dict of values for the inferred exogenous noise distributions.
    #         Note that like the distribution names for the exegenous variables, the 
    #         values' key names have suffix "_base".
    #     '''
    #     # assuming that we use transformed distributions for everything:
    #     cond_sample = pyro.condition(self.sample, data=obs)
    #     cond_trace = pyro.poutine.trace(cond_sample).get_trace(obs['x'].shape[0])

    #     output = {}
    #     for name, node in cond_trace.nodes.items():
    #         if 'fn' not in node.keys():
    #             continue

    #         fn = node['fn']
    #         if isinstance(fn, Independent):
    #             fn = fn.base_dist
    #         if isinstance(fn, TransformedDistribution):
    #             output[name + '_base'] = ComposeTransform(fn.transforms).inv(node['value'])

    #     return output

    @pyro_method
    def infer_exogenous(self, **obs: dict):
        '''
        Infer values for all the exogenous variables of the SCM given values of the endogenous variables.

        Nethod assumes x is the only variable generated from transformd distribution
        and all other variables are discrete and sampled using the Gumbel max trick.

        :param obs: dict, observed values of the endogenous variables.
        :return output: dict, values of the exogenous noise variables.
        '''
        
        cond_sample = pyro.condition(self.sample, data=obs)
        cond_trace = pyro.poutine.trace(cond_sample).get_trace(obs['x'].shape[0]) # this gets a trace per observation in obs['x']
        output = {}
        hair_value = cond_trace.nodes['hair']['value'].item()
        face_color_value = cond_trace.nodes['face_color']['value'].item()
        for name, node in cond_trace.nodes.items():
            if 'fn' not in node.keys():
                continue

            # if the node is one of the categorical features (all THE_FEATURES are categorical)
            # then infer their Gumbel values given their logits
            # FIXME: in a future version, we should make this more flexible so that
            # given a name of a variable we have tracked its parent dependencies and
            # can get its logits via the tracking
            if node['name'] in THE_FEATURES.keys():
                logits = getattr(self, node['name'] + '_logits') 
                if node['name'] == 'hair_color' or node['name'] == 'facial_hair':
                    # note that to get the logits for these variables, we have to index their CPTs
                    logits = logits[face_color_value, hair_value, ...]
                output[name + '_base'] = infer_gumbel_noise(k=node['value'], logits=logits)
            else:
                fn = node['fn']
                if isinstance(fn, Independent):
                    fn = fn.base_dist
                if isinstance(fn, TransformedDistribution):
                    breakpoint()
                    output[name + '_base'] = ComposeTransform(fn.transforms).inv(node['value'])
        return output
                

    @pyro_method
    def infer(self, **obs):
        raise NotImplementedError()

    @pyro_method
    def counterfactual(self, obs, condition=None):
        raise NotImplementedError()

    @classmethod
    def add_arguments(cls, parser):
        parser.add_argument('--preprocessing', default='realnvp', type=str, help="type of preprocessing (default: %(default)s)", choices=['realnvp', 'glow'])

        return parser


class BaseCovariateExperiment(pl.LightningModule):
    def __init__(self, hparams, pyro_model: BaseSEM):
        super().__init__()

        self.pyro_model = pyro_model

        hparams.experiment = self.__class__.__name__
        hparams.model = pyro_model.__class__.__name__
        self.hparams = hparams
        self.data_dir = hparams.data_dir
        self.train_batch_size = hparams.train_batch_size
        self.test_batch_size = hparams.test_batch_size

        if hasattr(hparams, 'num_sample_particles'):
            self.pyro_model._gen_counterfactual = partial(self.pyro_model.counterfactual, num_particles=self.hparams.num_sample_particles)
        else:
            self.pyro_model._gen_counterfactual = self.pyro_model.counterfactual

        if hparams.validate:
            import random

            torch.manual_seed(0)
            np.random.seed(0)
            random.seed(0)

            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.autograd.set_detect_anomaly(self.hparams.validate)
            pyro.enable_validation()

    # REPLACEME: this function is irrelevant for toonfaces. 
    # instead, we need to replace with a method to get the label of the endogenous variables
    # associated with the image.
    # def measure_image(self, x, normalize=False, threshold=0.5):
    #     imgs = x.detach().cpu().numpy()[:, 0]

    #     if normalize:
    #         imgs -= imgs.min()
    #         imgs /= imgs.max() + 1e-6

    #     with multiprocessing.Pool() as pool:
    #         measurements = measure.measure_batch(imgs, threshold=threshold, pool=pool)

    #     def get_intensity(imgs, threshold):

    #         img_min, img_max = imgs.min(axis=(1, 2), keepdims=True), imgs.max(axis=(1, 2), keepdims=True)
    #         mask = (imgs >= img_min + (img_max - img_min) * threshold)

    #         return np.array([np.median(i[m]) for i, m in zip(imgs, mask)])

    #     return measurements['thickness'].values, get_intensity(imgs, threshold)

    def prepare_data(self):
        # prepare transforms standard to DATA
        data_train = Toonfaces(self.data_dir)
        self.data_test = Toonfaces(self.data_dir)

        num_val = int(len(data_train) * 0.1)
        num_train = len(data_train) - num_val
        self.data_train, self.data_val = random_split(data_train, [num_train, num_val])

        self.torch_device = self.trainer.root_gpu if self.trainer.on_gpu else self.trainer.root_device
        print(f'using device: {self.torch_device}')

        # COME BACK:
        # Need some help here... I think we need to do the same thing as the below but for the features that we're experimenting with.

        # # =================================================================
        # thicknesses = 1. + torch.arange(3, dtype=torch.float, device=self.torch_device)
        # self.thickness_range = thicknesses.repeat(3).unsqueeze(1)
        # intensity = 48 * torch.arange(3, dtype=torch.float, device=self.torch_device) + 64
        # self.intensity_range = intensity.repeat_interleave(3).unsqueeze(1)
        # self.z_range = torch.randn([1, self.hparams.latent_dim], device=self.torch_device, dtype=torch.float).repeat((9, 1)) # only relevant for VI experiments

        # self.pyro_model.intensity_flow_norm_loc = self.data_train.metrics['intensity'].min().to(self.torch_device).float()
        # self.pyro_model.intensity_flow_norm_scale = (self.data_train.metrics['intensity'].max() - self.data_train.metrics['intensity'].min()).to(self.torch_device).float()  # noqa: E501

        # self.pyro_model.thickness_flow_lognorm_loc = self.data_train.metrics['thickness'].log().mean().to(self.torch_device).float()
        # self.pyro_model.thickness_flow_lognorm_scale = self.data_train.metrics['thickness'].log().std().to(self.torch_device).float()
        # # =================================================================

        # print(f'set thickness_flow_lognorm.loc to {self.pyro_model.thickness_flow_lognorm.loc}')

    def configure_optimizers(self):
        pass

    def train_dataloader(self):
        return DataLoader(self.data_train, batch_size=self.train_batch_size, shuffle=True)

    def val_dataloader(self):
        self.val_loader = DataLoader(self.data_test, batch_size=self.test_batch_size, shuffle=False)
        return self.val_loader

    def test_dataloader(self):
        self.test_loader = DataLoader(self.data_test, batch_size=self.test_batch_size, shuffle=False)
        return self.test_loader

    def forward(self, *args, **kwargs):
        pass

    def prep_batch(self, batch):
        raise NotImplementedError()

    def training_step(self, batch, batch_idx):
        raise NotImplementedError()

    def validation_step(self, batch, batch_idx):
        raise NotImplementedError()

    def validation_epoch_end(self, outputs):
        # outputs = self.assemble_epoch_end_outputs(outputs)
        #
        # metrics = {('val/' + k): v for k, v in outputs.items()}
        #
        # if self.current_epoch % self.hparams.sample_img_interval == 0:
        #     self.sample_images()
        #
        # val_loss = metrics['val/loss'] if isinstance(metrics['val/loss'], torch.Tensor) else torch.tensor(metrics['val/loss'])

        return dict() # {'val_loss': val_loss, 'log': metrics}

    # COME BACK: What do we do here?
    def test_epoch_end(self, outputs):
    #     print('Assembling outputs')
    #     outputs = self.assemble_epoch_end_outputs(outputs)

    #     samples = outputs.pop('samples')

    #     sample_trace = pyro.poutine.trace(self.pyro_model.sample).get_trace(self.hparams.test_batch_size)
    #     samples['unconditional_samples'] = {
    #         'x': sample_trace.nodes['x']['value'].cpu(),
    #         'thickness': sample_trace.nodes['thickness']['value'].cpu(),
    #         'intensity': sample_trace.nodes['intensity']['value'].cpu()
    #     }

    #     cond_data = {
    #         'thickness': self.thickness_range.repeat(self.hparams.test_batch_size, 1),
    #         'intensity': self.intensity_range.repeat(self.hparams.test_batch_size, 1),
    #         'z': torch.randn([self.hparams.test_batch_size, self.hparams.latent_dim], device=self.torch_device, dtype=torch.float).repeat_interleave(9, 0)
    #     }
    #     sample_trace = pyro.poutine.trace(pyro.condition(self.pyro_model.sample, data=cond_data)).get_trace(9 * self.hparams.test_batch_size)
    #     samples['conditional_samples'] = {
    #         'x': sample_trace.nodes['x']['value'].cpu(),
    #         'thickness': sample_trace.nodes['thickness']['value'].cpu(),
    #         'intensity': sample_trace.nodes['intensity']['value'].cpu()
    #     }

    #     print(f'Got samples: {tuple(samples.keys())}')

    #     metrics = {('test/' + k): v for k, v in outputs.items()}

    #     for k, v in samples.items():
    #         print(f'Measuring samples for {k}')
    #         measured_thickness, measured_intensity = self.measure_image(v['x'])

    #         p = os.path.join(self.trainer.logger.experiment.log_dir, f'{k}.pt')

    #         obj = {'measured_thickness': measured_thickness, 'measured_intensity': measured_intensity, **v}

    #         print(f'Saving samples for {k} to {p}')

    #         torch.save(obj, p)

    #     p = os.path.join(self.trainer.logger.experiment.log_dir, 'metrics.pt')
    #     torch.save(metrics, p)

    #     prob_maps = self.build_prob_maps()
    #     p = os.path.join(self.trainer.logger.experiment.log_dir, 'prob_maps.pt')
    #     torch.save(prob_maps, p)

        return dict() # {'test_loss': metrics['test/loss'], 'log': metrics}

    # FIXME
    def assemble_epoch_end_outputs(self, outputs):
        pass
        # num_items = len(outputs)

        # def handle_row(batch, assembled=None):
        #     if assembled is None:
        #         assembled = {}

        #     for k, v in batch.items():
        #         if k not in assembled.keys():
        #             if isinstance(v, dict):
        #                 assembled[k] = handle_row(v)
        #             elif isinstance(v, float):
        #                 assembled[k] = v
        #             elif np.prod(v.shape) == 1:
        #                 assembled[k] = v.cpu()
        #             else:
        #                 assembled[k] = v.cpu()
        #         else:
        #             if isinstance(v, dict):
        #                 assembled[k] = handle_row(v, assembled[k])
        #             elif isinstance(v, float):
        #                 assembled[k] += v
        #             elif np.prod(v.shape) == 1:
        #                 assembled[k] += v.cpu()
        #             else:
        #                 assembled[k] = torch.cat([assembled[k], v.cpu()], 0)

        #     return assembled

        # assembled = {}
        # for _, batch in enumerate(outputs):
        #     assembled = handle_row(batch, assembled)

        # for k, v in assembled.items():
        #     if (hasattr(v, 'shape') and np.prod(v.shape) == 1) or isinstance(v, float):
        #         assembled[k] /= num_items

        # return assembled

    # EDIT: changed the counterfactual scenarios
    def get_counterfactual_conditions(self, batch):
        counterfactuals = {
            'do(thickness=1.5)': {'thickness': torch.ones_like(batch['thickness']) * 1.5},
            'do(thickness=5)': {'thickness': torch.ones_like(batch['thickness']) * 5},
            'do(intensity=64)': {'intensity': torch.ones_like(batch['intensity']) * 64},
            'do(intensity=224)': {'intensity': torch.ones_like(batch['intensity']) * 224},
            'do(thickness=1.5, intensity=224)': {'thickness': torch.ones_like(batch['thickness']) * 1.5, 'intensity': torch.ones_like(batch['intensity']) * 224},
            'do(thickness=5, intensity=64)': {'thickness': torch.ones_like(batch['thickness']) * 5, 'intensity': torch.ones_like(batch['intensity']) * 64}
        }

        return counterfactuals

    # NOTE: this might be all we need, ex. training?
    def build_test_samples(self, batch):
        samples = {}
        samples['reconstruction'] = {'x': self.pyro_model.reconstruct(**batch, num_particles=self.hparams.num_sample_particles)}

        counterfactuals = self.get_counterfactual_conditions(batch)

        for name, condition in counterfactuals.items():
            samples[name] = self.pyro_model._gen_counterfactual(obs=batch, condition=condition)

        return samples

    # TODO: stuff to do
    def build_prob_maps(self):
        prob_maps = {}

        def sample_pgm(num_samples):
            with pyro.plate('observations', num_samples):
                return self.pyro_model.pgm_model()

        intensity_range = torch.arange(64, 255, 1, device=self.torch_device, dtype=torch.float)
        thickness_range = torch.arange(1, 5, 0.1, device=self.torch_device, dtype=torch.float)

        num_intensity = intensity_range.shape[0]
        num_thickness = thickness_range.shape[0]

        intensity_range = intensity_range.repeat(num_thickness).unsqueeze(1)
        thickness_range = thickness_range.repeat_interleave(num_intensity).unsqueeze(1)

        cond_data = {
            'thickness': thickness_range,
            'intensity': intensity_range
        }

        trace = pyro.poutine.trace(pyro.condition(sample_pgm, data=cond_data)).get_trace(len(intensity_range))
        trace.compute_log_prob()

        # TODO: something needs to be done here !!!
        prob_maps['base_distribution'] = {'log_prob': trace.nodes['thickness']['log_prob'] + trace.nodes['intensity']['log_prob'], **cond_data}

        return prob_maps

    def log_img_grid(self, tag, imgs, normalize=True, save_img=False, **kwargs):
        if save_img:
            p = os.path.join(self.trainer.logger.experiment.log_dir, f'{tag}.png')
            torchvision.utils.save_image(imgs, p)
        grid = torchvision.utils.make_grid(imgs, normalize=normalize, **kwargs)
        self.logger.experiment.add_image(tag, grid, self.current_epoch)

    def get_batch(self, loader):
        batch = next(iter(self.val_loader))
        if self.trainer.on_gpu:
            batch = self.trainer.transfer_batch_to_gpu(batch, self.torch_device)
        return batch

    def log_kdes(self, tag, data, save_img=False):
        """
        requires data to be:
        {'data1': {'x': x, 'y': y}, 'data2': {'x': x, 'y': y}, ..}
        """
        def np_val(x):
            return x.cpu().numpy().squeeze() if isinstance(x, torch.Tensor) else x.squeeze()

        fig, ax = plt.subplots(1, len(data), figsize=(5 * len(data), 3), sharex=True, sharey=True)
        for i, (name, covariates) in enumerate(data.items()):
            try:
                if len(covariates) == 1:
                    (x_n, x), = tuple(covariates.items())
                    sns.kdeplot(np_val(x), ax=ax[i], shade=True, shade_lowest=False)
                elif len(covariates) == 2:
                    (x_n, x), (y_n, y) = tuple(covariates.items())
                    sns.kdeplot(np_val(x), np_val(y), ax=ax[i], shade=True, shade_lowest=False)
                    ax[i].set_ylabel(y_n)
                else:
                    raise ValueError(f'got too many values: {len(covariates)}')
            except np.linalg.LinAlgError:
                print(f'got a linalg error when plotting {tag}/{name}')

            ax[i].set_title(name)
            ax[i].set_xlabel(x_n)

        sns.despine()

        if save_img:
            p = os.path.join(self.trainer.logger.experiment.log_dir, f'{tag}.png')
            plt.savefig(p, dpi=300)

        self.logger.experiment.add_figure(tag, fig, self.current_epoch)

    # def build_reconstruction(self, x, thickness, intensity, tag='reconstruction'):
    #     obs = {'x': x, 'thickness': thickness, 'intensity': intensity}

    #     recon = self.pyro_model.reconstruct(**obs, num_particles=self.hparams.num_sample_particles)
    #     self.log_img_grid(tag, torch.cat([x, recon], 0))
    #     self.logger.experiment.add_scalar(f'{tag}/mse', torch.mean(torch.square(x - recon).sum((1, 2, 3))), self.current_epoch)

    #     measured_thickness, measured_intensity = self.measure_image(recon)
    #     self.logger.experiment.add_scalar(
    #         f'{tag}/thickness_mae', torch.mean(torch.abs(thickness.cpu() - measured_thickness)), self.current_epoch)
    #     self.logger.experiment.add_scalar(
    #         f'{tag}/intensity_mae', torch.mean(torch.abs(intensity.cpu() - measured_intensity)), self.current_epoch)

    def build_counterfactual(self, tag, obs, conditions, absolute=None):
        _required_data = ('x', 'thickness', 'intensity')
        assert set(obs.keys()) == set(_required_data), 'got: {}'.format(tuple(obs.keys()))

        imgs = [obs['x']]
        if absolute == 'thickness':
            sampled_kdes = {'orig': {'intensity': obs['intensity']}}
        elif absolute == 'intensity':
            sampled_kdes = {'orig': {'thickness': obs['thickness']}}
        else:
            sampled_kdes = {'orig': {'thickness': obs['thickness'], 'intensity': obs['intensity']}}
        measured_kdes = {'orig': {'thickness': obs['thickness'], 'intensity': obs['intensity']}}

        for name, data in conditions.items():
            counterfactual = self.pyro_model._gen_counterfactual(obs=obs, condition=data)

            counter = counterfactual['x']
            sampled_thickness = counterfactual['thickness']
            sampled_intensity = counterfactual['intensity']

            measured_thickness, measured_intensity = self.measure_image(counter.cpu())

            imgs.append(counter)
            if absolute == 'thickness':
                sampled_kdes[name] = {'intensity': sampled_intensity}
            elif absolute == 'intensity':
                sampled_kdes[name] = {'thickness': sampled_thickness}
            else:
                sampled_kdes[name] = {'thickness': sampled_thickness, 'intensity': sampled_intensity}
            measured_kdes[name] = {'thickness': measured_thickness, 'intensity': measured_intensity}

            self.logger.experiment.add_scalar(
                f'{tag}/{name}/thickness_mae', torch.mean(torch.abs(sampled_thickness.cpu() - measured_thickness)), self.current_epoch)
            self.logger.experiment.add_scalar(
                f'{tag}/{name}/intensity_mae', torch.mean(torch.abs(sampled_intensity.cpu() - measured_intensity)), self.current_epoch)

        self.log_img_grid(tag, torch.cat(imgs, 0))
        self.log_kdes(f'{tag}_sampled', sampled_kdes, save_img=True)
        self.log_kdes(f'{tag}_measured', measured_kdes, save_img=True)

    # def sample_images(self):
    #     with torch.no_grad():
    #         sample_trace = pyro.poutine.trace(self.pyro_model.sample).get_trace(self.hparams.test_batch_size)

    #         samples = sample_trace.nodes['x']['value']
    #         sampled_thickness = sample_trace.nodes['thickness']['value']
    #         sampled_intensity = sample_trace.nodes['intensity']['value']

    #         self.log_img_grid('samples', samples.data[:8])

    #         # can't implement
    #         measured_thickness, measured_intensity = self.measure_image(samples)
    #         self.logger.experiment.add_scalar('samples/thickness_mae', torch.mean(torch.abs(sampled_thickness.cpu() - measured_thickness)), self.current_epoch)
    #         self.logger.experiment.add_scalar('samples/intensity_mae', torch.mean(torch.abs(sampled_intensity.cpu() - measured_intensity)), self.current_epoch)


    #         cond_data = {'thickness': self.thickness_range, 'intensity': self.intensity_range, 'z': self.z_range}
    #         samples, *_ = pyro.condition(self.pyro_model.sample, data=cond_data)(9)
    #         self.log_img_grid('cond_samples', samples.data, nrow=3)

    #         obs_batch = self.prep_batch(self.get_batch(self.val_loader))

    #         kde_data = {
    #             'batch': {'thickness': obs_batch['thickness'], 'intensity': obs_batch['intensity']},
    #             'sampled': {'thickness': sampled_thickness, 'intensity': sampled_intensity}
    #         }
    #         self.log_kdes('sample_kde', kde_data, save_img=True)

    #         exogeneous = self.pyro_model.infer(**obs_batch)

    #         for (tag, val) in exogeneous.items():
    #             self.logger.experiment.add_histogram(tag, val, self.current_epoch)

    #         obs_batch = {k: v[:8] for k, v in obs_batch.items()}

    #         self.log_img_grid('input', obs_batch['x'], save_img=True)

    #         if hasattr(self.pyro_model, 'reconstruct'):
    #             self.build_reconstruction(**obs_batch)

    #         conditions = {
    #             '+1': {'thickness': obs_batch['thickness'] + 1},
    #             '+2': {'thickness': obs_batch['thickness'] + 2},
    #             '+3': {'thickness': obs_batch['thickness'] + 3}
    #         }
    #         self.build_counterfactual('do(thickness=+x)', obs=obs_batch, conditions=conditions)

    #         conditions = {
    #             '1': {'thickness': torch.ones_like(obs_batch['thickness'])},
    #             '2': {'thickness': torch.ones_like(obs_batch['thickness']) * 2},
    #             '3': {'thickness': torch.ones_like(obs_batch['thickness']) * 3}
    #         }
    #         self.build_counterfactual('do(thickness=x)', obs=obs_batch, conditions=conditions, absolute='thickness')

    #         conditions = {
    #             '-64': {'intensity': obs_batch['intensity'] - 64},
    #             '-32': {'intensity': obs_batch['intensity'] - 32},
    #             '+32': {'intensity': obs_batch['intensity'] + 32},
    #             '+64': {'intensity': obs_batch['intensity'] + 64}
    #         }
    #         self.build_counterfactual('do(intensity=+x)', obs=obs_batch, conditions=conditions)

    #         conditions = {
    #             '64': {'intensity': torch.zeros_like(obs_batch['intensity']) + 64},
    #             '96': {'intensity': torch.zeros_like(obs_batch['intensity']) + 96},
    #             '192': {'intensity': torch.zeros_like(obs_batch['intensity']) + 192},
    #             '224': {'intensity': torch.zeros_like(obs_batch['intensity']) + 224}
    #         }
    #         self.build_counterfactual('do(intensity=x)', obs=obs_batch, conditions=conditions, absolute='intensity')

    @classmethod
    def add_arguments(cls, parser):
        parser.add_argument(
            '--data_dir', default="/vol/biomedic2/np716/data/gemini/synthetic/thickness_intensity/all_fixed_scale05/", type=str, help="data dir (default: %(default)s)")  # noqa: E501
        parser.add_argument('--sample_img_interval', default=10, type=int, help="interval in which to sample and log images (default: %(default)s)")
        parser.add_argument('--train_batch_size', default=256, type=int, help="train batch size (default: %(default)s)")
        parser.add_argument('--test_batch_size', default=256, type=int, help="test batch size (default: %(default)s)")
        parser.add_argument('--validate', default=False, action='store_true', help="whether to validate (default: %(default)s)")
        parser.add_argument('--lr', default=1e-4, type=float, help="lr of deep part (default: %(default)s)")
        parser.add_argument('--pgm_lr', default=5e-3, type=float, help="lr of pgm (default: %(default)s)")
        parser.add_argument('--l2', default=0., type=float, help="weight decay (default: %(default)s)")
        parser.add_argument('--use_amsgrad', default=False, action='store_true', help="use amsgrad? (default: %(default)s)")

        return parser
