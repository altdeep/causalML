from deepscm.experiments import morphomnist  # noqa: F401
from .base_experiment import EXPERIMENT_REGISTRY, MODEL_REGISTRY

import torch
import inspect


if __name__ == '__main__':
    from pytorch_lightning import Trainer
    import argparse
    import os

    exp_parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    exp_parser.add_argument('--checkpoint_path', '-c', help='which checkpoint to load')

    exp_args, other_args = exp_parser.parse_known_args()

    print(f'Running test with {exp_args}')

    base_path = os.path.join(exp_args.checkpoint_path, 'checkpoints')
    checkpoint_path = os.path.join(base_path, os.listdir(base_path)[0])

    print(f'using checkpoint {checkpoint_path}')

    hparams = torch.load(checkpoint_path, map_location=torch.device('cpu'))['hparams']

    print(f'found hparams: {hparams}')

    exp_class = EXPERIMENT_REGISTRY[hparams['experiment']]

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser = Trainer.add_argparse_args(parser)
    parser.set_defaults(checkpoint_callback=True)

    parser._action_groups[1].title = 'lightning_options'

    args = parser.parse_args(other_args)

    if args.gpus is not None and isinstance(args.gpus, int):
        # Make sure that it only uses a single GPU..
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpus)
        args.gpus = 1

    # TODO: push to lightning
    args.gradient_clip_val = float(args.gradient_clip_val)

    groups = {}
    for group in parser._action_groups:
        group_dict = {a.dest: getattr(args, a.dest, None) for a in group._group_actions}
        groups[group.title] = argparse.Namespace(**group_dict)

    lightning_args = groups['lightning_options']

    trainer = Trainer.from_argparse_args(lightning_args)
    trainer.logger.experiment.log_dir = exp_args.checkpoint_path

    model_class = MODEL_REGISTRY[hparams['model']]

    model_params = {
        k: v for k, v in hparams.items() if (k in inspect.signature(model_class.__init__).parameters
                                             or k in k in inspect.signature(model_class.__bases__[0].__init__).parameters
                                             or k in k in inspect.signature(model_class.__bases__[0].__bases__[0].__init__).parameters)
    }

    print(f'building model with params: {model_params}')

    model = model_class(**model_params)

    experiment = exp_class.load_from_checkpoint(checkpoint_path, pyro_model=model)

    print(f'Loaded {experiment.__class__}:\n{experiment}')

    trainer.test(experiment)
