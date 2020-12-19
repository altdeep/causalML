from deepscm.experiments import toonfaces  # noqa: F401
from .base_experiment import EXPERIMENT_REGISTRY, MODEL_REGISTRY

if __name__ == '__main__':
    from pytorch_lightning import Trainer
    from pytorch_lightning.logging import TensorBoardLogger
    import argparse
    import os

    exp_parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    exp_parser.add_argument('--experiment', '-e', help='which experiment to load', choices=tuple(EXPERIMENT_REGISTRY.keys()))
    exp_parser.add_argument('--model', '-m', help='which model to load', choices=tuple(MODEL_REGISTRY.keys()))

    exp_args, other_args = exp_parser.parse_known_args()

    exp_class = EXPERIMENT_REGISTRY[exp_args.experiment]
    model_class = MODEL_REGISTRY[exp_args.model]

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser = Trainer.add_argparse_args(parser)
    parser.set_defaults(checkpoint_callback=True)

    parser._action_groups[1].title = 'lightning_options'

    experiment_group = parser.add_argument_group('experiment')
    exp_class.add_arguments(experiment_group)

    model_group = parser.add_argument_group('model')
    model_class.add_arguments(model_group)

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

    logger = TensorBoardLogger(lightning_args.default_root_dir, name=f'{exp_args.experiment}/{exp_args.model}')
    lightning_args.logger = logger

    hparams = groups['experiment']
    model_params = groups['model']

    for k, v in vars(model_params).items():
        setattr(hparams, k, v)

    trainer = Trainer.from_argparse_args(lightning_args)

    model = model_class(**vars(model_params))
    experiment = exp_class(hparams, model)

    trainer.fit(experiment)
