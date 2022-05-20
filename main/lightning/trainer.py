import sys
import yaml
import argparse
from tqdm import tqdm
from pathlib import Path

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ProgressBar


class MeterlessProgressBar(ProgressBar):

    def init_train_tqdm(self):
        bar = tqdm(
            desc='Training', initial=self.train_batch_idx,
            position=(2 * self.process_position), disable=self.is_disabled,
            leave=True, dynamic_ncols=False, ncols=100, file=sys.stdout,
            smoothing=0,
        )
        return bar

    def init_validation_tqdm(self):
        has_main_bar = self.main_progress_bar is not None
        bar = tqdm(
            desc='Validating', position=(2 * self.process_position + has_main_bar),
            disable=self.is_disabled, leave=False, dynamic_ncols=False,
            ncols=100, file=sys.stdout
        )
        return bar

    def init_test_tqdm(self):
        bar = tqdm(
            desc="Testing", position=(2 * self.process_position),
            disable=self.is_disabled, leave=True, dynamic_ncols=False,
            ncols=100, file=sys.stdout
        )
        return bar


def get_base_parser(model_cls):
    parser = argparse.ArgumentParser()
    # base
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--resume_dir', default='', type=str)
    parser.add_argument('--seed', default=42, type=int)
    # model
    parser = model_cls.add_model_specific_args(parser)
    # trainer
    parser = pl.Trainer.add_argparse_args(parser)
    # use GPUs by default
    if torch.cuda.is_available():
        parser.set_defaults(gpus=1, strategy='ddp')
    return parser


def get_model_and_trainer(lightning_model_cls, args, default_root_dir):
    args.max_epochs = args.epochs
    args.default_root_dir = str(default_root_dir)
    if args.strategy == 'ddp' or args.accelerator == 'ddp':
        args.batch_size = int(args.batch_size / max(1, args.gpus))
        args.num_workers = int(args.num_workers / max(1, args.gpus))
        args.sync_batchnorm = (args.gpus > 1)
    pl.seed_everything(args.seed)
    if args.resume:
        assert args.resume_dir
        checkpoint_path = sorted(Path(args.resume_dir).rglob('*.ckpt'),
                                 key=lambda x: int(x.stem.split('-')[0][6:]))[-1]
        with open(Path(args.resume_dir) / 'hparams.yaml', 'r') as f:
            args.__dict__.update(yaml.load(f, Loader=yaml.FullLoader))
        args.resume_from_checkpoint = str(checkpoint_path)
        model = lightning_model_cls.load_from_checkpoint(checkpoint_path, **vars(args))
    else:
        model = lightning_model_cls(**vars(args))
    trainer = pl.Trainer.from_argparse_args(args, callbacks=[MeterlessProgressBar()])
    return model, trainer
