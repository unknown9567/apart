import yaml
import argparse
from pathlib import Path

import torch
import pytorch_lightning as pl
from .progress_bar import MeterlessProgressBar


class CIFARModel(pl.LightningModule):

    def configure_optimizers(self):
        args = self.hparams
        optimizer = SAM(self.model.parameters(), torch.optim.SGD,
                        rho=(1.0 + args.adv_ratio) * args.rho,
                        adaptive=args.adaptive, lr=args.lr_max,
                        momentum=0.9, weight_decay=args.weight_decay)
        epochs = args.epochs
        if args.more_lr_decay:
            milestones = [int(0.3 * epochs), int(0.6 * epochs), int(0.8 * epochs)]
            gamma = 0.2
        else:
            milestones = [int(0.5 * epochs), int(0.75 * epochs)]
            gamma = 0.1
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma)
        return [optimizer], [scheduler]


def get_cifar_base_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--resume_dir', type=str)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    return parser


def get_cifar_model_and_trainer(model_cls, args, default_root_dir, epochs=None):
    if torch.cuda.is_available():
        gpus, strategy = 1, 'dp'
    else:
        gpus, strategy = 0, 'cpu'
    pl.seed_everything(args.seed)
    if args.resume:
        assert args.resume_dir
        checkpoint_dir = Path(args.resume_dir) / 'checkpoints'
        checkpoint_path = sorted(checkpoint_dir.glob('*.ckpt'),
                                 key=lambda x: int(x.stem.split('-')[0][6:]))[-1]
        with open(Path(args.resume_dir) / 'hparams.yaml', 'r') as f:
            args.__dict__.update(yaml.load(f, Loader=yaml.FullLoader))
        model = model_cls.load_from_checkpoint(checkpoint_path, **args.__dict__)
        trainer = pl.Trainer(
            default_root_dir=default_root_dir, gpus=gpus, strategy=strategy,
            resume_from_checkpoint=checkpoint_path, max_epochs=(epochs or args.epochs),
            callbacks=[MeterlessProgressBar()]
        )
    else:
        model = model_cls(**args.__dict__)
        trainer = pl.Trainer(
            default_root_dir=default_root_dir, gpus=gpus, strategy=strategy,
            max_epochs=(epochs or args.epochs), callbacks=[MeterlessProgressBar()]
        )
    return model, trainer
