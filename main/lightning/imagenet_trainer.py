import yaml
import argparse
from pathlib import Path

import torch
import pytorch_lightning as pl
from torchvision import transforms as T
from torchvision.datasets import ImageFolder

from .progress_bar import MeterlessProgressBar


class ImageNetModelBase(pl.LightningModule):

    normalize = T.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    def setup(self, stage=None):
        self.train_dataset = \
            ImageFolder(Path(self.hparams.data_dir) / 'train',
                        T.Compose([T.RandomResizedCrop(224),
                                   T.RandomHorizontalFlip(),
                                   T.ToTensor(), self.normalize]))
        self.val_dataset = \
            ImageFolder(Path(self.hparams.data_dir) / 'val',
                        T.Compose([T.Resize(256), T.CenterCrop(224),
                                   T.ToTensor(), self.normalize]))

    def configure_optimizers(self):
        args = self.hparams
        optimizer = torch.optim.SGD(self.parameters(), lr=args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            [int(epoch * args.epochs / 105) for epoch in [30, 60, 90, 100]],
            0.1
        )
        return [optimizer], [scheduler]

    def train_dataloader(self):
        args = self.hparams
        return torch.utils.data.DataLoader(
            self.train_dataset, shuffle=True,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=getattr(args, 'drop_last', False)
        )

    def val_dataloader(self):
        args = self.hparams
        return torch.utils.data.DataLoader(
            self.val_dataset, shuffle=False, batch_size=args.batch_size,
            num_workers=args.num_workers, pin_memory=True
        )

    @staticmethod
    def topk_acc(out, y, topk=(1,)):
        _, indices = out.topk(max(topk), -1)
        acc_list = []
        for k in topk:
            is_correct = ((indices[:, :k].unsqueeze(-1) ==
                           y.view(-1, 1, 1)).long().sum(1) > 0)
            acc_list.append(is_correct.float().mean())
        return acc_list

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group('ImageNetModelBase')
        parser.add_argument('-b', '--batch_size', default=256, type=int)
        parser.add_argument('--epochs', default=90, type=int)
        parser.add_argument('--num_workers', default=8, type=int)
        parser.add_argument('--lr', default=0.1, type=float)
        parser.add_argument('--momentum', default=0.9, type=float)
        parser.add_argument('--weight_decay', default=1e-4, type=float)
        return parent_parser


def get_imagenet_base_parser(model_cls):
    parser = argparse.ArgumentParser()
    # base
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--resume_dir', default='', type=str)
    parser.add_argument('--data_dir', required=True, type=str)
    parser.add_argument('--seed', default=42, type=int)
    # model
    parser = model_cls.add_model_specific_args(parser)
    # trainer
    parser = pl.Trainer.add_argparse_args(parser)
    # use GPUs by default
    if torch.cuda.is_available():
        parser.set_defaults(gpus=1, strategy='ddp')
    return parser


def get_imagenet_model_and_trainer(model_cls, args, default_root_dir, epochs=None):
    args.max_epochs = epochs or args.epochs
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
        model = model_cls.load_from_checkpoint(checkpoint_path, **vars(args))
    else:
        model = model_cls(**vars(args))
    trainer = pl.Trainer.from_argparse_args(args, callbacks=[MeterlessProgressBar()])
    return model, trainer
