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
        parser.add_argument('--data_dir', required=True, type=str)
        parser.add_argument('-b', '--batch_size', default=256, type=int)
        parser.add_argument('--epochs', default=105, type=int)
        parser.add_argument('--num_workers', default=8, type=int)
        parser.add_argument('--lr', default=0.1, type=float)
        parser.add_argument('--momentum', default=0.9, type=float)
        parser.add_argument('--weight_decay', default=1e-4, type=float)
        return parent_parser
