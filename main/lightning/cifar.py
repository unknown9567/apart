import yaml
import argparse
from pathlib import Path

import torch
import pytorch_lightning as pl

from main import models
from main.utils import DATA_DIR, get_func_kwargs


class CIFARModelBase(pl.LightningModule):

    def get_model(self):
        args = self.hparams
        model = args.model.lower()
        weight_path = DATA_DIR / args.dataset / f'{model}_seed_{args.seed}.pth'
        model_cls = {k.lower(): v for k, v in models.__dict__.items()}[model]
        model_args = get_func_kwargs(model_cls, vars(args))
        model_args['num_classes'] = 10 if args.dataset == 'cifar10' else 100
        model = model_cls(**model_args)
        weight_path.parent.mkdir(parents=True, exist_ok=True)
        if weight_path.is_file():
            model.load_state_dict(torch.load(weight_path, 'cpu'))
        else:
            torch.save(model.state_dict(), weight_path)
        return model

    def configure_optimizers(self):
        args = self.hparams
        optimizer = torch.optim.SGD(
            self.parameters(), lr=args.lr,
            momentum=args.momentum, weight_decay=args.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, [int(0.5 * args.epochs), int(0.75 * args.epochs)], 0.1
        )
        return [optimizer], [scheduler]

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group('CIFARModelBase')
        parser.add_argument('-b', '--batch_size', default=128, type=int)
        parser.add_argument('--epochs', default=200, type=int)
        parser.add_argument('--num_workers', default=4, type=int)
        parser.add_argument('--lr', default=0.1, type=float)
        parser.add_argument('--momentum', default=0.9, type=float)
        parser.add_argument('--weight_decay', default=5e-4, type=float)
        return parent_parser
