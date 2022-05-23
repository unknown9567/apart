from pathlib import Path

import torch
import pytorch_lightning as pl
from torchvision import transforms as T
from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import pil_loader

from main import models
from main.utils import DATA_DIR, get_func_kwargs


# store all images loaded in the first epoch
class TinyImagenetLoader(object):

    def __init__(self):
        self.images = dict()

    def __call__(self, path):
        path = str(path)
        image = self.images.get(path, None)
        if image is None:
            image = pil_loader(path)
            self.images[path] = image
        return image


class TinyImageNetModelBase(pl.LightningModule):

    normalize = T.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    def get_model(self):
        args = self.hparams
        model = args.model.lower()
        weight_path = DATA_DIR / 'tiny-imagenet' / f'{model}_seed_{args.seed}.pth'
        model_cls = {k.lower(): v for k, v in vars(models).items()}[model]
        model = model_cls(**get_func_kwargs(model_cls, vars(args)), num_classes=200)
        weight_path.parent.mkdir(parents=True, exist_ok=True)
        if weight_path.is_file():
            model.load_state_dict(torch.load(weight_path, 'cpu'))
        else:
            torch.save(model.state_dict(), weight_path)
        return model

    def setup(self, stage=None):
        self.train_dataset = \
            ImageFolder(Path(self.hparams.data_dir) / 'train',
                        T.Compose([T.RandomCrop(64, padding=4),
                                   T.RandomHorizontalFlip(),
                                   T.ToTensor(), self.normalize]),
                        loader=TinyImagenetLoader())
        self.val_dataset = \
            ImageFolder(Path(self.hparams.data_dir) / 'val',
                        T.Compose([T.ToTensor(), self.normalize]),
                        loader=TinyImagenetLoader())

    def configure_optimizers(self):
        args = self.hparams
        optimizer = torch.optim.SGD(self.parameters(), lr=args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, [int(0.5 * args.epochs), int(0.75 * args.epochs)], 0.1
        )
        return [optimizer], [scheduler]

    def on_epoch_end(self):
        super(TinyImageNetModelBase, self).on_epoch_end()
        torch.cuda.empty_cache()

    def train_dataloader(self):
        args = self.hparams
        return torch.utils.data.DataLoader(
            self.train_dataset, shuffle=True,
            batch_size=args.batch_size, num_workers=args.num_workers,
            pin_memory=False, drop_last=getattr(args, 'drop_last', False)
        )

    def val_dataloader(self):
        args = self.hparams
        return torch.utils.data.DataLoader(
            self.val_dataset, shuffle=False, batch_size=args.batch_size,
            num_workers=args.num_workers, pin_memory=False
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
        parser = parent_parser.add_argument_group('TinyImageNetModelBase')
        parser.add_argument('--data_dir', required=True, type=str)
        parser.add_argument('-b', '--batch_size', default=256, type=int)
        parser.add_argument('--epochs', default=200, type=int)
        parser.add_argument('--num_workers', default=4, type=int)
        parser.add_argument('--lr', default=0.1, type=float)
        parser.add_argument('--momentum', default=0.9, type=float)
        parser.add_argument('--weight_decay', default=5e-4, type=float)
        return parent_parser
