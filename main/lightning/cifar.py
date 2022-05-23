import torch
import pytorch_lightning as pl
from torchvision import datasets
from torchvision import transforms as T

from main import models
from main.utils import DATA_DIR, get_func_kwargs


class CIFAR(torch.utils.data.Dataset):

    normalize = T.Normalize([0.4914, 0.4822, 0.4465], [0.2471, 0.2435, 0.2616])

    def __init__(self, train, augmentation=False, dataset='cifar10'):
        dataset_cls = datasets.__dict__[dataset.upper()]
        transforms = []
        if augmentation:
            transforms.extend([T.RandomCrop(32, 4), T.RandomHorizontalFlip()])
        transforms.append(T.ToTensor())
        transforms.append(self.normalize)
        self.dataset = dataset_cls(
            str(DATA_DIR / 'datasets' / 'cifar'), download=True, train=train,
            transform=T.Compose(transforms)
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        return self.dataset[item]

    def __iter__(self):
        return iter(self.dataset)


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

    def setup(self, stage=None):
        self.train_dataset = CIFAR(True, True, self.hparams.dataset)
        self.val_dataset = CIFAR(False, False, self.hparams.dataset)

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

    def on_epoch_end(self):
        super(CIFARModelBase, self).on_epoch_end()
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
        parser = parent_parser.add_argument_group('CIFARModelBase')
        parser.add_argument('--dataset', required=True, type=str,
                            choices=['cifar10', 'cifar100'])
        parser.add_argument('--model', default='WideResNet40_2', type=str)
        parser.add_argument('-b', '--batch_size', default=128, type=int)
        parser.add_argument('--epochs', default=200, type=int)
        parser.add_argument('--num_workers', default=4, type=int)
        parser.add_argument('--lr', default=0.1, type=float)
        parser.add_argument('--momentum', default=0.9, type=float)
        parser.add_argument('--weight_decay', default=5e-4, type=float)
        return parent_parser
