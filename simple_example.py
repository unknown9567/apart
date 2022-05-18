from tqdm import tqdm
from argparse import ArgumentParser

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import models
from torchvision import datasets
from torchvision import transforms as T

from apart import APART


r"""Simple example of deploying APART by additional FOUR lines to train a CIFAR model"""


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def parse_args():
    parser = ArgumentParser()
    # Basic settings
    parser.add_argument('--dataset', default='cifar100', type=str,
                        choices=['cifar10', 'cifar100'])
    parser.add_argument('--model', default='resnet18', type=str)
    parser.add_argument('--lr', default=0.1, type=float,
                        help='learning rate of SGD')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='momentum of SGD')
    parser.add_argument('--nesterov', action='store_true',
                        help='enables Nesterov momentum in SGD')
    parser.add_argument('--weight_decay', default=5e-4, type=float)
    parser.add_argument('--milestones', nargs='*', default=[100, 150],
                        type=int, help='list of epoch indices. Must be increasing')
    parser.add_argument('--gamma', default=0.1, type=float,
                        help='multiplicative factor of learning rate decay')
    # APART
    parser.add_argument('--epsilon', default=0.1, type=float,
                        help='perturbation radius')
    parser.add_argument('--groups', default=16, type=int,
                        help='group number')
    return parser.parse_args()


def get_dataloader(args):
    normalize = T.Normalize([0.4914, 0.4822, 0.4465], [0.2471, 0.2435, 0.2616])
    # standard augmentation for CIFAR
    transforms = [T.RandomCrop(32, 4), T.RandomHorizontalFlip(), T.ToTensor(), normalize]
    dataset_cls = {'cifar10': datasets.CIFAR10, 'cifar100': datasets.CIFAR100}[args.dataset]
    train_dataset = dataset_cls('./data/dataset', True, T.Compose(transforms), download=True)
    test_dataset = dataset_cls('./data/dataset', False, T.Compose([T.ToTensor(), normalize]), download=True)
    # drop the last batch to avoid incorrectly grouping samples
    return DataLoader(train_dataset, shuffle=True, batch_size=128, num_workers=2, drop_last=True), \
           DataLoader(test_dataset, shuffle=True, batch_size=128, num_workers=2)


def get_model(args):
    # get the model class from `torchvision.models`
    model_cls = {k.lower(): v for k, v in vars(models).items()}[args.model.lower()]
    model = model_cls(num_classes={'cifar10': 10, 'cifar100': 100}[args.dataset])
    if isinstance(model, models.ResNet):
        # replace some layers to get a CIFAR-based version of resnet
        model.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        model.maxpool = nn.Sequential()
    return model.to(DEVICE)


def get_optimizer(args, model):
    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.lr, momentum=args.momentum,
        weight_decay=args.weight_decay, nesterov=args.nesterov
    )
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, args.milestones, args.gamma
    )
    return optimizer, lr_scheduler


def eval_acc(pred, target):
    return (pred.argmax(-1) == target).float().sum().item()


def main():
    # basic experimental settings
    args = parse_args()
    train_loader, test_loader = get_dataloader(args)
    normal_model, apart_model = get_model(args), get_model(args)
    apart_model.load_state_dict(normal_model.state_dict())
    normal_optimizer, normal_lr_scheduler = get_optimizer(args, normal_model)
    apart_optimizer, apart_lr_scheduler = get_optimizer(args, apart_model)
    # instantiate APART with specified perturbation radius and group number
    apart = APART(apart_model, args.epsilon, args.groups).to(DEVICE)
    for epoch in range(200):
        sample_num, normal_acc, apart_acc = 0.0, 0.0, 0.0
        normal_model.train(), apart_model.train()
        with tqdm(train_loader, ncols=100) as pbar:
            for x, y in pbar:
                x, y = x.to(DEVICE), y.to(DEVICE)
                sample_num += x.size(0)

                # forward and backward passes in normal training
                normal_model.zero_grad()
                pred = normal_model(x)
                F.cross_entropy(pred, y).backward()
                normal_optimizer.step()
                normal_acc += eval_acc(pred, y)

                # forward and backward passes in APART
                # with only THREE additional lines for APART
                apart_model.zero_grad()
                with apart.to_proxy():  # prepare for APART's first step
                    # normal forward and backward passes with scaling the loss by 0.5
                    pred = apart_model(x)
                    (0.5 * F.cross_entropy(pred, y)).backward()
                with apart.to_adver():  # prepare for APART's second step
                    # perform forward and backward passes again with scaling the loss by (1 - 0.5)
                    ((1 - 0.5) * F.cross_entropy(apart_model(x), y)).backward()
                apart_optimizer.step()
                apart_acc += eval_acc(pred, y)

                pbar.set_description(f'epoch {epoch}: '
                                     f'normal: {normal_acc/sample_num*100:.2f}% | ' 
                                     f'apart: {apart_acc/sample_num*100:.2f}%')

        sample_num, normal_acc, apart_acc = 0.0, 0.0, 0.0
        normal_model.eval(), apart_model.eval()
        with tqdm(test_loader, ncols=100) as pbar:
            for x, y in pbar:
                x, y = x.to(DEVICE), y.to(DEVICE)
                sample_num += x.size(0)

                with torch.no_grad():
                    normal_acc += eval_acc(normal_model(x), y)

                with torch.no_grad():
                    apart_acc += eval_acc(apart_model(x), y)

                pbar.set_description(f'eval: '
                                     f'normal: {normal_acc / sample_num * 100:.2f}% | '
                                     f'apart: {apart_acc / sample_num * 100:.2f}% | '
                                     f'gap: {(apart_acc - normal_acc) / sample_num * 100:.2f}%')

        normal_lr_scheduler.step(), apart_lr_scheduler.step()


if __name__ == '__main__':
    main()



