import torch
import torch.nn.functional as F

from apart import APART
from main.sam import SAM
from main.utils import LOG_DIR
from main.lightning import CIFARModelBase
from main.lightning import get_base_parser, get_model_and_trainer


class CIFARModel(CIFARModelBase):

    def __init__(self, **kwargs):
        super(CIFARModel, self).__init__()
        self.save_hyperparameters()
        assert self.hparams.ratio > 0.0
        self.model = self.get_model()
        self.apart = APART(self.model, self.hparams.epsilon, self.hparams.groups)
        self.automatic_optimization = False

    def configure_optimizers(self):
        args = self.hparams
        optimizer = SAM(
            self.parameters(), torch.optim.SGD, args.rho, adaptive=False,
            lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, [int(0.5 * args.epochs), int(0.75 * args.epochs)], 0.1
        )
        return [optimizer], [scheduler]

    def on_train_start(self):
        if self.hparams.sync_batchnorm:
            self.apart = APART(self.model, self.hparams.epsilon, self.hparams.groups)
        super(CIFARModel, self).on_train_start()

    def forward(self, x, mode='original'):
        if mode == 'original':
            return self.model(x)
        elif mode == 'proxy':
            with self.apart.to_proxy():
                return self.model(x)
        elif mode == 'adver':
            with self.apart.to_adver():
                return self.model(x)
        else:
            raise Exception(f'Unknown mode: {mode}')

    def training_step(self, batch, batch_idx):
        self.zero_grad()
        x, y = batch
        ratio = self.hparams.ratio

        out = self(x, 'proxy')
        loss = F.cross_entropy(out, y)
        self.log('loss', loss, on_step=False, on_epoch=True)
        self.log('acc', self.topk_acc(out, y)[0], on_step=False, on_epoch=True)
        self.manual_backward(1.0 / (1.0 + ratio) * loss)
        self.optimizers().first_step(zero_grad=False)

        split = int(ratio * x.size(0))
        self.manual_backward(
            ratio / (1.0 + ratio) * F.cross_entropy(self(x[:split], 'adver'), y[:split])
        )
        self.optimizers().second_step(zero_grad=False)

        if self.trainer.is_last_batch:
            self.lr_schedulers().step()

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        x, y = batch
        self.log('val_acc', self.topk_acc(self.model(x), y)[0],
                 on_step=False, on_epoch=True)


def get_parser():
    parent_parser = get_base_parser(CIFARModel)
    # APART parameters
    parser = parent_parser.add_argument_group('APART')
    parser.add_argument('-r', '--ratio', default=0.0, type=float,
                        help='ratio of sample numbers in APART`s two steps')
    parser.add_argument('-eps', '--epsilon', default=0.1, type=float,
                        help='perturbation radius of APART')
    parser.add_argument('-g', '--groups', default=0, type=int,
                        help='group number of APART')
    # SAM parameters
    parser = parent_parser.add_argument_group('SAM')
    parser.add_argument('--rho', default=0.1, type=float,
                        help='perturbation radius of SAM')
    return parent_parser


def main():
    args = get_parser().parse_args()
    args.drop_last = (args.groups > 1)
    default_root_dir = LOG_DIR / args.dataset / 'apart-sam'
    model, trainer = get_model_and_trainer(CIFARModel, args, default_root_dir)
    trainer.fit(model)


if __name__ == '__main__':
    main()
