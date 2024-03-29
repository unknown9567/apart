import torch
import torch.nn.functional as F

from apart import APART
from main.utils import LOG_DIR
from main.lightning import ImageNetModelBase
from main.lightning import get_base_parser, get_model_and_trainer


class ImageNetModel(ImageNetModelBase):

    def __init__(self, **kwargs):
        super(ImageNetModel, self).__init__()
        self.save_hyperparameters()
        assert self.hparams.ratio > 0.0
        self.model = self.get_model()
        self.apart = APART(self.model, self.hparams.epsilon, self.hparams.groups)
        self.automatic_optimization = False

    def on_train_start(self):
        if self.hparams.sync_batchnorm:
            self.apart = APART(self.model, self.hparams.epsilon, self.hparams.groups)
        super(ImageNetModel, self).on_train_start()

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
        top1, top5 = self.topk_acc(out, y, [1, 5])
        self.log('loss', loss, on_step=False, on_epoch=True)
        self.log('top1', top1, on_step=False, on_epoch=True)
        self.log('top5', top5, on_step=False, on_epoch=True)
        self.manual_backward(1.0 / (1.0 + ratio) * loss)

        split = int(ratio * x.size(0))
        self.manual_backward(
            ratio / (1.0 + ratio) *
            F.cross_entropy(self(x[:split], 'adver'), y[:split]))

        self.optimizers().step()
        if self.trainer.is_last_batch:
            self.lr_schedulers().step()

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        x, y = batch
        top1, top5 = self.topk_acc(self(x), y, [1, 5])
        self.log('val_top1', top1, on_step=False, on_epoch=True)
        self.log('val_top5', top5, on_step=False, on_epoch=True)


def get_parser():
    parent_parser = get_base_parser(ImageNetModel)
    # APART parameters
    parser = parent_parser.add_argument_group('APART')
    parser.add_argument('-r', '--ratio', default=1.0, type=float,
                        help='ratio of sample numbers in APART`s two steps')
    parser.add_argument('-eps', '--epsilon', default=0.1, type=float,
                        help='perturbation radius of APART')
    parser.add_argument('-g', '--groups', default=0, type=int,
                        help='group number of APART')
    return parent_parser


def main():
    args = get_parser().parse_args()
    args.drop_last = (args.groups > 1)
    default_root_dir = LOG_DIR / 'imagenet' / 'apart'
    model, trainer = get_model_and_trainer(ImageNetModel, args, default_root_dir)
    trainer.fit(model)


if __name__ == '__main__':
    main()
