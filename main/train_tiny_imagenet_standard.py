import torch
import torch.nn.functional as F

from main.utils import LOG_DIR
from main.lightning import TinyImageNetModelBase
from main.lightning import get_base_parser, get_model_and_trainer


class TinyImageNetModel(TinyImageNetModelBase):

    def __init__(self, **kwargs):
        super(TinyImageNetModel, self).__init__()
        self.save_hyperparameters()
        self.model = self.get_model()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        out = self(x)
        loss = F.cross_entropy(out, y)
        top1, top5 = self.topk_acc(out, y, [1, 5])
        self.log('loss', loss, on_step=False, on_epoch=True)
        self.log('top1', top1, on_step=False, on_epoch=True)
        self.log('top5', top5, on_step=False, on_epoch=True)
        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        x, y = batch
        top1, top5 = self.topk_acc(self(x), y, [1, 5])
        self.log('val_top1', top1, on_step=False, on_epoch=True)
        self.log('val_top5', top5, on_step=False, on_epoch=True)


def get_parser():
    return get_base_parser(TinyImageNetModel)


def main():
    args = get_parser().parse_args()
    default_root_dir = LOG_DIR / 'tiny-imagenet' / 'standard'
    model, trainer = get_model_and_trainer(TinyImageNetModel, args, default_root_dir)
    trainer.fit(model)


if __name__ == '__main__':
    main()
