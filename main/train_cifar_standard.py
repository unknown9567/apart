import torch
import torch.nn.functional as F

from main.utils import LOG_DIR
from main.lightning import CIFARModelBase
from main.lightning import get_base_parser, get_model_and_trainer


class CIFARModel(CIFARModelBase):

    def __init__(self, **kwargs):
        super(CIFARModel, self).__init__()
        self.save_hyperparameters()
        self.model = self.get_model()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        out = self(x)
        loss = F.cross_entropy(out, y)
        self.log('loss', loss, on_step=False, on_epoch=True)
        self.log('acc', self.topk_acc(out, y)[0],
                 on_step=False, on_epoch=True)
        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        x, y = batch
        self.log('val_acc', self.topk_acc(self.model(x), y)[0],
                 on_step=False, on_epoch=True)


def get_parser():
    return get_base_parser(CIFARModel)


def main():
    args = get_parser().parse_args()
    default_root_dir = LOG_DIR / args.dataset / 'standard'
    model, trainer = get_model_and_trainer(CIFARModel, args, default_root_dir)
    trainer.fit(model)


if __name__ == '__main__':
    main()
