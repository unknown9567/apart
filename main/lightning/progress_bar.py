import sys
from tqdm import tqdm
from pytorch_lightning.callbacks import ProgressBar


class MeterlessProgressBar(ProgressBar):

    def init_train_tqdm(self):
        bar = tqdm(
            desc='Training',
            initial=self.train_batch_idx,
            position=(2 * self.process_position),
            disable=self.is_disabled,
            leave=True,
            dynamic_ncols=False,
            ncols=100,
            file=sys.stdout,
            smoothing=0,
        )
        return bar

    def init_validation_tqdm(self):
        has_main_bar = self.main_progress_bar is not None
        bar = tqdm(
            desc='Validating',
            position=(2 * self.process_position + has_main_bar),
            disable=self.is_disabled,
            leave=False,
            dynamic_ncols=False,
            ncols=100,
            file=sys.stdout
        )
        return bar

    def init_test_tqdm(self):
        bar = tqdm(
            desc="Testing",
            position=(2 * self.process_position),
            disable=self.is_disabled,
            leave=True,
            dynamic_ncols=False,
            ncols=100,
            file=sys.stdout
        )
        return bar
