from lightning import LightningDataModule
from torch.utils.data import DataLoader, random_split

from star_analysis.data.configs import SdssDataModuleConfig
from star_analysis.data.datasets import Sdss


class SdssDataModule(LightningDataModule):

    def __init__(
            self,
            config: SdssDataModuleConfig
    ):
        super().__init__()
        assert config.train_size < 1, "train_size must be smaller than 1"

        self.config = config

        self.full_dataset = Sdss(config.dataset_config)
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def prepare_data(self):
        self.full_dataset.prepare()

    def setup(self, stage):
        num_train = int(len(self.full_dataset) * self.config.train_size)
        num_val = len(self.full_dataset) - num_train
        self.train_dataset, self.val_dataset = random_split(
            self.full_dataset,
            [
                num_train,
                num_val
            ],
        )
        # TODO setup test dataset and create test for similiarity check of distributions

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            shuffle=self.config.shuffle_train,
            pin_memory=True,
            persistent_workers=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            shuffle=False,
            pin_memory=True,
            persistent_workers=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            shuffle=False,
            pin_memory=True,
            persistent_workers=True
        )
