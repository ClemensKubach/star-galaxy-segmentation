from lightning import LightningDataModule
from torch.utils.data import DataLoader, random_split

from star_analysis.data.datasets import Sdss


class SdssDataModule(LightningDataModule):

    def __init__(
            self,
            dataset: Sdss,
            batch_size=128,
            shuffle_train=True,
            train_size=0.8,
            val_size=0.1
    ):
        super().__init__()
        assert train_size + val_size <= 1, "train_size + val_size must be smaller than 1"
        self.full_dataset = dataset
        self.batch_size = batch_size
        self.num_workers = 1
        self.shuffle_train = shuffle_train
        self.train_size = train_size
        self.val_size = val_size

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def prepare_data(self):
        pass

    def setup(self, stage):
        num_train = int(len(self.full_dataset) * self.train_size)
        num_val = int(len(self.full_dataset) * self.val_size)
        num_test = len(self.full_dataset) - num_train - num_val
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            self.full_dataset,
            [
                num_train,
                num_val,
                num_test
            ],
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=self.shuffle_train,
            pin_memory=True,
            persistent_workers=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
            persistent_workers=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
            persistent_workers=True
        )
