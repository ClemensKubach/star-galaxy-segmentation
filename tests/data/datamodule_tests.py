import os
import unittest

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from star_analysis.data.configs import SdssDatasetConfig, SdssDataModuleConfig
from star_analysis.data.datamodules import SdssDataModule
from star_analysis.dataprovider.sdss_dataprovider import SDSSDataProvider


class DataModuleTestCase(unittest.TestCase):
    def setUp(self):
        self.batch_size = 32
        self.shuffle_train = False
        dataset_config = SdssDatasetConfig(
            patch_shape=None,
            prepare=True,
            run=SDSSDataProvider.FIXED_VALIDATION_RUN
        )
        module_config = SdssDataModuleConfig(
            dataset_config=dataset_config,
            batch_size=self.batch_size,
            shuffle_train=self.shuffle_train,
            train_size=0.8,
            num_workers=4
        )
        self.datamodule = SdssDataModule(module_config)
        self.datamodule.prepare_data()
        self.datamodule.setup("fit")

    def test_loading(self):
        train_loader = self.datamodule.train_dataloader()
        for e in train_loader:
            x, y = e
            assert x.shape == (self.batch_size, 224, 224, 5)
            assert y.shape == (self.batch_size, 224, 224, 2)
            break

    def test_exclusivity_train_subsets(self):
        train_loader = self.datamodule.train_dataloader()
        val_loader = self.datamodule.val_dataloader()

        train_set = set(train_loader.dataset.indices)
        val_set = set(val_loader.dataset.indices)
        in_both = train_set & val_set
        assert len(in_both) == 0

    def test_exclusivity_test_set(self):
        test_loader = self.datamodule.test_dataloader()
        train_loader = self.datamodule.train_dataloader()
        val_loader = self.datamodule.val_dataloader()
        for e in tqdm(test_loader.dataset):
            x, _ = e
            assert self.__value_exclusivity(test_loader, x) == 1
            assert self.__value_exclusivity(train_loader, x) == 0
            assert self.__value_exclusivity(val_loader, x) == 0

    def test_exclusivity_train_set(self):
        loader = DataLoader(
            self.datamodule.train_dataset,
            batch_size=1,
            num_workers=1,
            shuffle=False,
            pin_memory=True,
            persistent_workers=True,
        )

        o = []
        for e in tqdm(loader):
            x, _ = e
            #assert self.__value_exclusivity(test_loader, x) == 0
            #assert self.__value_exclusivity(train_loader, x) == 1
            #assert self.__value_exclusivity(val_loader, x) == 0
            t = self.__value_exclusivity(self.datamodule.train_dataloader(), x)
            o.append(t)
        print(o)

    def test_exclusivity_val_set(self):
        test_loader = self.datamodule.test_dataloader()
        train_loader = self.datamodule.train_dataloader()
        val_loader = self.datamodule.val_dataloader()
        for e in tqdm(val_loader.dataset):
            x, _ = e
            assert self.__value_exclusivity(test_loader, x) == 0
            assert self.__value_exclusivity(train_loader, x) == 0
            assert self.__value_exclusivity(val_loader, x) == 1

    def __value_exclusivity(self, loader: DataLoader, x_compare: torch.Tensor):
        num_equals = 0
        for batch in loader:
            xb, _ = batch
            for x in xb:
                if torch.equal(x, x_compare):
                    num_equals += 1
        return num_equals

    def test_deterministic_shuffling(self):
        pass

    def test_equal_label_distribution(self):
        pass


if __name__ == '__main__':
    unittest.main()
