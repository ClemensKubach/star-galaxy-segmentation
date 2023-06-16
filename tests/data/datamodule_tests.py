import unittest

import torch
from tqdm import tqdm

from star_analysis.data.configs import SdssDatasetConfig, SdssDataModuleConfig
from star_analysis.data.datamodules import SdssDataModule
from star_analysis.dataprovider.sdss_dataprovider import SDSSDataProvider


class DataModuleTestCase(unittest.TestCase):
    def setUp(self):
        self.batch_size = 32
        self.shuffle_train = True
        dataset_config = SdssDatasetConfig(
            patch_shape=(5, 5),
            prepare=True,
            run=SDSSDataProvider.FIXED_VALIDATION_RUN
        )
        module_config = SdssDataModuleConfig(
            dataset_config=dataset_config,
            batch_size=self.batch_size,
            shuffle_train=self.shuffle_train,
            train_size=0.8,
            num_workers=1
        )
        self.datamodule = SdssDataModule(module_config)
        self.datamodule.prepare_data()
        self.datamodule.setup("fit")

    def test_loading(self):
        train_loader = self.datamodule.train_dataloader()
        train_loader_iter = iter(train_loader)
        batch_x, batch_y = next(train_loader_iter)
        assert batch_x.shape == (self.batch_size, 5, 5)
        #assert batch_y.shape == (self.batch_size, 5, 5)

    def test_exclusivity(self):
        train_loader = self.datamodule.train_dataloader()
        val_loader = self.datamodule.val_dataloader()
        test_loader = self.datamodule.test_dataloader()

        train_set = set(train_loader.dataset.indices)
        val_set = set(val_loader.dataset.indices)
        test_set = set(test_loader.dataset.indices)
        in_all = train_set & val_set & test_set
        assert len(in_all) == 0

    def test_exclusivity_on_sample(self):
        train_loader = self.datamodule.train_dataloader()
        val_loader = self.datamodule.val_dataloader()
        test_loader = self.datamodule.test_dataloader()

        last_train_item = train_loader.dataset[-1]
        x, _ = last_train_item
        for i in tqdm(range(len(val_loader.dataset))):
            x_val, _ = val_loader.dataset[i]
            assert torch.equal(x, x_val)

        for i in tqdm(range(len(test_loader.dataset))):
            x_test, _ = test_loader.dataset[i]
            assert torch.equal(x, x_test)


    def test_deterministic_shuffling(self):
        pass

    def test_equal_label_distribution(self):
        pass


if __name__ == '__main__':
    #unittest.main()
    d = DataModuleTestCase()
    d.setUp()
    d.test_exclusivity_on_sample()
