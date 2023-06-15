import unittest

from star_analysis.data.configs import SdssDatasetConfig, SdssDataModuleConfig
from star_analysis.data.datamodules import SdssDataModule
from star_analysis.dataprovider.sdss_dataprovider import SDSSDataProvider


class DataModuleTestCase(unittest.TestCase):
    def setUp(self):
        self.batch_size = 32
        self.shuffle_train = True
        dataset_config = SdssDatasetConfig(
            patch_shape=(5, 5),
            prepare=False,
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
        pass

    def test_deterministic_shuffling(self):
        pass

    def test_equal_label_distribution(self):
        pass


if __name__ == '__main__':
    unittest.main()
