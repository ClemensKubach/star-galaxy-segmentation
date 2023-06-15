from typing import Any

from lightning import LightningDataModule, Trainer
from torch.utils.data import DataLoader

from star_analysis.data.configs import SdssDatasetConfig, SdssDataModuleConfig
from star_analysis.data.datamodules import SdssDataModule
from star_analysis.dataprovider.sdss_dataprovider import SDSSDataProvider
from star_analysis.model.types import ModelTypes
from star_analysis.runner.executable import Executable
from star_analysis.utils.constants import CHECKPOINT_DIR


class SdssRunner(Executable):

    def __init__(
            self,
            data_dir: str,
            model_type: ModelTypes = ModelTypes.FCN,
            project_name: str = "sdss-tests",
            batch_size: int = 32,
            learning_rate_init: float = 1e-3,

            transform: Any = None,
            target_transform: Any = None,
            shuffle_train: bool = True,
            train_size: float = 0.8,
            workers: int = 1,
            patch_size: int = 224,
    ):
        super().__init__(
            data_dir=data_dir,
            model_type=model_type,
            project_name=project_name,
            batch_size=batch_size,
            learning_rate_init=learning_rate_init
        )
        dataset_config = SdssDatasetConfig(
            data_dir=self.data_dir,
            patch_shape=(patch_size, patch_size),
            prepare=False,
            run=SDSSDataProvider.FIXED_VALIDATION_RUN,
            transform=transform,
            target_transform=target_transform
        )
        self.module_config = SdssDataModuleConfig(
            dataset_config=dataset_config,
            batch_size=self.batch_size,
            shuffle_train=shuffle_train,
            train_size=train_size,
            num_workers=workers
        )
        self.trainer = None

    def _setup_data(self) -> LightningDataModule:
        return SdssDataModule(self.module_config)

    def train(
            self,
            max_epochs=10,
            limit_train_batches=200,
            limit_val_batches=100,
    ):
        self.trainer = Trainer(
            max_epochs=max_epochs,
            logger=self.logger,
            num_nodes=-1,
            limit_train_batches=limit_train_batches,
            limit_val_batches=limit_val_batches,
            enable_checkpointing=True,
            callbacks=None,  # [early_stopping]
            default_root_dir=CHECKPOINT_DIR
        )
        self.trainer.fit(
            model=self.model,
            datamodule=self.data_module
        )

    def test(self):
        self.trainer.test(
            ckpt_path="best"
        )

    def predict(self, data_loader: DataLoader):
        self.trainer.predict(
            model=self.model,
            dataloaders=data_loader
        )

