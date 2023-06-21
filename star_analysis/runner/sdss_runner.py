import os
from typing import Any

from lightning import LightningDataModule, Trainer
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.tuner import Tuner
from torch.utils.data import DataLoader
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler

from star_analysis.data.augmentations import get_transforms, Augmentations
from star_analysis.data.configs import SdssDatasetConfig, SdssDataModuleConfig
from star_analysis.data.datamodules import SdssDataModule
from star_analysis.dataprovider.sdss_dataprovider import SDSSDataProvider
from star_analysis.model.types import ModelTypes
from star_analysis.runner.executable import Executable
from star_analysis.utils.callbacks import PlottingCallback
from star_analysis.utils.constants import CHECKPOINT_DIR, DATAFILES_ROOT


class SdssRunner(Executable):

    def __init__(
            self,
            data_dir: str = DATAFILES_ROOT,
            model_type: ModelTypes = ModelTypes.UNET,
            project_name: str = "sdss-tests",
            batch_size: int = None,
            learning_rate_init: float = None,

            augmentation: Augmentations = Augmentations.NONE,
            shuffle_train: bool = True,
            train_size: float = 0.8,
            workers: int = os.cpu_count(),
            patch_size: int = 224,
            use_mmap: bool = True
    ):
        super().__init__(
            data_dir=data_dir,
            model_type=model_type,
            project_name=project_name,
            batch_size=batch_size,
            learning_rate_init=learning_rate_init
        )

        transform = get_transforms(augmentation)
        dataset_config = SdssDatasetConfig(
            data_dir=self.data_dir,
            patch_shape=(patch_size, patch_size),
            prepare=False,
            run=SDSSDataProvider.FIXED_VALIDATION_RUN,
            transform=transform,
            use_mmap=use_mmap
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
            max_epochs=15,
            limit_train_batches=None,
            limit_val_batches=None,
    ):
        lr_monitor = LearningRateMonitor(logging_interval='step')
        checkpointing_callback = ModelCheckpoint(
            dirpath=CHECKPOINT_DIR,
            monitor='val_loss',
            save_top_k=3
        )
        self.trainer = Trainer(
            max_epochs=max_epochs,
            logger=self.logger,
            num_nodes=-1,
            limit_train_batches=limit_train_batches,
            limit_val_batches=limit_val_batches,
            enable_checkpointing=True,
            callbacks=[
                PlottingCallback(),
                lr_monitor,
                checkpointing_callback
            ],
            default_root_dir=CHECKPOINT_DIR
        )
        tuner = Tuner(self.trainer)
        if self.batch_size is None:
            tuner.scale_batch_size(
                self.model, mode="power", datamodule=self.data_module)
        if self.learning_rate_init is None:
            tuner.lr_find(self.model, datamodule=self.data_module)
        self.trainer.fit(
            model=self.model,
            datamodule=self.data_module
        )

    def tune(
            self,
            max_epochs=10,
    ):
        # Define the hyperparameter search space
        config = {
            'learning_rate': tune.loguniform(1e-4, 1e-2)
        }

        # Define the objective function for hyperparameter tuning
        def objective(config):
            self.trainer = Trainer(
                max_epochs=max_epochs,
                logger=self.logger,
                num_nodes=-1,
                callbacks=[
                    PlottingCallback()
                ],  # [early_stopping]
                default_root_dir=CHECKPOINT_DIR
            )
            tuner = Tuner(self.trainer)
            tuner.scale_batch_size(
                self.model, mode="power", datamodule=self.data_module)
            tuner.lr_find(self.model)
            self.trainer.fit(
                model=self.model,
                datamodule=self.data_module
            )
            return self.trainer.callback_metrics['val_loss']

        # Set up the Ray Tune scheduler
        scheduler = ASHAScheduler(
            max_t=10,
            grace_period=1,
            reduction_factor=2
        )
        reporter = CLIReporter(metric_columns=["val_loss"])
        # Perform hyperparameter tuning with Ray Tune
        analysis = tune.run(
            objective,
            config=config,
            num_samples=10,
            scheduler=scheduler,
            progress_reporter=reporter
        )
        # Retrain the model with the best hyperparameters
        best_config = analysis.get_best_config(metric='val_loss')
        # best_model = MyModel(best_config['input_size'], best_config['hidden_size'], best_config['output_size'])
        # trainer = Trainer(max_epochs=10)
        # trainer.fit(best_model)

    def test(self):
        self.trainer.test(
            ckpt_path="best"
        )

    def predict(self, data_loader: DataLoader):
        self.trainer.predict(
            model=self.model,
            dataloaders=data_loader
        )


if __name__ == '__main__':
    runner = SdssRunner(shuffle_train=False, model_type=ModelTypes.UNET)
    runner.init()
    runner.train(limit_train_batches=10, limit_val_batches=10, max_epochs=10)
