from dataclasses import dataclass
from datetime import datetime
from typing import Any

from lightning import LightningDataModule, LightningModule, Trainer
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.tuner import Tuner
from torch.nn import Module

from star_analysis.data.augmentations import Augmentations, get_transforms
from star_analysis.data.configs import SdssDatasetConfig, SdssDataModuleConfig
from star_analysis.data.datamodules import SdssDataModule
from star_analysis.dataprovider.sdss_dataprovider import SDSSDataProvider
from star_analysis.model.neural_networks.model_config import ModelConfig
from star_analysis.utils.callbacks import PlottingCallback
from star_analysis.utils.constants import CHECKPOINT_DIR


@dataclass
class TrainerConfig:
    logger: Any | None
    limit_train_batches: float | int | None = None
    limit_val_batches: float | int | None = None
    max_epochs: int = 10
    devices: int | str = "auto"
    log_every_n_steps: int = 50


@dataclass
class OptunaTuneTrainerConfig(TrainerConfig):
    num_trials: int = None
    timeout: int = 60 * 60 * 2
    num_jobs: int = -1
    show_progress_bar: bool = True
    gc_after_trial: bool = True


@dataclass
class RunConfig:
    model_config: ModelConfig | None = None
    augmentation: Augmentations = Augmentations.NONE
    shuffle_train: bool = True
    train_size: float = 0.8
    patch_size: int = 224
    trainer: Trainer | None = None
    use_mmap: bool = True


class Run:
    def __init__(
            self,
            config: RunConfig,
            name: str | None = None,
            datamodule: LightningDataModule | None = None,
    ):
        self.__name = name
        self.__config = config
        self.__built = False
        self.__trained = False

        self.__data_module: LightningDataModule | None = datamodule
        self.__trainer: Trainer | None = config.trainer
        self.__tuner: Tuner | None = None
        self.__model: LightningModule | None = config.model_config.model_module
        self.__loss: Module | None = config.model_config.loss_module

    @property
    def name(self) -> str:
        return self.__name

    @property
    def config(self) -> RunConfig:
        return self.__config

    @property
    def data_module(self) -> LightningDataModule | None:
        return self.__data_module

    @property
    def trainer(self) -> Trainer | None:
        return self.__trainer

    @property
    def model(self) -> LightningModule | None:
        return self.__model

    @property
    def loss(self) -> Module | None:
        return self.__loss

    @property
    def built(self) -> bool:
        return self.__built

    @property
    def trained(self) -> bool:
        return self.__trained

    def build(self, data_dir: str, num_workers: int, trainer_config: TrainerConfig):
        self._build_pipeline(
            data_dir=data_dir, num_workers=num_workers, trainer_config=trainer_config
        )

    def prebuild(self, data_dir: str, num_workers: int):
        self._build_pipeline(
            prebuild=True,
            data_dir=data_dir, num_workers=num_workers
        )

    def fit(self):
        self.trainer.fit(
            model=self.model,
            datamodule=self.data_module
        )
        self.__trained = True

    def rebuild(
            self,
            rebuild_data_module: bool = False,
            rebuild_trainer: bool = True,
            rebuild_model: bool = True,
            rebuild_loss: bool = False,
            data_dir: str | None = None, num_workers: int | None = None, trainer_config: TrainerConfig | None = None
    ):
        self.__built = False
        self._build_pipeline(
            force_build_data_module=rebuild_data_module,
            force_build_trainer=rebuild_trainer,
            force_build_model=rebuild_model,
            force_build_loss=rebuild_loss,
            data_dir=data_dir,
            num_workers=num_workers,
            trainer_config=trainer_config
        )

    def _build_pipeline(
            self,
            prebuild: bool = False,
            force_build_data_module: bool = False,
            force_build_trainer: bool = False,
            force_build_model: bool = False,
            force_build_loss: bool = False,
            data_dir: str | None = None, num_workers: int | None = None, trainer_config: TrainerConfig | None = None
    ):
        if self.built:
            print(f"Run {self.name} already built. Try to rebuild.")
            return

        if self.config.model_config is None:
            raise ValueError("Model config is None")
        else:
            if self.name is None:
                self.__name = f'run-{self.config.model_config.model_type}-{str(datetime.now())}'
            if self.loss is None or force_build_loss:
                self.__loss = self._build_loss()
            if self.model is None or force_build_model:
                self.__model = self._build_model()

        if self.data_module is None or force_build_data_module:
            self.__data_module = self._build_datamodule(data_dir, num_workers)

        if not prebuild:
            if self.trainer is None or force_build_trainer:
                if trainer_config is None:
                    raise ValueError("Trainer config is None")
                self.__trainer = self._build_trainer(trainer_config)

            if self.__tuner is None:
                self.__tuner = self._build_tuner(self.__trainer)

            self.__built = True

    def _build_loss(self) -> Module:
        return self.config.model_config.get_loss()

    def _build_model(self) -> LightningModule:
        return self.config.model_config.get_model(self.loss)

    def _build_datamodule(self, data_dir: str, num_workers: int):
        transform = get_transforms(self.config.augmentation)
        dataset_config = SdssDatasetConfig(
            data_dir=data_dir,
            patch_shape=(self.config.patch_size, self.config.patch_size),
            prepare=False,
            run=SDSSDataProvider.FIXED_VALIDATION_RUN,
            transform=transform,
            use_mmap=self.config.use_mmap
        )
        module_config = SdssDataModuleConfig(
            dataset_config=dataset_config,
            batch_size=self.config.model_config.batch_size,
            shuffle_train=self.config.shuffle_train,
            train_size=self.config.train_size,
            num_workers=num_workers
        )
        return SdssDataModule(module_config)

    def _build_trainer(self, config: TrainerConfig) -> Trainer:
        lr_monitor = LearningRateMonitor(logging_interval='step')
        checkpointing_callback = ModelCheckpoint(
            monitor='val_loss',
            save_top_k=3,
            auto_insert_metric_name=True
        )
        if config.logger is None:
            print("No logger provided!")
        trainer = Trainer(
            max_epochs=config.max_epochs,
            logger=config.logger,
            num_nodes=-1,
            limit_train_batches=config.limit_train_batches,
            limit_val_batches=config.limit_val_batches,
            enable_checkpointing=True,
            callbacks=[
                PlottingCallback(),
                lr_monitor,
                checkpointing_callback
            ],
            default_root_dir=CHECKPOINT_DIR,
            devices=config.devices,
            log_every_n_steps=config.log_every_n_steps,
        )
        return trainer

    def _build_tuner(self, trainer: Trainer):
        tuner = Tuner(trainer)
        if self.config.model_config.batch_size is None:
            tuner.scale_batch_size(
                model=self.model,
                mode="power",
                datamodule=self.data_module
            )
        if self.config.model_config.learning_rate is None:
            tuner.lr_find(
                model=self.model,
                datamodule=self.data_module
            )
        return tuner
