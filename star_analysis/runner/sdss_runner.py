from typing import Any

from lightning import LightningDataModule, Trainer

from star_analysis.data.datamodules import SdssDataModule
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

            full_sequences: bool = False,
            use_ground_truth_target: bool = False,
            transform: Any = None,
            target_transform: Any = None,
            shuffle_train: bool = False,
            train_size: float = 0.8,
            val_size: float = 0.1
    ):
        super().__init__(
            data_dir=data_dir,
            model_type=model_type,
            project_name=project_name,
            batch_size=batch_size,
            learning_rate_init=learning_rate_init
        )
        self.full_sequences = full_sequences
        self.use_ground_truth_target = use_ground_truth_target
        self.transform = transform
        self.target_transform = target_transform
        self.shuffle_train = shuffle_train
        self.train_size = train_size
        self.val_size = val_size

    def _setup_data(self) -> LightningDataModule:
        datamodule = SdssDataModule(
            dataset=dataset,
            batch_size=self.batch_size,
            shuffle_train=self.shuffle_train,
            train_size=self.train_size,
            val_size=self.val_size
        )
        return datamodule

    def train(
            self,
            max_epochs=10,
            limit_train_batches=200,
            limit_val_batches=100,
    ):
        trainer = Trainer(
            max_epochs=max_epochs,
            logger=self.logger,
            num_nodes=-1,
            limit_train_batches=limit_train_batches,
            limit_val_batches=limit_val_batches,
            callbacks=None,  # [early_stopping]
            default_root_dir=CHECKPOINT_DIR
        )
        trainer.fit(
            model=self.model,
            datamodule=self.data_module
        )

    def eval(self):
        super().eval()

    def predict(self):
        super().predict()
