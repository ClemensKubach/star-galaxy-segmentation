import os
from dataclasses import dataclass

import torch
from lightning import Trainer, LightningModule

from star_analysis.data.augmentations import Augmentations
from star_analysis.model.neural_networks.losses.types import LossType
from star_analysis.model.neural_networks.model_config import ModelConfig
from star_analysis.model.types import ModelTypes
from star_analysis.runner.run import RunConfig
from star_analysis.runner.runner import Runner
from star_analysis.utils.constants import DATAFILES_ROOT


class SdssRunner(Runner):
    def __init__(
            self,
            data_dir: str = DATAFILES_ROOT,
            project_name: str = "sdss-project",
            num_workers: int = os.cpu_count(),
    ):
        super().__init__(data_dir=data_dir, project_name=project_name, num_workers=num_workers)


@dataclass
class SdssModelConfig(ModelConfig):
    learning_rate: float = 1e-3
    batch_size: int = 32
    image_shape: tuple[int, int] = (224, 224)
    num_classes: int = 2
    model_type = ModelTypes.UNET
    model_module: LightningModule | None = None
    loss_type: LossType = LossType.DICE
    loss_mode: str | None = 'multilabel'
    loss_module: torch.nn.Module | None = None


@dataclass
class SdssRunConfig(RunConfig):
    model_config: ModelConfig | None = SdssModelConfig()
    augmentation: Augmentations = Augmentations.NONE
    shuffle_train: bool = True
    train_size: float = 0.8
    patch_size: int = 224
    trainer: Trainer | None = None
    use_mmap: bool = True
