from dataclasses import dataclass
from typing import Any

from star_analysis.dataprovider.sdss_dataprovider import SDSSDataProvider
from star_analysis.utils.constants import DATAFILES_ROOT


@dataclass
class SdssDatasetConfig:
    data_dir: str = DATAFILES_ROOT
    patch_shape: tuple[int, int] | None = (32, 32)
    prepare: bool = False
    run: str = SDSSDataProvider.FIXED_VALIDATION_RUN
    transform: Any = None
    target_transform: Any = None


@dataclass
class SdssDataModuleConfig:
    dataset_config: SdssDatasetConfig
    batch_size = 32
    shuffle_train = True
    train_size = 0.8
    val_size = 0.1
    num_workers = 1
