from dataclasses import dataclass
from typing import Any

from star_analysis.dataprovider.sdss_dataprovider import SDSSDataProvider
from star_analysis.utils.constants import DATAFILES_ROOT


@dataclass
class SdssDatasetConfig:
    data_dir: str = DATAFILES_ROOT
    patch_shape: tuple[int, int] | None = (224, 224)
    prepare: bool = False
    run: str = SDSSDataProvider.FIXED_VALIDATION_RUN
    include_train_set: bool = True
    include_test_set: bool = False
    transform: Any = None
    target_transform: Any = None


@dataclass
class SdssDataModuleConfig:
    """Config for the SdssDataModule.

    dataset_config (SdssDatasetConfig): Config for the Sdss dataset. include_train_set and include_test_set are
    handled by the SdssDataModule. Thus, they are ignored here.
    """
    dataset_config: SdssDatasetConfig
    batch_size: int = 32
    shuffle_train: bool = True
    train_size: float = 0.8
    num_workers: int = 1
