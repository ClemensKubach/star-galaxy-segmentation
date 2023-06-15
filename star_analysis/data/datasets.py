import os

import numpy as np
import torch.utils.data as data
from torch import Tensor

from star_analysis.data.configs import SdssDatasetConfig
from star_analysis.dataprovider.image_downloader import ImageDownloader
from star_analysis.dataprovider.sdss_dataprovider import SDSSDataProvider

INNER_IMAGE_PADDING = 128


class Sdss(data.Dataset):
    def __init__(
            self,
            config: SdssDatasetConfig
    ):
        self.config = config
        self.patch_shape = config.patch_shape
        self.inner_padding = INNER_IMAGE_PADDING

        self.provider = SDSSDataProvider(
            downloader=ImageDownloader(config.data_dir, max_workers=os.cpu_count()),
            include_train_set=config.include_train_set,
            include_test_set=config.include_test_set,
        )
        if config.prepare:
            self.prepare()

        fixed_test_sample = self.provider.get_provided_validation_set()
        image_width, image_height = fixed_test_sample[0].shape[:2]

        unique_image_width = image_width - 2 * self.inner_padding
        unique_image_height = image_height - 2 * self.inner_padding

        if self.patch_shape is None:
            self.patch_shape = (unique_image_width, unique_image_height)

        self.cropped_image_shape = (
            unique_image_width - unique_image_width % self.patch_shape[0],
            unique_image_height - unique_image_height % self.patch_shape[1]
        )

        self.num_patches_per_field = (self.cropped_image_shape[0] // self.patch_shape[0]) * \
                                     (self.cropped_image_shape[1] // self.patch_shape[1])

    @property
    def num_fields(self) -> int:
        return len(self.provider)

    def prepare(self):
        self.provider.prepare(run=self.config.run)

    def __len__(self) -> int:
        return self.num_fields * self.num_patches_per_field

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        field_idx = idx // self.num_patches_per_field
        patch_idx = idx % self.num_patches_per_field

        field = self.provider[field_idx]
        if field is None:
            raise ValueError(f"Field with index {field_idx} is None.")

        patch_x, patch_y = self._get_patch(field, patch_idx)
        patch_x, patch_y = Tensor(patch_x), Tensor(patch_y)
        if self.config.transform:
            patch_x = self.config.transform(patch_x)
        if self.config.target_transform:
            patch_y = self.config.target_transform(patch_y)
        return patch_x, patch_y

    def _get_patch(self, field: tuple[np.ndarray, np.ndarray], patch_idx: int) -> tuple[np.ndarray, np.ndarray]:
        x, y = self._get_cropped_data(field)
        patch_width_slice = slice(
            patch_idx * self.patch_shape[0],
            (patch_idx + 1) * self.patch_shape[0]
        )
        patch_height_slice = slice(
            patch_idx * self.patch_shape[1],
            (patch_idx + 1) * self.patch_shape[1]
        )
        x_patch, y_patch = x[patch_width_slice, patch_height_slice], y[patch_width_slice, patch_height_slice]
        return x_patch, y_patch

    def _get_cropped_data(self, field: tuple[np.ndarray, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        field_x, field_y = field
        cropping_width_slice = slice(self.inner_padding, self.inner_padding + self.cropped_image_shape[0])
        cropping_height_slice = slice(self.inner_padding, self.inner_padding + self.cropped_image_shape[1])
        copped_x = field_x[cropping_width_slice, cropping_height_slice]
        copped_y = field_y[cropping_width_slice, cropping_height_slice]
        return copped_x, copped_y
