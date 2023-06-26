from __future__ import annotations

import numpy as np
import torch
import torchvision.transforms
from torchvision.transforms import transforms
from enum import Enum

from star_analysis.service.statistics_service import StatisticsService
from star_analysis.utils.conversions import relocate_channels


class Augmentations(Enum):
    NONE = None
    NORMALIZE = "normalize"
    ROTATE = "rotate"
    FLIP = "flip"
    ZOOM = "zoom"
    ROTATE_FLIP = "rotate_flip"
    BALANCE_CLASSES = "balance_classes"


class BalanceClasses:
    def __init__(self, labels_start: int):
        self.__labels_start = labels_start
        self.__cls_distribution = StatisticsService(
        ).get_distribution_per_class(calculate=False)

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        image = image.numpy()
        local_dist = image[:, :, self.__labels_start:].sum(axis=(0, 1))

        needed_class = np.argmin(local_dist, axis=-1)
        needed_count = np.abs(local_dist[0] - local_dist[1])

        random_cls_values = torch.normal(
            torch.as_tensor(self.__cls_distribution[0][needed_class]).expand(
                (int(needed_count), len(self.__cls_distribution[0][needed_class]))),
            torch.as_tensor(self.__cls_distribution[1][needed_class]).expand(
                (int(needed_count), len(self.__cls_distribution[0][needed_class])))
        )

        x, y = ((~image[:, :, self.__labels_start:].astype(bool)).all(
            axis=-1)).nonzero()
        coords = np.concatenate(
            (x[:, None], y[:, None]), axis=-1)

        pixels_to_choose = len(coords)
        if needed_count > pixels_to_choose:
            pixels_to_choose = needed_count

        chosen_ids = np.random.choice(
            range(pixels_to_choose), size=int(needed_count), replace=False)
        chosen_pixels = coords[chosen_ids]

        image[chosen_pixels[:, 0], chosen_pixels[:, 1],
        :self.__labels_start] = random_cls_values

        image[chosen_pixels[:, 0], chosen_pixels[:, 1],
        self.__labels_start + needed_class] = 1

        image = torch.from_numpy(image)

        return image

    def __repr__(self):
        return f"BalanceClasses({self.__labels_start})"


class PreparePatch:

    def __call__(self, image: torch.Tensor):
        return relocate_channels(image)


def get_transforms(augmentations: Augmentations) -> transforms.Compose | None:
    """
    Get the transforms for input and target images.

    :param augmentations:
    :return: tuple of transforms for input and target images
    """
    transf = None
    if augmentations == Augmentations.NONE:
        return None
    elif augmentations == Augmentations.NORMALIZE:
        transf = [
            torchvision.transforms.Normalize(
                mean=[0.01829018, 0.06763462, 0.03478437, 0.00994593, 0.09194765],
                std=[0.6351227, 1.1362617, 0.8386613, 0.7339489, 3.5971174]
            )
        ]
    elif augmentations == Augmentations.ROTATE:
        transf = [
            PreparePatch(),
            transforms.RandomRotation(degrees=45),
            PreparePatch()
        ]
    elif augmentations == Augmentations.FLIP:
        transf = [
            PreparePatch(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            PreparePatch()
        ]
    elif augmentations == Augmentations.ZOOM:
        raise NotImplementedError("Zoom not implemented")
    elif augmentations == Augmentations.ROTATE_FLIP:
        transf = [
            PreparePatch(),
            transforms.RandomRotation(degrees=45),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            PreparePatch()
        ]
    elif augmentations == Augmentations.BALANCE_CLASSES:
        transf = [
            BalanceClasses(5)
        ]
    else:
        raise ValueError(f"Unknown augmentation: {augmentations}")

    return transforms.Compose(transf)
