from enum import StrEnum

from torchvision.transforms import transforms


class Augmentations(StrEnum):
    NONE = None
    ROTATE = "rotate"
    FLIP = "flip"
    ZOOM = "zoom"
    ROTATE_FLIP = "rotate_flip"


def get_transforms(augmentations: Augmentations) -> tuple[transforms.Compose | None, transforms.Compose | None]:
    """
    Get the transforms for input and target images.

    :param augmentations:
    :return: tuple of transforms for input and target images
    """
    if augmentations == Augmentations.NONE:
        return None, None
    elif augmentations == Augmentations.ROTATE:
        t = transforms.Compose([
            transforms.RandomRotation(degrees=30),  # Random rotations up to 30 degrees
            transforms.ToTensor(),
        ])
        return t, t
    elif augmentations == Augmentations.FLIP:
        t = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
        ])
        return t, t
    elif augmentations == Augmentations.ZOOM:
        raise NotImplementedError("Zoom not implemented")
    elif augmentations == Augmentations.ROTATE_FLIP:
        t = transforms.Compose([
            transforms.RandomRotation(degrees=30),  # Random rotations up to 30 degrees
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
        ])
        return t, t
    else:
        raise ValueError(f"Unknown augmentation: {augmentations}")
