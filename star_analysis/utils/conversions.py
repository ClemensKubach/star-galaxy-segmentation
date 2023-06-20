import torch


def vectorize_image(image: torch.Tensor, num_classes) -> torch.Tensor:
    return image.contiguous().view(image.size(0), num_classes, -1)


def relocate_channels(image: torch.Tensor) -> torch.Tensor:
    return image.permute(0, 3, 2, 1)
