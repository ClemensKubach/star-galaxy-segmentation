import torch


def vectorize_image(image: torch.Tensor, num_classes) -> torch.Tensor:
    """Converts an image of shape (N, C, H|W, H|W) to (bs, num_classes, W*H)"""
    return image.contiguous().view(image.size(0), num_classes, -1)


def restore_image(vec_img: torch.Tensor, bs: int, channels: int, w: int, h: int):
    """Converts an image of shape (N, C, W*H) to (bs, C, H, W)"""
    return vec_img.contiguous().view(bs, channels, h, w)


def relocate_channels(image: torch.Tensor) -> torch.Tensor:
    dims = len(image.shape)
    if dims < 3:
        raise ValueError(f"Image should have at least 3 dimensions, got {dims}")

    new_order = list(range(dims-3)) + list(range(dims-1, dims-3-1, -1))
    return image.permute(*new_order)
