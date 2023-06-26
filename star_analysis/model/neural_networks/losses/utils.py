import numpy as np
import torch
from distancemap import distance_map_from_binary_matrix


def aggregate_loss(loss: torch.Tensor) -> torch.Tensor:
    return loss.mean()


def _normalize(d: torch.Tensor, max_possible_distance: int, normed: bool = False):
    scaled = d / max_possible_distance
    if normed:
        if scaled.max() > 0:
            return scaled / scaled.max()
    return scaled


def compute_distance_map(y_true: torch.Tensor, normed: bool = False) -> torch.Tensor:
    """Takes a binary matrix and returns the distance map with values normalized to [0, 1].

    :param y_true as Tensor of shape (N, C, W, H), (C, W, H),  (W, H)
    :param normed as bool. True applies min max scaling between 0 and 1
    """
    dims = len(y_true.shape)
    assert 2 < dims <= 4

    y_true_np = y_true.cpu().numpy()
    distance_map = np.empty_like(y_true_np, dtype=np.float16)
    max_distance = max(y_true_np.shape[-2:]) - 1

    if len(y_true.shape) == 2:
        d = distance_map_from_binary_matrix(y_true_np)
        distance_map = _normalize(d, max_distance, normed)
    elif len(y_true.shape) == 3:
        for cls in range(y_true_np.shape[0]):
            d = distance_map_from_binary_matrix(y_true_np[cls])
            distance_map[cls] = _normalize(d, max_distance, normed)
    elif len(y_true.shape) == 4:
        for sample in range(y_true_np.shape[0]):
            for cls in range(y_true_np.shape[1]):
                d = distance_map_from_binary_matrix(y_true_np[sample, cls])
                distance_map[sample, cls] = _normalize(d, max_distance, normed)
    else:
        raise ValueError("Not supported shape.")
    return torch.from_numpy(distance_map).to(y_true.device)
