import numpy as np
import torch
from distancemap import distance_map_from_binary_matrix


def aggregate_loss(loss: torch.Tensor) -> torch.Tensor:
    return loss.mean()


def compute_distance_map(y_true: torch.Tensor) -> torch.Tensor:
    """Takes a binary matrix and returns the distance map with values normalized to [0, 1]."""
    y_true_np = y_true.cpu().numpy()
    distance_map = np.empty_like(y_true_np)
    max_distance = max(y_true_np.shape[2:]) - 1

    for sample in range(y_true_np.shape[0]):
        for channel in range(y_true_np.shape[1]):
            distance_map[sample, channel] = distance_map_from_binary_matrix(y_true_np[sample, channel])
    normalized_distance_map = torch.from_numpy(distance_map / max_distance)
    return normalized_distance_map.to(y_true.device)
