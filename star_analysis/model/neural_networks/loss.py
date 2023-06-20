from typing import Optional, List

import numpy as np
import torch.nn.functional as F
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from distancemap import distance_map_from_binary_matrix
from segmentation_models_pytorch.losses import MULTICLASS_MODE, BINARY_MODE, MULTILABEL_MODE
from torch.nn.modules.loss import _Loss

from star_analysis.utils.conversions import vectorize_image


def custom_loss(pred, target, class_pred, class_target):
    # Calculate the pixel-wise distance loss (e.g., Mean Squared Error)
    pixel_loss = F.mse_loss(pred, target)

    # Calculate the classification loss (e.g., Cross-Entropy Loss)
    class_loss = F.cross_entropy(class_pred, class_target)

    total_loss = pixel_loss + class_loss
    return total_loss


def kl_loss(pred, target):
    # Calculate the KL divergence loss
    kl_loss = F.kl_div(F.log_softmax(pred, dim=1), F.softmax(target, dim=1), reduction='batchmean')
    return kl_loss


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


DiceLoss = smp.losses.DiceLoss


class DiceDistanceLoss(_Loss):
    def __init__(
            self,
            mode: str,
            num_classes: int,
            classes: Optional[List[int]] = None,
            log_loss: bool = False,
            from_logits: bool = True,
            smooth: float = 0.0,
            ignore_index: Optional[int] = None,
            eps: float = 1e-7,
    ):
        super(DiceDistanceLoss, self).__init__()
        self.dice_loss = smp.losses.DiceLoss(
            mode=mode,
            classes=classes,
            log_loss=log_loss,
            from_logits=from_logits,
            smooth=smooth,
            ignore_index=ignore_index,
            eps=eps
        )
        self.num_classes = num_classes
        self.distance_loss = DistanceLoss(mode, from_logits=from_logits)

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        y_pred_img = y_pred
        y_true_img = y_true
        y_pred = vectorize_image(y_pred, self.num_classes)
        y_true = vectorize_image(y_true, self.num_classes)

        dice_loss = self.dice_loss(y_pred, y_true)
        distance_loss = self.distance_loss(y_pred_img, y_true_img)

        return (dice_loss + distance_loss) * 0.5


class DistanceLoss(nn.Module):

    def __init__(self, mode: str, from_logits: bool = True):
        """
        Shape
             - **y_pred** - torch.Tensor of shape (N, C, H, W)
             - **y_true** - torch.Tensor of shape (N, C, H, W)
        """
        super().__init__()
        self.mode = mode
        self.from_logits = from_logits

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        assert y_true.size() == y_pred.size()
        assert y_true.dim() == 4

        if self.from_logits:
            # Apply activations to get [0..1] class probabilities
            # Using Log-Exp as this gives more numerically stable result and does not cause vanishing gradient on
            # extreme values 0 and 1
            if self.mode == MULTICLASS_MODE:
                y_pred = y_pred.log_softmax(dim=1).exp()
            else:
                y_pred = F.logsigmoid(y_pred).exp()
        else:
            raise NotImplementedError("DistanceLoss only supports from_logits=True")

        sqe_loss = torch.pow(y_pred - y_true, 2)

        # Distance penalty
        distance_map = compute_distance_map(y_true)
        loss = torch.mean((1 + distance_map) * sqe_loss, dim=(0, 2, 3))

        return self.aggregate_loss(loss)

    def aggregate_loss(self, loss: torch.Tensor) -> torch.Tensor:
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
