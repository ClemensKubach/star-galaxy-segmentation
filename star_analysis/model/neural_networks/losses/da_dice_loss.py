from typing import Optional, List

import torch.nn.functional as F
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.losses import MULTICLASS_MODE
from torch.nn.modules.loss import _Loss

from star_analysis.model.neural_networks.losses.da_mse_loss import DistanceLoss
from star_analysis.utils.conversions import vectorize_image


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
