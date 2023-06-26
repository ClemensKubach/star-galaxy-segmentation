from typing import Optional, List

import torch

from star_analysis.model.neural_networks.losses.dice_loss import DiceLoss
from star_analysis.model.neural_networks.losses.utils import compute_distance_map
from star_analysis.utils.conversions import vectorize_image


class DADiceLoss(DiceLoss):
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
        super(DADiceLoss, self).__init__(
            mode=mode,
            num_classes=num_classes,
            classes=classes,
            log_loss=log_loss,
            from_logits=from_logits,
            smooth=smooth,
            ignore_index=ignore_index,
            eps=eps
        )

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        d = vectorize_image(
            compute_distance_map(y_true),
            self.num_classes
        )

        y_pred = vectorize_image(y_pred, self.num_classes)
        y_true = vectorize_image(y_true, self.num_classes)

        dice_loss = super().forward(y_pred, y_true)
        return (1 + d) * dice_loss
