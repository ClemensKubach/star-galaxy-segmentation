from typing import Optional, List

import segmentation_models_pytorch as smp
import torch

from star_analysis.utils.conversions import vectorize_image


class DiceLoss(smp.losses.DiceLoss):
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
        super().__init__(
            mode=mode,
            classes=classes,
            log_loss=log_loss,
            from_logits=from_logits,
            smooth=smooth,
            ignore_index=ignore_index,
            eps=eps
        )
        self.num_classes = num_classes

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        y_pred = vectorize_image(y_pred, self.num_classes)
        y_true = vectorize_image(y_true, self.num_classes)

        return super().forward(y_pred, y_true)
