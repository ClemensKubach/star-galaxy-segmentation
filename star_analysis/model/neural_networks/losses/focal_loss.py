from typing import Optional, List

import segmentation_models_pytorch as smp
import torch

from star_analysis.utils.conversions import vectorize_image


class FocalLoss(smp.losses.FocalLoss):
    def __init__(
            self,
            mode: str,
            num_classes: int,
            alpha: float | None = None,
            gamma: float | None = 2.0,
            ignore_index: int | None = None,
            reduction: str | None = "mean",
            normalized: bool = False,
            reduced_threshold: float | None = None
    ):
        super().__init__(
            mode=mode,
            alpha=alpha,
            gamma=gamma,
            ignore_index=ignore_index,
            reduction=reduction,
            normalized=normalized,
            reduced_threshold=reduced_threshold
        )
        self.num_classes = num_classes

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        y_pred = vectorize_image(y_pred, self.num_classes)
        y_true = vectorize_image(y_true, self.num_classes)

        return super().forward(y_pred, y_true)
