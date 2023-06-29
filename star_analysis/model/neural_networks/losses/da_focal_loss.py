from functools import partial
from typing import Optional, List

import torch.nn.functional as F
import torch
from segmentation_models_pytorch.losses import MULTICLASS_MODE, BINARY_MODE, MULTILABEL_MODE, FocalLoss
from torch.nn.modules.loss import _Loss

from star_analysis.model.neural_networks.losses.utils import compute_distance_map
from star_analysis.utils.conversions import vectorize_image, restore_image


def da_focal_loss_with_logits(
        output: torch.Tensor,
        target: torch.Tensor,
        batch_size: int,
        num_classes: int,
        image_shape: tuple[int, int],
        gamma: float = 2.0,
        alpha: Optional[float] = 0.25,
        reduction: str = "mean",
        normalized: bool = False,
        reduced_threshold: Optional[float] = None,
        eps: float = 1e-6,
) -> torch.Tensor:
    target = target.to(dtype=output.dtype, device=output.device)

    logpt = F.binary_cross_entropy_with_logits(output, target, reduction="none")
    pt = torch.exp(-logpt)

    d = vectorize_image(
        compute_distance_map(
            restore_image(target, batch_size, num_classes, image_shape[0], image_shape[1])
        ),
        num_classes
    )
    d = d.view(-1)

    # compute the loss
    if reduced_threshold is None:
        focal_term = (1 + d) * (1.0 - pt).pow(gamma)
    else:
        focal_term = ((1.0 - pt) / reduced_threshold).pow(gamma)
        focal_term[pt < reduced_threshold] = 1

    loss = focal_term * logpt

    if alpha is not None:
        loss *= alpha * target + (1 - alpha) * (1 - target)

    if normalized:
        norm_factor = focal_term.sum().clamp_min(eps)
        loss /= norm_factor

    if reduction == "mean":
        loss = loss.mean()
    if reduction == "sum":
        loss = loss.sum()
    if reduction == "batchwise_mean":
        loss = loss.sum(0)

    return loss


class DAFocalLoss(_Loss):
    def __init__(
            self,
            mode: str,
            num_classes: int,
            image_shape: tuple[int, int],
            alpha: Optional[float] = None,
            gamma: Optional[float] = 2.0,
            ignore_index: Optional[int] = None,
            reduction: Optional[str] = "mean",
            normalized: bool = False,
            reduced_threshold: Optional[float] = None,
    ):
        """
        Default Focal loss for segmentation task with additional distance penalty.

        Shape
             - **y_pred** - torch.Tensor of shape (N, C, H, W)
             - **y_true** - torch.Tensor of shape (N, C, H, W)

        Reference
            https://github.com/BloodAxe/pytorch-toolbelt

        """
        assert mode in {BINARY_MODE, MULTILABEL_MODE, MULTICLASS_MODE}
        super().__init__()

        self.mode = mode
        self.ignore_index = ignore_index
        self.num_classes = num_classes
        self.image_shape = image_shape

        self.da_focal_loss_fn = partial(
            da_focal_loss_with_logits,
            num_classes=num_classes,
            image_shape=image_shape,
            alpha=alpha,
            gamma=gamma,
            reduced_threshold=reduced_threshold,
            reduction=reduction,
            normalized=normalized,
        )

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        assert y_pred.dim() == 4
        assert y_true.dim() == 4
        assert y_pred.shape[0] == y_true.shape[0]
        assert y_pred.shape[1] == self.num_classes
        assert y_pred.shape[2] == y_true.shape[2]
        assert y_pred.shape[3] == y_true.shape[3]

        batch_size = y_true.shape[0]

        y_true = vectorize_image(y_true, self.num_classes)
        y_pred = vectorize_image(y_pred, self.num_classes)

        if self.mode in {BINARY_MODE, MULTILABEL_MODE}:
            y_true = y_true.view(-1)
            y_pred = y_pred.view(-1)

            if self.ignore_index is not None:
                # Filter predictions with ignore label from loss computation
                not_ignored = y_true != self.ignore_index
                y_pred = y_pred[not_ignored]
                y_true = y_true[not_ignored]

            loss = self.da_focal_loss_fn(y_pred, y_true, batch_size=batch_size)

        elif self.mode == MULTICLASS_MODE:
            num_classes = y_pred.size(1)
            loss = 0

            # Filter anchors with -1 label from loss computation
            if self.ignore_index is not None:
                not_ignored = y_true != self.ignore_index

            for cls in range(num_classes):
                cls_y_true = (y_true == cls).long()
                cls_y_pred = y_pred[:, cls, ...]

                if self.ignore_index is not None:
                    cls_y_true = cls_y_true[not_ignored]
                    cls_y_pred = cls_y_pred[not_ignored]

                loss += self.da_focal_loss_fn(cls_y_pred, cls_y_true)
        return loss
