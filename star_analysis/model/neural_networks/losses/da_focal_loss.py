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
    """Compute binary focal loss between target and output logits.
    See :class:`~pytorch_toolbelt.losses.FocalLoss` for details.

    Args:
        output: Tensor of arbitrary shape (predictions of the model)
        target: Tensor of the same shape as input
        gamma: Focal loss power factor
        alpha: Weight factor to balance positive and negative samples. Alpha must be in [0...1] range,
            high values will give more weight to positive class.
        reduction (string, optional): Specifies the reduction to apply to the output:
            'none' | 'mean' | 'sum' | 'batchwise_mean'. 'none': no reduction will be applied,
            'mean': the sum of the output will be divided by the number of
            elements in the output, 'sum': the output will be summed. Note: :attr:`size_average`
            and :attr:`reduce` are in the process of being deprecated, and in the meantime,
            specifying either of those two args will override :attr:`reduction`.
            'batchwise_mean' computes mean loss per sample in batch. Default: 'mean'
        normalized (bool): Compute normalized focal loss (https://arxiv.org/pdf/1909.07829.pdf).
        reduced_threshold (float, optional): Compute reduced focal loss (https://arxiv.org/abs/1903.01347).

    References:
        https://github.com/open-mmlab/mmdetection/blob/master/mmdet/core/loss/losses.py
    """
    # target = target.type(output.type())
    # target = target.to(dtype=output.dtype, device=output.device)

    logpt = F.binary_cross_entropy_with_logits(output, target, reduction="none")
    # logpt2 = F.binary_cross_entropy_with_logits(1-output, target, reduction="none")
    pt = torch.exp(-logpt)

    d = vectorize_image(
        compute_distance_map(
            restore_image(target, batch_size, num_classes, image_shape[0], image_shape[1])
        ),
        num_classes
    )
    y = 1 - d
    y = y.view(-1)
    d = d.view(-1)

    # compute the loss
    if reduced_threshold is None:
        focal_term = (1 + d) * (1.0 - pt).pow(gamma)
        # focal_term2 = (1.0 - y) * y.pow(gamma)
    else:
        focal_term = ((1.0 - pt) / reduced_threshold).pow(gamma)
        focal_term[pt < reduced_threshold] = 1

    loss = focal_term * logpt
    # loss1 = focal_term * logpt
    # loss2 = focal_term2 * logpt2
    # loss = loss2
    # loss[y == 1] = loss1[y == 1]

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
            batch_size: int,
            num_classes: int,
            image_shape: tuple[int, int],
            alpha: Optional[float] = None,
            gamma: Optional[float] = 2.0,
            ignore_index: Optional[int] = None,
            reduction: Optional[str] = "mean",
            normalized: bool = False,
            reduced_threshold: Optional[float] = None,
    ):
        """Compute Focal loss

        Args:
            mode: Loss mode 'binary', 'multiclass' or 'multilabel'
            alpha: Prior probability of having positive value in target.
            gamma: Power factor for dampening weight (focal strength).
            ignore_index: If not None, targets may contain values to be ignored.
                Target values equal to ignore_index will be ignored from loss computation.
            normalized: Compute normalized focal loss (https://arxiv.org/pdf/1909.07829.pdf).
            reduced_threshold: Switch to reduced focal loss. Note, when using this mode you
                should use `reduction="sum"`.

        Shape
             - **y_pred** - torch.Tensor of shape (N, C, H, W)
             - **y_true** - torch.Tensor of shape (N, H, W) or (N, C, H, W)

        Reference
            https://github.com/BloodAxe/pytorch-toolbelt

        """
        assert mode in {BINARY_MODE, MULTILABEL_MODE, MULTICLASS_MODE}
        super().__init__()

        self.mode = mode
        self.ignore_index = ignore_index
        self.focal_loss_fn = partial(
            da_focal_loss_with_logits,
            batch_size=batch_size,
            num_classes=num_classes,
            image_shape=image_shape,
            alpha=alpha,
            gamma=gamma,
            reduced_threshold=reduced_threshold,
            reduction=reduction,
            normalized=normalized,
        )

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:

        if self.mode in {BINARY_MODE, MULTILABEL_MODE}:
            y_true = y_true.view(-1)
            y_pred = y_pred.view(-1)

            if self.ignore_index is not None:
                # Filter predictions with ignore label from loss computation
                not_ignored = y_true != self.ignore_index
                y_pred = y_pred[not_ignored]
                y_true = y_true[not_ignored]

            loss = self.focal_loss_fn(y_pred, y_true)

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

                loss += self.focal_loss_fn(cls_y_pred, cls_y_true)
        return loss
