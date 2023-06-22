from functools import partial

import numpy as np
import torch
from scipy import ndimage
from torch.nn.modules.loss import _Loss


def _neg_loss(
        pred: torch.Tensor,
        gt: torch.Tensor,
        reduction='none',
        normalized: bool = False,
):
    ''' Modified focal loss. Exactly the same as CornerNet.
        Runs faster and costs a little bit more memory
      Arguments:
        pred (batch x c x h x w)
        gt_regr (batch x c x h x w)
    '''
    pos_inds = gt.eq(1).float()
    neg_inds = gt.lt(1).float()

    neg_weights = torch.pow(1 - gt, 4)

    loss = 0

    pred[pred >= 1] = 1 - 1e-7
    gt[gt >= 1] = 1 - 1e-7

    pred[pred <= 0] = 1e-10
    gt[gt <= 0] = 1e-10

    # pred = torch.exp(torch.log(pred))

    pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds  # here can occure NaN
    neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds  # here can occure NaN

    num_pos = pos_inds.float().sum()

    if reduction == 'sum':
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()
    elif reduction == 'none':
        num_pos = 1
    if num_pos == 0:
        loss = loss - neg_loss
    else:
        loss = loss - (pos_loss + neg_loss) / num_pos

    if normalized:
        loss = loss / pred.shape[0]

    return loss


def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def draw_umich_gaussian(heatmap, center, radius, k=1):
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)

    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:  # TODO debug
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap


def np_pseudo_nms_max_filter(heat, kernel=3):
    pad = (kernel - 1) // 2

    hmax_ = ndimage.maximum_filter(heat, size=None, footprint=np.ones((kernel, kernel)), output=None,
                                   mode='reflect', cval=0.0, origin=0)
    keep = (hmax_ == heat).astype(float)
    return heat * keep


class CornerNetLoss(_Loss):

    def __init__(
            self,
            mode: str
    ):
        super(CornerNetLoss, self).__init__()
        self.mode = mode
        self.loss_fn = partial(
            _neg_loss,
            reduction="sum"
        )

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        y_true = draw_umich_gaussian(
            y_true,
            (y_true.shape[0] // 2, y_true.shape[1] // 2),
            2
        )

        loss = self.loss_fn(y_pred, y_true)
        return loss
