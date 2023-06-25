import torch
from segmentation_models_pytorch.losses import MULTICLASS_MODE
from torch import nn
import torch.nn.functional as F

from star_analysis.model.neural_networks.losses.utils import aggregate_loss


class MseLoss(nn.Module):

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
            raise NotImplementedError("Only supports from_logits=True")

        loss = torch.mean(torch.pow(y_pred - y_true, 2), dim=(0, 2, 3))

        return aggregate_loss(loss)
