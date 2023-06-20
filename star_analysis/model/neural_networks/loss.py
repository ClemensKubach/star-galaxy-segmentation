import torch.nn.functional as F
import torch
import torch.nn as nn


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
