import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    def forward(self, pred, target):
        pred = torch.sigmoid(pred)

        smooth = 1e-6
        intersection = (pred * target).sum()
        total = pred.sum() + target.sum()

        dice = (2 * intersection + smooth) / (total + smooth)
        return 1 - dice


bce = nn.BCEWithLogitsLoss()
dice = DiceLoss()


def deep_supervision_loss(outputs, target):
    final, aux2, aux3, aux4 = outputs

    loss_final = bce(final, target) + dice(final, target)
    loss_aux2 = bce(aux2, target) + dice(aux2, target)
    loss_aux3 = bce(aux3, target) + dice(aux3, target)
    loss_aux4 = bce(aux4, target) + dice(aux4, target)

    return loss_final + 0.4 * loss_aux2 + 0.3 * loss_aux3 + 0.2 * loss_aux4