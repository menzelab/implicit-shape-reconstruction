from typing import Optional

import torch


def dice_coeff(probabilities: torch.Tensor, labels: torch.Tensor, threshold: Optional[float] = None,
               reduction: Optional[str] = 'mean') -> torch.Tensor:
    """Compute a mean hard or soft dice coefficient between a batch of probabilities and target
    labels. Reduction happens over the batch dimension; if None, return dice per example.
    """
    # This factor prevents division by 0 if both prediction and GT don't have any foreground voxels
    smooth = 1e-3

    if threshold is not None:
        probabilities = probabilities.gt(threshold).float()
    # Flatten all dims except for the batch
    probabilities_flat = torch.flatten(probabilities, start_dim=1)
    labels_flat = torch.flatten(labels, start_dim=1)

    intersection = (probabilities_flat * labels_flat).sum(dim=1)
    volume_sum = probabilities_flat.sum(dim=1) + labels_flat.sum(dim=1)  # it's not the union!
    dice = (2. * intersection + smooth) / (volume_sum + smooth)
    if reduction == 'mean':
        dice = torch.mean(dice)
    elif reduction == 'sum':
        dice = torch.sum(dice)

    return dice


class DiceLoss(torch.nn.Module):
    """Takes logits as input."""
    def __init__(self, threshold: Optional[float] = None, reduction: Optional[str] = 'mean',
                 do_report_metric: bool = False):
        """If no threshold is given, soft dice is computed, otherwise the predicted values are
        thresholded. Reduction happens over the batch dimension; if None, return dice per example.
        If do_report_metric, report the dice score instead of the dice loss (1 - dice score).
        """
        super().__init__()

        if not do_report_metric and threshold is not None:
            raise ValueError('Dice metric should not use thresholding when used as a loss.')

        self.threshold = threshold
        self.reduction = reduction
        self.do_report_metric = do_report_metric

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        probabilities = torch.sigmoid(logits)
        if self.do_report_metric:
            return dice_coeff(probabilities, target, self.threshold, self.reduction)

        return 1.0 - dice_coeff(probabilities, target, self.threshold, self.reduction)


class BCEWithDiceLoss(torch.nn.Module):
    """Weighted sum of Dice loss with binary cross-entropy."""
    def __init__(self, reduction: str, bce_weight: float = 1.0):
        super().__init__()
        self.dice = DiceLoss(None, reduction, False)
        self.bce = torch.nn.BCEWithLogitsLoss(reduction=reduction)
        self.bce_weight = bce_weight

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.bce_weight * self.bce(logits, target) + self.dice(logits, target)
