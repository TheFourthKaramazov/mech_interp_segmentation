###########
# Imports #
###########


import numpy as np
import torch
import torch.nn as nn


# fix possible truncation during training
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True



#################################
# Losses and Evaluation Metrics #
#################################

def should_return_low_loss(inputs, targets, threshold_zero_loss):
    """Helper function for loss"""

    return (inputs.sum(dim=(1, 2, 3)) <= threshold_zero_loss) & (targets.sum(dim=(1, 2, 3)) <= threshold_zero_loss)


def compute_low_loss(inputs, targets, threshold_zero_loss):
    """ Helper function for loss"""
    return 1 / 2 ** ((2.75 * threshold_zero_loss - inputs.sum(dim=(1, 2, 3)) - targets.sum(dim=(1, 2, 3))))


class DiceLoss(nn.Module):
    """
    Dice Loss for binary segmentation tasks. Measures the overlap between
    predicted and ground truth masks, emphasizing small structures.
    """

    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        # flatten tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        # calculate Dice score
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)
        return 1 - dice


class ScaledDiceLoss(nn.Module):
    """
    Scaled version of the dice loss


    """

    def __init__(self, smooth=1, threshold_zero_loss=10):
        super(ScaledDiceLoss, self).__init__()
        self.tissueDice = DiceLoss(smooth)
        self.threshold_zero_loss = threshold_zero_loss

    def forward(self, inputs, targets):
        mask_of_low_loss = should_return_low_loss(inputs, targets, self.threshold_zero_loss)

        inputs_low_loss = inputs[mask_of_low_loss]
        targets_low_loss = targets[mask_of_low_loss]

        if inputs_low_loss.numel() > 0:
            low_losses = compute_low_loss(inputs_low_loss, targets_low_loss, self.threshold_zero_loss)
        else:
            low_losses = torch.tensor(0.0, device=inputs.device)

        inputs_normal_loss = inputs[~mask_of_low_loss]
        targets_normal_loss = targets[~mask_of_low_loss]

        if inputs_normal_loss.numel() > 0:
            scaling_factor = torch.log(
                (inputs_normal_loss.sum() + targets_normal_loss.sum()) / inputs_normal_loss.size(0))
            normal_losses = scaling_factor * self.tissueDice(inputs_normal_loss, targets_normal_loss)
        else:
            normal_losses = torch.tensor(0.0, device=inputs.device)

        return (
            ~mask_of_low_loss).float().mean() * normal_losses.mean() + mask_of_low_loss.float().mean() * low_losses.mean()


def calculate_iou_infer(pred, target):
    """
    Calculate Intersection over Union (IoU) between two binary masks.
    Treat cases with zero union as perfect matches.

    Args:
        pred (numpy.ndarray): Predicted binary mask.
        target (numpy.ndarray): Ground truth binary mask.

    Returns:
        float: IoU score.
    """
    pred = (pred > 0.5).astype(np.uint8)
    target = (target > 0.5).astype(np.uint8)
    intersection = np.logical_and(pred, target).sum()
    union = np.logical_or(pred, target).sum()

    if union == 0:  # if union is zero, treat as a perfect match
        return 1.0

    return intersection / (union + 1e-6)  # calculate IoU


def calculate_dice_infer(pred, target):
    """
    Calculate Dice coefficient between two binary masks.
    Treat cases with zero denominator as perfect matches.

    Args:
        pred (numpy.ndarray): Predicted binary mask.
        target (numpy.ndarray): Ground truth binary mask.

    Returns:
        float: Dice coefficient score.
    """
    pred = (pred > 0.5).astype(np.uint8)
    target = (target > 0.5).astype(np.uint8)
    intersection = np.logical_and(pred, target).sum()
    denominator = pred.sum() + target.sum()

    if denominator == 0:  # if denominator is zero, treat as a perfect match
        return 1.0

    return (2 * intersection) / (denominator + 1e-6)  # calculate Dice


