import torch
import torch.nn.functional as F

def multiclass_dice_loss_small(pred, target, smooth=1):
    """
    Computes Dice Loss for the small model segmentation.
    Assumes pred shape: [B,8,H,W] and target is one-hot encoded with 8 channels.
    Ignores background (channel 0) and computes loss only for channels 1 to 7.
    
    Args:
        pred: Tensor of logits, shape [B,8,H,W].
        target: One-hot encoded ground truth, shape [B,8,H,W].
        smooth: Smoothing factor.
    Returns:
        Scalar Dice Loss.
    """
    pred = F.softmax(pred, dim=1)
    num_classes = pred.shape[1]
    dice = 0.0
    count = 0
    # Loop over foreground channels (1 to 7)
    for c in range(1, num_classes):
        pred_c = pred[:, c]
        target_c = target[:, c]
        intersection = (pred_c * target_c).sum(dim=(1, 2))
        union = pred_c.sum(dim=(1, 2)) + target_c.sum(dim=(1, 2))
        dice += (2. * intersection + smooth) / (union + smooth)
        count += 1
    return 1 - dice.mean() / count

def multiclass_dice_loss_medium(pred, target, smooth=1):
    """
    Computes Dice Loss for the medium model segmentation.
    Assumes pred shape: [B,5,H,W] and target is one-hot encoded with 5 channels.
    Ignores background (channel 0) and computes loss only for channels 1 to 4.
    
    Args:
        pred: Tensor of logits, shape [B,5,H,W].
        target: One-hot encoded ground truth, shape [B,5,H,W].
        smooth: Smoothing factor.
    Returns:
        Scalar Dice Loss.
    """
    pred = F.softmax(pred, dim=1)
    num_classes = pred.shape[1]
    dice = 0.0
    count = 0
    # Loop over foreground channels (1 to 4)
    for c in range(1, num_classes):
        pred_c = pred[:, c]
        target_c = target[:, c]
        intersection = (pred_c * target_c).sum(dim=(1, 2))
        union = pred_c.sum(dim=(1, 2)) + target_c.sum(dim=(1, 2))
        dice += (2. * intersection + smooth) / (union + smooth)
        count += 1
    return 1 - dice.mean() / count

def multiclass_dice_loss_big(pred, target, smooth=1):
    """
    Computes Dice Loss for the big model segmentation.
    Assumes pred shape: [B,9,H,W] and target is one-hot encoded with 9 channels.
    Ignores background (channel 0) and computes loss only for channels 1 to 8.
    
    Args:
        pred: Tensor of logits, shape [B,9,H,W].
        target: One-hot encoded ground truth, shape [B,9,H,W].
        smooth: Smoothing factor.
    Returns:
        Scalar Dice Loss.
    """
    pred = F.softmax(pred, dim=1)
    num_classes = pred.shape[1]
    dice = 0.0
    count = 0
    # Loop over foreground channels (1 to 8)
    for c in range(1, num_classes):
        pred_c = pred[:, c]
        target_c = target[:, c]
        intersection = (pred_c * target_c).sum(dim=(1, 2))
        union = pred_c.sum(dim=(1, 2)) + target_c.sum(dim=(1, 2))
        dice += (2. * intersection + smooth) / (union + smooth)
        count += 1
    return 1 - dice.mean() / count
