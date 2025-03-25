import torch
from torchvision.datasets import Cityscapes
from torchvision.utils import make_grid

# --- Label Conversion and Visualization Utilities ---

# Convert raw Cityscapes "ID" labels to "train IDs" (0..18 or 255 for ignore).
id_to_trainid = {cls.id: cls.train_id for cls in Cityscapes.classes}
def convert_to_train_id(label_img: torch.Tensor) -> torch.Tensor:
    return label_img.apply_(lambda x: id_to_trainid[x])

# Dictionary to map train IDs -> colors for visualization.
train_id_to_color = {cls.train_id: cls.color for cls in Cityscapes.classes if cls.train_id != 255}
train_id_to_color[255] = (0, 0, 0)  # Black for ignored labels

def convert_train_id_to_color(prediction: torch.Tensor) -> torch.Tensor:
    """
    Convert a prediction mask (B x 1 x H x W) of train IDs into a color image (B x 3 x H x W).
    """
    batch, _, height, width = prediction.shape
    color_image = torch.zeros((batch, 3, height, width), dtype=torch.uint8)
    for train_id, color in train_id_to_color.items():
        mask = prediction[:, 0] == train_id
        for i in range(3):
            color_image[:, i][mask] = color[i]
    return color_image

def compose_predictions(pred_small, pred_medium, pred_big, bg_small=0, bg_medium=0, bg_big=0):
    """
    Overwrite logic: 
    1) If small != bg_small, use small’s class.
    2) Else if medium != bg_medium, use medium’s class.
    3) Else if big != bg_big, use big’s class.
    Otherwise remain background.
    """
    # Start from small’s predictions
    final = pred_small.clone()

    # Where small is background, try medium
    small_bg_mask = (pred_small == bg_small)
    # Overwrite only where medium is non-bg
    final[small_bg_mask & (pred_medium != bg_medium)] = pred_medium[small_bg_mask & (pred_medium != bg_medium)]

    # Where final is still background, try big
    final_bg_mask = (final == bg_small)
    final[final_bg_mask & (pred_big != bg_big)] = pred_big[final_bg_mask & (pred_big != bg_big)]

    return final

import torch

def compose_predictions(pred_small, pred_medium, pred_big, bg_small=0, bg_medium=0, bg_big=0):
    """
    Compose predictions from three submodels by performing majority voting.
    Assumes each prediction is a tensor of shape [B, H, W].
    """
    # Stack predictions along a new dimension: [B, 3, H, W]
    stacked_preds = torch.stack([pred_small, pred_medium, pred_big], dim=1)
    # Use mode to compute the majority vote along the second dimension.
    composed_pred, _ = torch.mode(stacked_preds, dim=1)
    return composed_pred
