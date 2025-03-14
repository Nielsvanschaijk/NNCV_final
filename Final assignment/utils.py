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

def compose_predictions(pred_small, pred_medium, pred_big, ignore_index=255):
    """
    Combine the three predictions so that:
      small > medium > big > background.
    Each prediction has shape (B, H, W). Returns a composed prediction (B, H, W).
    """
    final = torch.full_like(pred_small, fill_value=ignore_index)
    mask_small = (pred_small != ignore_index)
    final[mask_small] = pred_small[mask_small]
    mask_medium = (final == ignore_index) & (pred_medium != ignore_index)
    final[mask_medium] = pred_medium[mask_medium]
    mask_big = (final == ignore_index) & (pred_big != ignore_index)
    final[mask_big] = pred_big[mask_big]
    return final
