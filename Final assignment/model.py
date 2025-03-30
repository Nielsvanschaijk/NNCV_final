# model.py
import torch
import torch.nn as nn

# Make sure these are the same references used in train.py:
from models.Model_small import get_model as get_small_model
from models.Model_medium import get_model as get_medium_model
from models.Model_big import get_model as get_big_model

import torch
from torchvision.datasets import Cityscapes
from torchvision.utils import make_grid

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


class Model(nn.Module):
    """
    A single PyTorch model that contains the small, medium, and big submodels.
    It returns a single final segmentation map of shape [B, H, W],
    where each pixel is a training ID.

    If you need out-of-distribution detection:
      - Add an extra head and return (segmentation, classification).
    """

    def __init__(self):
        super().__init__()
        # Instantiate each submodel on CPU by default
        self.model_small = get_small_model(device="cpu")    # 8 classes (0..7)
        self.model_medium = get_medium_model(device="cpu")  # 5 classes (0..4)
        self.model_big = get_big_model(device="cpu")        # 9 classes (0..8)

    def forward(self, x):
        """
        Inputs:
            x: a Tensor of shape [B, 3, H, W] (RGB image).
        Returns:
            A final segmentation map of shape [B, H, W],
            where each pixel is an integer training ID (0..18, or 255 for ignore).
        """
        # Get raw logits from each submodel
        out_small = self.model_small(x)   # [B, 8, H, W]
        out_medium = self.model_medium(x) # [B, 5, H, W]
        out_big = self.model_big(x)       # [B, 9, H, W]

        # Argmax each submodel's logits to get predicted class per pixel
        pred_small = out_small.argmax(dim=1)     # [B, H, W], classes in {0..7}
        pred_medium = out_medium.argmax(dim=1)   # [B, H, W], classes in {0..4}
        pred_big = out_big.argmax(dim=1)         # [B, H, W], classes in {0..8}

        # Compose them into a single segmentation map with Cityscapes train IDs
        # The arguments bg_small=0, bg_medium=0, bg_big=0 indicate the "background" ID in each submodel
        composed_pred = compose_predictions(
            pred_small,
            pred_medium,
            pred_big,
            bg_small=0, 
            bg_medium=0, 
            bg_big=0
        )
        # shape: [B, H, W], values typically in {0..18, 255 for ignore}

        return composed_pred
    
