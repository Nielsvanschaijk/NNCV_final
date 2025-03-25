# model.py

import torch
import torch.nn as nn

# Make sure these are the same references used in train.py:
from models.Model_small import get_model as get_small_model
from models.Model_medium import get_model as get_medium_model
from models.Model_big import get_model as get_big_model

# If compose_predictions is in your utils:
from utils import compose_predictions


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

        # If you needed OOD detection, you might do something like:
        #   classification = some_network(x)  # shape [B, 1] or [B, 2]
        #   return composed_pred, classification
