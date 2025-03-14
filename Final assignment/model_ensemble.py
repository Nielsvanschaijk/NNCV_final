import torch
import torch.nn as nn
import torch.nn.functional as F

# Import your submodel constructors and any remap functions if needed
from models.Model_small import get_model as get_small_model
from models.Model_medium import get_model as get_medium_model
from models.Model_big import get_model as get_big_model

# If you want to replicate the final compose logic (optional):
from utils import compose_predictions

class EnsembleModel(nn.Module):
    """
    A single PyTorch model that contains the small, medium, and big submodels.
    During forward, it runs all three and (optionally) composes predictions.
    """
    def __init__(self):
        super().__init__()
        
        # Instantiate each of the three submodels.
        self.model_small = get_small_model(device="cpu")   # or device="cuda" if needed
        self.model_medium = get_medium_model(device="cpu")
        self.model_big = get_big_model(device="cpu")

    def forward(self, x):
        """
        Forward pass: returns either
         - separate logits (for each submodel), or
         - a composed prediction (via argmax or your custom logic).
        Adjust this as you like.
        """

        # 1) Forward all submodels
        out_small = self.model_small(x)    # [B, C_small, H_small, W_small]
        out_medium = self.model_medium(x)  # [B, C_medium, H_medium, W_medium]
        out_big = self.model_big(x)        # [B, C_big, H_big, W_big]

        # 2) (Optional) Upsample them so they match the original image size
        #    Replace (H, W) below with x.shape[-2:] or your desired final size
        H, W = x.shape[-2], x.shape[-1]
        out_small_upsampled = F.interpolate(out_small, size=(H, W), mode='bilinear', align_corners=True)
        out_medium_upsampled = F.interpolate(out_medium, size=(H, W), mode='bilinear', align_corners=True)
        out_big_upsampled = F.interpolate(out_big, size=(H, W), mode='bilinear', align_corners=True)

        # 3) (Optional) If you want the "composed" final segmentation using
        #    your `compose_predictions` logic, you can do something like:
        #
        #    pred_small = out_small_upsampled.softmax(dim=1).argmax(dim=1)
        #    pred_medium = out_medium_upsampled.softmax(dim=1).argmax(dim=1)
        #    pred_big = out_big_upsampled.softmax(dim=1).argmax(dim=1)
        #
        #    composed_seg = compose_predictions(pred_small, pred_medium, pred_big, ignore_index=255)
        #
        #    return composed_seg
        #
        # Otherwise, you could return all three sets of logits so your eval
        # code can do whatever it wants with them:
        return out_small_upsampled, out_medium_upsampled, out_big_upsampled
