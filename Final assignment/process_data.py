# process_data.py

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import numpy as np

def preprocess(img):
    """
    Receives a PIL.Image (original size), and transforms it to the
    format your model expects. Returns a 4D PyTorch tensor of shape:
        [1, 3, H, W]
    without moving it to any specific device (CPU/GPU).
    """

    # 1) Resize the image to the same size used in training
    img_resized = TF.resize(img, (512, 1024))  # <-- Adjust if trained on another size
    
    # 2) Convert to a torch.FloatTensor (range [0,1])
    x = TF.to_tensor(img_resized)  # shape [3, H, W]
    
    # 3) Normalize channels (same mean/std as your training)
    x = TF.normalize(x, mean=[0.286, 0.325, 0.283], std=[0.176, 0.180, 0.177])

    # 4) Add batch dimension: [1, 3, H, W]
    x = x.unsqueeze(0)
    
    return x

def postprocess(pred_logits: torch.Tensor, original_shape: tuple[int, int]) -> np.ndarray:
    """
    Takes raw logits from the model and returns predicted class indices [H, W]
    resized to original image resolution.
    """
    # 1. Make sure it's float
    pred_logits = pred_logits.float()

    # 2. Add batch dim if missing
    if pred_logits.dim() == 3:
        pred_logits = pred_logits.unsqueeze(0)

    # 3. Upsample to original image shape
    upsampled = F.interpolate(pred_logits, size=original_shape, mode='bilinear', align_corners=False)

    # 4. Convert logits to predicted class (argmax over channel dim)
    pred_class = upsampled.argmax(dim=1)  # [B, H, W]

    # 5. Remove batch dim and convert to numpy
    pred_class_np = pred_class.squeeze(0).detach().cpu().numpy()  # [H, W]

    return pred_class_np