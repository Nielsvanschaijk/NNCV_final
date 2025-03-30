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
    img_resized = TF.resize(img, (512, 512))  # <-- Adjust if trained on another size
    
    # 2) Convert to a torch.FloatTensor (range [0,1])
    x = TF.to_tensor(img_resized)  # shape [3, H, W]
    
    # 3) Normalize channels (same mean/std as your training)
    x = TF.normalize(x, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    
    # 4) Add batch dimension: [1, 3, H, W]
    x = x.unsqueeze(0)
    
    return x

def postprocess(pred_logits: torch.Tensor, original_shape: tuple[int, int]) -> np.ndarray:
    """
    Example postprocess function if `pred_logits` has shape [n_classes, H, W].
    Returns an np.array of shape [original_shape[0], original_shape[1], n_classes].
    """

    # 1) Convert to float for F.interpolate()
    pred_logits_float = pred_logits.float()

    # 2) Add a batch dimension if necessary
    #    (assuming pred_logits is [n_classes, H, W], so shape becomes [1, n_classes, H, W])
    if pred_logits_float.dim() == 3: 
        pred_logits_float = pred_logits_float.unsqueeze(0)

    # 3) Upsample to original_shape
    upsampled = F.interpolate(pred_logits_float, size=original_shape, mode='nearest')

    # 4) Remove batch dimension if we added it: [1, n_classes, H, W] -> [n_classes, H, W]
    upsampled = upsampled.squeeze(0)

    # 5) (Optional) permute if you want [H, W, n_classes] instead of [n_classes, H, W]
    upsampled = upsampled.permute(1, 2, 0)  # [H, W, n_classes]

    # 6) If you need integer labels, cast to .long():
    # upsampled = upsampled.long()  # only if you're done with floating ops

    # 7) Move to CPU and convert to numpy
    upsampled_np = upsampled.cpu().numpy()

    return upsampled_np