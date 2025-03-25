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
    img_resized = TF.resize(img, (16, 16))  # <-- Adjust if trained on another size
    
    # 2) Convert to a torch.FloatTensor (range [0,1])
    x = TF.to_tensor(img_resized)  # shape [3, H, W]
    
    # 3) Normalize channels (same mean/std as your training)
    x = TF.normalize(x, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    
    # 4) Add batch dimension: [1, 3, H, W]
    x = x.unsqueeze(0)
    
    return x


def postprocess(prediction, shape):
    """
    Receives:
      - prediction: a PyTorch tensor of shape [1, num_classes, H, W]
                    (direct output from the model's forward pass).
      - shape: the (height, width) of the original image.

    1) Upsample back to the original (height, width).
    2) Convert to a single-channel of class IDs with shape [height, width, 1].
    3) Return as a NumPy array (uint8 or int).

    Cityscapes typically has classes {0..18} plus 255 as ignore label.
    """
    original_height, original_width = shape

    # 1) Upsample / resize model output to original image size
    upsampled = F.interpolate(
        prediction,
        size=(original_height, original_width),
        mode='bilinear',
        align_corners=False
    )  # shape: [1, num_classes, original_height, original_width]

    # 2) Get the predicted class per pixel via argmax
    labels = upsampled.argmax(dim=1)  # shape: [1, original_height, original_width]

    # 3) Convert to NumPy, shape [original_height, original_width]
    labels = labels.squeeze(0).cpu().numpy().astype(np.uint8)
    
    # 4) Expand to [original_height, original_width, 1], as required
    labels = np.expand_dims(labels, axis=-1)

    return labels
