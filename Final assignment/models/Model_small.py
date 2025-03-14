import torch
import torch.nn as nn
from .hrnet import hrnet48
from unet import UNet

def get_model(device, num_classes=19):
    """
    Return an instance of HRNet for segmentation, with the final layer modified
    to produce `num_classes` output channels.
    """
    # Instantiate HRNet (using hrnet48 as an example). Set pretrained=False to train from scratch.
    model = UNet(in_channels=3, num_classes=8).to(device)
    return model

def remap_label_small(mask, ignore_index=255):
    """
    Remaps Cityscapes train IDs so that the 'small' classes become 1..7,
    everything else is 0 (background), and originally ignored pixels (255)
    remain 255 (so they're skipped by CrossEntropyLoss).
    
    'small' classes (train IDs):
      5   = pole
      6   = traffic light
      7   = traffic sign
      11  = person
      12  = rider
      17  = motorcycle
      18  = bicycle
    """
    new_mask = torch.zeros_like(mask)  # everything defaults to background=0
    
    # Map small classes to 1..7
    class_mapping = {
        5 : 1,
        6 : 2,
        7 : 3,
        11: 4,
        12: 5,
        17: 6,
        18: 7,
    }
    for old_id, new_id in class_mapping.items():
        new_mask[mask == old_id] = new_id
    
    # Keep originally ignored pixels (255) as 255
    new_mask[mask == 255] = ignore_index
    return new_mask
