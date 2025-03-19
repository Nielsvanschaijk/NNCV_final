import torch
import torch.nn as nn
from network.hrnet import hrnet48
from unet import UNet

def get_model(device, num_classes=19):
    """Return an instance of the UNet model for small classes."""
    model = UNet(in_channels=3, num_classes=8).to(device)
    return model

def remap_label_small(mask, ignore_index=255):
    """
    Remap Cityscapes train IDs into 8 classes for training.
    
    Mapping:
        Cityscapes train IDs -> new class index:
          5   (pole)           -> 1
          6   (traffic light)  -> 2
          7   (traffic sign)   -> 3
          11  (person)         -> 4
          12  (rider)          -> 5
          17  (motorcycle)     -> 6
          18  (bicycle)        -> 7
        All other classes become background (0), and ignored pixels (255) remain 255.
    """
    # Start with background (0) everywhere
    new_mask = torch.zeros_like(mask)
    
    # Define the mapping for the 7 classes
    class_mapping = {
        5: 1,    # pole
        6: 2,    # traffic light
        7: 3,    # traffic sign
        11: 4,   # person
        12: 5,   # rider
        17: 6,   # motorcycle
        18: 7,   # bicycle
    }
    for old_id, new_id in class_mapping.items():
        new_mask[mask == old_id] = new_id

    new_mask[mask == 255] = ignore_index
    return new_mask

def decode_label_small(pred, ignore_index=255):
    """
    Convert predictions (with 8 channels) back to Cityscapes train IDs.
    
    Mapping (inverse of remap_label_small):
        Prediction index -> Cityscapes train ID:
          1 -> 5    (pole)
          2 -> 6    (traffic light)
          3 -> 7    (traffic sign)
          4 -> 11   (person)
          5 -> 12   (rider)
          6 -> 17   (motorcycle)
          7 -> 18   (bicycle)
        Background (0) remains 0, and pixels with ignore_index remain as ignore_index.
    """
    # Initialize with ignore_index
    new_pred = torch.full_like(pred, ignore_index)
    
    # Inverse mapping for the 7 classes
    class_mapping = {
        1: 5,   # pole
        2: 6,   # traffic light
        3: 7,   # traffic sign
        4: 11,  # person
        5: 12,  # rider
        6: 17,  # motorcycle
        7: 18,  # bicycle
    }
    # Assign Cityscapes train IDs for the mapped classes
    for sub_id, city_id in class_mapping.items():
        new_pred[pred == sub_id] = city_id

    # Background remains as 0
    new_pred[pred == 0] = 0

    return new_pred
