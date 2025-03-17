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

def decode_label_small(pred, ignore_index=255):
    """
    'pred' is the output from the Small model, where:
        0 -> background
        1 -> pole           (Cityscapes trainID=5)
        2 -> traffic light  (6)
        3 -> traffic sign   (7)
        4 -> person         (11)
        5 -> rider          (12)
        6 -> motorcycle     (17)
        7 -> bicycle        (18)
    """
    new_pred = torch.full_like(pred, ignore_index)

    class_mapping = {
        1: 5,   # pole
        2: 6,   # traffic light
        3: 7,   # traffic sign
        4: 11,  # person
        5: 12,  # rider
        6: 17,  # motorcycle
        7: 18,  # bicycle
    }
    for sub_id, city_id in class_mapping.items():
        new_pred[pred == sub_id] = city_id

    return new_pred
