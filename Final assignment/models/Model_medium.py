import torch
from unet import UNet

def get_model(device, num_classes=19):
    """Return an instance of the UNet model for small classes."""
    model = UNet(in_channels=3, num_classes=5).to(device)
    return model

def remap_label_medium(mask, ignore_index=255):
    """
    Remaps Cityscapes train IDs so that the 'medium' classes become 1..4,
    everything else is 0 (background), and originally ignored pixels (255)
    remain 255.
    
    'medium' classes (train IDs):
      1  = sidewalk
      3  = wall
      4  = fence
      13 = car
    """
    new_mask = torch.zeros_like(mask)

    class_mapping = {
        1 : 1,
        3 : 2,
        4 : 3,
        13: 4,
    }
    for old_id, new_id in class_mapping.items():
        new_mask[mask == old_id] = new_id

    new_mask[mask == 255] = ignore_index
    return new_mask

def decode_label_medium(pred, ignore_index=255):
    """
    'pred' is the output from the Medium model, where:
        0 -> background
        1 -> sidewalk   (Cityscapes trainID=1)
        2 -> wall       (3)
        3 -> fence      (4)
        4 -> car        (13)
    """
    new_pred = torch.full_like(pred, ignore_index)

    class_mapping = {
        1: 1,   # sub-model 1 -> cityscapes 1  (sidewalk)
        2: 3,   # wall
        3: 4,   # fence
        4: 13,  # car
    }
    for sub_id, city_id in class_mapping.items():
        new_pred[pred == sub_id] = city_id

    return new_pred
