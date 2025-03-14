import torch
from unet import UNet

def get_model(device, num_classes=19):
    """Return an instance of the UNet model for small classes."""
    model = UNet(in_channels=3, num_classes=9).to(device)
    return model

def remap_label_big(mask, ignore_index=255):
    """
    Remaps Cityscapes train IDs so that the 'big' classes become 1..8,
    everything else is 0 (background), and originally ignored pixels (255)
    remain 255.
    
    'big' classes (train IDs):
      0  = road
      2  = building
      8  = vegetation
      9  = terrain
      10 = sky
      14 = truck
      15 = bus
      16 = train
    """
    new_mask = torch.zeros_like(mask)

    class_mapping = {
        0 : 1,
        2 : 2,
        8 : 3,
        9 : 4,
        10: 5,
        14: 6,
        15: 7,
        16: 8,
    }
    for old_id, new_id in class_mapping.items():
        new_mask[mask == old_id] = new_id

    new_mask[mask == 255] = ignore_index
    return new_mask