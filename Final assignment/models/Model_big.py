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

def decode_label_big(pred, ignore_index=255):
    """
    'pred' is the output from the Big model, where:
        0 -> background
        1 -> road       (original Cityscapes trainID=0)
        2 -> building   (2)
        3 -> vegetation (8)
        4 -> terrain    (9)
        5 -> sky        (10)
        6 -> truck      (14)
        7 -> bus        (15)
        8 -> train      (16)

    We map them back to their Cityscapes train IDs.
    """
    new_pred = torch.full_like(pred, ignore_index)  # start filled with 255

    class_mapping = {
        1: 0,   # sub-model 1 -> cityscapes 0 (road)
        2: 2,   # building
        3: 8,   # vegetation
        4: 9,   # terrain
        5: 10,  # sky
        6: 14,  # truck
        7: 15,  # bus
        8: 16,  # train
    }
    for sub_id, city_id in class_mapping.items():
        new_pred[pred == sub_id] = city_id

    # Everything that was 0 in sub-model is 'background' => keep 255 or 0,
    # but commonly we set background to 255 for easy ignoring. Already set above.
    return new_pred
