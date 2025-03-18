import torch
import torch.nn as nn
from network.hrnet import hrnet48

def get_model(device, num_classes=8):
    """
    Load an HRNet-W48 model pretrained on Cityscapes, override it to
    produce `num_classes` channels, freeze the backbone, and return it
    as a PyTorch nn.Module that you can forward() in your existing code.

    Assumptions:
      1) You have the file 'hrnet_w48_cityscapes_cls19_1024x2048_trainset.pth'
         (pretrained for 19-class Cityscapes) in the current directory.
      2) Your `network/hrnet.py` defines `hrnet48(...)` which by default
         creates a model with 19 output channels in `model.last_layer`.
      3) This snippet replaces that 19-channel final layer with an 8-channel
         layer (for the 7 "small" classes + 1 background).
    """

    # 1) Build HRNet-W48 with default (ImageNet) or no pretrained weights
    model = hrnet48(pretrained=False, progress=False)
    # If the code in hrnet.py tries to load from the net's default URLs,
    # you can set `pretrained=False`. We'll manually load cityscapes next.

    # 2) Load the Cityscapes checkpoint (19 classes) from file
    checkpoint_path = 'hrnet_w48_cityscapes_cls19_1024x2048_trainset.pth'
    state_dict = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(state_dict, strict=False)

    # 3) Override the final layer (which is a Sequential) to produce 8 channels
    #    The last item in model.last_layer is typically a Conv2d(..., 19, ...)
    #    so we replace it with a Conv2d(..., 8, ...).
    if isinstance(model.last_layer, nn.Sequential) and len(model.last_layer) >= 4:
        in_channels = model.last_layer[-1].in_channels
        model.last_layer[-1] = nn.Conv2d(in_channels, num_classes, kernel_size=1, stride=1)
    else:
        # If the structure differs, adapt accordingly
        raise RuntimeError("Unexpected HRNet final layer structure. Adjust code accordingly.")

    # 4) Freeze everything except the final layer (so the backbone won't get updated)
    for name, param in model.named_parameters():
        if 'last_layer' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    return model.to(device)


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
    'pred' is the output from the "small" model, where:
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
