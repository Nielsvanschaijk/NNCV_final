import torch
import torch.nn as nn
from network.hrnet import hrnet48

def get_model(device, num_classes=8):
    model = hrnet48(pretrained=False, progress=False)

    # Load the Cityscapes checkpoint
    checkpoint_path = 'hrnet_w48_cityscapes_cls19_1024x2048_trainset.pth'
    state_dict = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(state_dict, strict=False)

    # Replace the final layer to output 8 channels instead of 19
    if isinstance(model.last_layer, nn.Sequential) and len(model.last_layer) >= 4:
        in_channels = model.last_layer[-1].in_channels
        # Create a new Conv2d layer with 8 output channels (0: background, 1-7: small classes)
        model.last_layer[-1] = nn.Conv2d(in_channels, num_classes, kernel_size=1, stride=1)
    else:
        raise RuntimeError("Unexpected HRNet final layer structure. Adjust code accordingly.")

    # Freeze all parameters except for the new final layer
    for name, param in model.named_parameters():
        if 'last_layer' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    return model.to(device)

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
