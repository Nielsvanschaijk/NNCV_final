import torch
import torch.nn as nn
from network.hrnet import hrnet48  # adjust the import if needed

class HRNetSmall(nn.Module):
    def __init__(self, device, num_classes=8):
        super(HRNetSmall, self).__init__()
        # Instantiate the HRNet backbone with pretrained disabled
        self.backbone = hrnet48(pretrained=False)
        
        # Load your checkpoint and strip the "model." prefix if present
        state_dict = torch.load("hrnet.pth", map_location=device)
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("model."):
                new_state_dict[k[len("model."):]] = v
            else:
                new_state_dict[k] = v
        # Load with strict=False to ignore mismatches in the head
        self.backbone.load_state_dict(new_state_dict, strict=False)
        
        # Freeze backbone parameters
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        # Since the backbone returns 19 channels, update seg_head to accept 19 channels.
        self.seg_head = nn.Conv2d(19, num_classes, kernel_size=1)

    def forward(self, x):
        features = self.backbone(x)  # features shape: [B, 19, H, W]
        out = self.seg_head(features)  # now outputs shape: [B, num_classes, H, W]
        return out

def get_model(device, num_classes=8):
    model = HRNetSmall(device, num_classes=num_classes)
    model.to(device)
    return model

def remap_label_small(mask, ignore_index=255):
    """
    Remap Cityscapes train IDs into 8 classes for training.
    Mapping:
        5 -> 1, 6 -> 2, 7 -> 3, 11 -> 4, 12 -> 5, 17 -> 6, 18 -> 7.
    All other classes become background (0), and ignored pixels (255) remain unchanged.
    """
    new_mask = torch.zeros_like(mask)
    class_mapping = {5: 1, 6: 2, 7: 3, 11: 4, 12: 5, 17: 6, 18: 7}
    for old_id, new_id in class_mapping.items():
        new_mask[mask == old_id] = new_id
    new_mask[mask == 255] = ignore_index
    return new_mask

def decode_label_small(pred, ignore_index=255):
    """
    Convert predictions (with 8 channels) back to Cityscapes train IDs.
    Mapping (inverse of remap_label_small):
         1 -> 5, 2 -> 6, 3 -> 7, 4 -> 11, 5 -> 12, 6 -> 17, 7 -> 18.
    Background (0) remains as ignore_index.
    """
    new_pred = torch.full_like(pred, ignore_index)
    class_mapping = {1: 5, 2: 6, 3: 7, 4: 11, 5: 12, 6: 17, 7: 18}
    for sub_id, city_id in class_mapping.items():
        new_pred[pred == sub_id] = city_id
    return new_pred
