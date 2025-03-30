import torch
import torch.nn as nn
from network.hrnet import hrnet48  # adjust the import if needed

class HRNetSmall(nn.Module):
    def __init__(self, device):
        super(HRNetSmall, self).__init__()
        # Instantiate HRNet with pretrained disabled
        self.backbone = hrnet48(pretrained=False)
        
        # Load your checkpoint and remove any "model." prefix from keys
        state_dict = torch.load("hrnet.pth", map_location=device)
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("model."):
                new_state_dict[k[len("model."):]] = v
            else:
                new_state_dict[k] = v
        # Load state dict in non-strict mode (ignoring head mismatches)
        self.backbone.load_state_dict(new_state_dict, strict=False)
        
        # Freeze all backbone parameters
        for param in self.backbone.parameters():
            param.requires_grad = False

    def forward(self, x):
        # Get raw HRNet logits (shape: [B, 19, H, W])
        logits = self.backbone(x)
        return logits

def get_model(device):
    model = HRNetSmall(device)
    model.to(device)
    return model

def remap_label_small(mask, ignore_index=255):
    """
    Remap Cityscapes train IDs (from HRNet’s 19 classes) into 8 classes.
    
    Mapping:
      Cityscapes ID 5   -> 1
      Cityscapes ID 6   -> 2
      Cityscapes ID 7   -> 3
      Cityscapes ID 11  -> 4
      Cityscapes ID 12  -> 5
      Cityscapes ID 17  -> 6
      Cityscapes ID 18  -> 7
      
    All other classes (except ignored 255) become background (0).
    """
    new_mask = torch.zeros_like(mask)
    mapping = {5: 1, 6: 2, 7: 3, 11: 4, 12: 5, 17: 6, 18: 7}
    for old_id, new_id in mapping.items():
        new_mask[mask == old_id] = new_id
    # For pixels not in the mapping (and not ignored), set background (0)
    valid_mask = (mask != 255) & (~torch.isin(mask, torch.tensor(list(mapping.keys()), device=mask.device)))
    new_mask[valid_mask] = 0
    new_mask[mask == 255] = ignore_index
    return new_mask

def decode_label_small(pred, ignore_index=255):
    """
    Convert predictions (with 8 channels) back to Cityscapes train IDs.
    
    Inverse Mapping:
      1 -> 5, 2 -> 6, 3 -> 7, 4 -> 11, 5 -> 12, 6 -> 17, 7 -> 18.
    Background (0) remains as ignore_index.
    """
    new_pred = torch.full_like(pred, ignore_index)
    mapping = {1: 5, 2: 6, 3: 7, 4: 11, 5: 12, 6: 17, 7: 18}
    for sub_id, city_id in mapping.items():
        new_pred[pred == sub_id] = city_id
    return new_pred
