import torch
import torch.nn as nn
from network.hrnet import hrnet48  # adjust import if needed

class HRNetSmall(nn.Module):
    def __init__(self, device):
        super(HRNetSmall, self).__init__()
        # 1) Instantiate HRNet (19‐channel output)
        self.backbone = hrnet48(pretrained=False)

        # 2) Load your checkpoint
        state_dict = torch.load("hrnet.pth", map_location=device)
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("model."):
                new_state_dict[k[len("model."):]] = v
            else:
                new_state_dict[k] = v
        self.backbone.load_state_dict(new_state_dict, strict=False)

        # 3) Freeze backbone
        for param in self.backbone.parameters():
            param.requires_grad = False

        # 4) Add a NEW trainable layer from 19 -> 8
        #    (the final “classifier” for your 8 classes)
        self.final_conv = nn.Conv2d(19, 8, kernel_size=1)
        # This layer stays unfrozen:
        for param in self.final_conv.parameters():
            param.requires_grad = True

    def forward(self, x):
        # Raw HRNet logits: [B, 19, H, W]
        logits_19 = self.backbone(x)
        # Now project them into 8 channels
        logits_8 = self.final_conv(logits_19)
        return logits_8

def get_model(device):
    model = HRNetSmall(device)
    model.to(device)
    return model


def remap_label_small(mask, ignore_index=255):
    """
    Map Cityscapes train ID => 8 classes:
       5 -> 1, 6 -> 2, 7 -> 3, 11 -> 4, 12 -> 5, 17 -> 6, 18 -> 7
    Otherwise => 0,  and 255 => ignore_index
    """
    new_mask = torch.zeros_like(mask)
    mapping = {5: 1, 6: 2, 7: 3, 11: 4, 12: 5, 17: 6, 18: 7}
    for old_id, new_id in mapping.items():
        new_mask[mask == old_id] = new_id
    valid_mask = (
        (mask != 255) &
        (~torch.isin(mask, torch.tensor(list(mapping.keys()), device=mask.device)))
    )
    new_mask[valid_mask] = 0
    new_mask[mask == 255] = ignore_index
    return new_mask

def decode_label_small(pred, ignore_index=255):
    """
    Inverse mapping:
      1 -> 5, 2 -> 6, ...
    0 => ignore_index
    """
    new_pred = torch.full_like(pred, ignore_index)
    mapping = {1: 5, 2: 6, 3: 7, 4: 11, 5: 12, 6: 17, 7: 18}
    for sub_id, city_id in mapping.items():
        new_pred[pred == sub_id] = city_id
    return new_pred
