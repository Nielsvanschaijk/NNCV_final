import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
import wandb

from torchvision.transforms.v2 import (
    Compose,
    RandomHorizontalFlip,
    RandomResizedCrop,
    ColorJitter,
    GaussianBlur,
    ToImage,
    ToDtype,
    Normalize
)

from torchvision.datasets import Cityscapes, wrap_dataset_for_transforms_v2
from torchvision.transforms.v2 import Compose, Normalize, Resize, ToImage, ToDtype
from torchvision.utils import make_grid

# Dice losses for each model variant
# NOTE: We will manually apply ignore-masking in this script below, before calling these functions.
from dice_loss import (
    multiclass_dice_loss_small,
    multiclass_dice_loss_medium,
    multiclass_dice_loss_big
)

# Submodel components
from models.Model_small import (
    get_model as get_small_model,
    remap_label_small as remap_small,
    decode_label_small
)
from models.Model_medium import (
    get_model as get_medium_model,
    remap_label_medium as remap_medium,
    decode_label_medium
)
from models.Model_big import (
    get_model as get_big_model,
    remap_label_big as remap_big,
    decode_label_big
)

# Utility functions
from utils import (
    convert_to_train_id,
    convert_train_id_to_color
)

from argparse import ArgumentParser

###########################################
# (NEW) Colorize the 3-head contribution
###########################################
def colorize_contribution(contrib_map: torch.Tensor) -> torch.Tensor:
    """
    Converts a [B, H, W] tensor with values in {0,1,2,3} into an RGB image
    [B, 3, H, W], where:
       0 -> black (no head contributed; background)
       1 -> red   (small head contributed)
       2 -> green (medium head contributed)
       3 -> blue  (big head contributed)
    """
    B, H, W = contrib_map.shape
    color_image = torch.zeros((B, 3, H, W), dtype=torch.uint8, device=contrib_map.device)

    # 1 => red channel
    color_image[:, 0, :, :][contrib_map == 1] = 255

    # 2 => green channel
    color_image[:, 1, :, :][contrib_map == 2] = 255

    # 3 => blue channel
    color_image[:, 2, :, :][contrib_map == 3] = 255

    return color_image

################################################################################
# Inverse mappings for each head (sub-ID -> Cityscapes trainID).
# Adjust as needed to match your actual "remap_label_*" functions
################################################################################

inverse_small = {
    0: 255,   # background / not handled by small
    1: 5,
    2: 6,
    3: 7,
    4: 11,
    5: 12,
    6: 17,
    7: 18
}

inverse_medium = {
    0: 255,  # background
    1: 1,    # sidewalk
    2: 3,    # wall
    3: 4,    # fence
    4: 13,   # car
}

inverse_big = {
    0: 255,  # background
    # EXAMPLE only – fill in with actual classes you handle in "remap_label_big"
    1: 0,   # road
    2: 2,   # building
    3: 8,   # etc
    4: 10,
    5: 15,
    6: 16,
    7: 18,  # maybe overlaps with small – that's ok
    8: 5    # etc
}


################################################################################
# (NEW) Confidence-based composition
################################################################################
def confidence_composition(logits_small, logits_medium, logits_big,
                           inv_map_small, inv_map_medium, inv_map_big):
    """
    logits_small:  [B, Cs, H, W] raw model outputs from small head
    logits_medium: [B, Cm, H, W]
    logits_big:    [B, Cb, H, W]

    inv_map_small:  dict { subID -> cityID }
    inv_map_medium: dict { subID -> cityID }
    inv_map_big:    dict { subID -> cityID }

    Returns:
      composed_pred: [B, H, W] in 0..18 (Cityscapes trainIDs)
    """
    device = logits_small.device
    B, _, H, W = logits_small.shape

    # We'll store -9999 as "impossible" so that if a class hasn't been set, it won't get chosen by argmax
    composed_scores = torch.full((B, 19, H, W), -9999.0, device=device)

    # Fill from small head
    for sub_id in range(logits_small.shape[1]):
        city_id = inv_map_small.get(sub_id, 255)
        if city_id < 19:
            # choose the max of old vs new
            composed_scores[:, city_id, :, :] = torch.maximum(
                composed_scores[:, city_id, :, :],
                logits_small[:, sub_id, :, :]
            )

    # Fill from medium head
    for sub_id in range(logits_medium.shape[1]):
        city_id = inv_map_medium.get(sub_id, 255)
        if city_id < 19:
            composed_scores[:, city_id, :, :] = torch.maximum(
                composed_scores[:, city_id, :, :],
                logits_medium[:, sub_id, :, :]
            )

    # Fill from big head
    for sub_id in range(logits_big.shape[1]):
        city_id = inv_map_big.get(sub_id, 255)
        if city_id < 19:
            composed_scores[:, city_id, :, :] = torch.maximum(
                composed_scores[:, city_id, :, :],
                logits_big[:, sub_id, :, :]
            )

    # Argmax over the 19 channels => [B, H, W]
    composed_pred = composed_scores.argmax(dim=1)
    return composed_pred


################################################################################
# MultiHead UNet definition
################################################################################
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super(DoubleConv, self).__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up, self).__init__()
        # Use bilinear upsampling by default
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # Ensure size match by padding if needed
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # Concatenate along channel dimension
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    """1x1 convolution for the final output"""
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class MultiHeadUNet(nn.Module):
    """
    A multi-head segmentation network with a shared UNet encoder and three separate decoder heads.
    - Small head predicts 8 classes.
    - Medium head predicts 5 classes.
    - Big head predicts 9 classes.
    """
    def __init__(self, in_channels=3):
        super(MultiHeadUNet, self).__init__()
        # Shared encoder
        self.inc   = DoubleConv(in_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)  # Bottleneck

        # SMALL HEAD (8 classes)
        self.up1_small = Up(512 + 512, 256)
        self.up2_small = Up(256 + 256, 128)
        self.up3_small = Up(128 + 128, 64)
        self.up4_small = Up(64 + 64, 64)
        self.outc_small = OutConv(64, 8)

        # MEDIUM HEAD (5 classes)
        self.up1_medium = Up(512 + 512, 256)
        self.up2_medium = Up(256 + 256, 128)
        self.up3_medium = Up(128 + 128, 64)
        self.up4_medium = Up(64 + 64, 64)
        self.outc_medium = OutConv(64, 5)

        # BIG HEAD (9 classes)
        self.up1_big = Up(512 + 512, 256)
        self.up2_big = Up(256 + 256, 128)
        self.up3_big = Up(128 + 128, 64)
        self.up4_big = Up(64 + 64, 64)
        self.outc_big = OutConv(64, 9)

    def forward(self, x):
        # Encoder
        x1 = self.inc(x)    
        x2 = self.down1(x1) 
        x3 = self.down2(x2) 
        x4 = self.down3(x3) 
        x5 = self.down4(x4) 

        # SMALL HEAD
        x_s = self.up1_small(x5, x4)
        x_s = self.up2_small(x_s, x3)
        x_s = self.up3_small(x_s, x2)
        x_s = self.up4_small(x_s, x1)
        logits_small = self.outc_small(x_s)

        # MEDIUM HEAD
        x_m = self.up1_medium(x5, x4)
        x_m = self.up2_medium(x_m, x3)
        x_m = self.up3_medium(x_m, x2)
        x_m = self.up4_medium(x_m, x1)
        logits_medium = self.outc_medium(x_m)

        # BIG HEAD
        x_b = self.up1_big(x5, x4)
        x_b = self.up2_big(x_b, x3)
        x_b = self.up3_big(x_b, x2)
        x_b = self.up4_big(x_b, x1)
        logits_big = self.outc_big(x_b)

        return logits_small, logits_medium, logits_big

################################################################################
# Standard training script
################################################################################
def get_args_parser():
    parser = ArgumentParser("Training script for multi-head UNet segmentation.")
    parser.add_argument("--data-dir", type=str, default="./data/cityscapes", help="Path to Cityscapes data")
    parser.add_argument("--batch-size", type=int, default=16, help="Training batch size")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.005, help="Learning rate")
    parser.add_argument("--num-workers", type=int, default=10, help="DataLoader num_workers")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--experiment-id", type=str, default="unet-training", help="Experiment ID for Weights & Biases")
    return parser

def update_confusion_matrix(conf_mat: torch.Tensor, pred: torch.Tensor, target: torch.Tensor, ignore_index=255) -> torch.Tensor:
    if conf_mat.device != pred.device:
        conf_mat = conf_mat.to(pred.device)
    pred = pred.view(-1)
    target = target.view(-1)
    valid_mask = (target != ignore_index)
    pred = pred[valid_mask]
    target = target[valid_mask]
    indices = target * conf_mat.shape[0] + pred
    conf_mat += torch.bincount(indices, minlength=conf_mat.numel()).view(conf_mat.size())
    return conf_mat

def compute_miou(conf_mat: torch.Tensor) -> float:
    intersection = torch.diag(conf_mat)
    gt_plus_pred = conf_mat.sum(dim=1) + conf_mat.sum(dim=0) - intersection
    valid_mask = (gt_plus_pred > 0)
    iou_per_class = torch.zeros_like(intersection, dtype=torch.float)
    iou_per_class[valid_mask] = intersection[valid_mask].float() / gt_plus_pred[valid_mask].float()
    return iou_per_class.mean().item()

def main(args):
    print(f"Checking dataset path: {args.data_dir}")
    if os.path.exists(args.data_dir):
        print("Contents:", os.listdir(args.data_dir))
    else:
        print("Data directory not found!")
        return

    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    wandb.finish()
    wandb.init(
        project="5lsm0-cityscapes-segmentation",
        name=args.experiment_id,
        config=vars(args),
        reinit=True,
    )

    # Basic transforms
    transform = Compose([
        ToImage(),
        RandomHorizontalFlip(p=0.5),
        RandomResizedCrop(size=(512, 1024), scale=(0.8, 1.2), ratio=(2.0, 2.0), antialias=True),
        ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0)),
        ToDtype(torch.float32, scale=True),
        Normalize((0.286, 0.325, 0.283), (0.176, 0.180, 0.177))
    ])

    train_dataset = Cityscapes(args.data_dir, split="train", mode="fine", target_type="semantic", transforms=transform)
    valid_dataset = Cityscapes(args.data_dir, split="val", mode="fine", target_type="semantic", transforms=transform)
    train_dataset = wrap_dataset_for_transforms_v2(train_dataset)
    valid_dataset = wrap_dataset_for_transforms_v2(valid_dataset)

    train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=4)
    valid_dataloader = DataLoader(valid_dataset, batch_size=2, shuffle=False, num_workers=4)

    os.makedirs("checkpoints", exist_ok=True)

    model = MultiHeadUNet(in_channels=3).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr)

    # CE losses
    criterion_small_ce = nn.CrossEntropyLoss(ignore_index=255)
    criterion_medium_ce = nn.CrossEntropyLoss(ignore_index=255)
    criterion_big_ce = nn.CrossEntropyLoss(ignore_index=255)

    # Weighted combination
    dice_weight = 1.0
    ce_weight = 1.0
    SMALL_WEIGHT = 1.0
    MEDIUM_WEIGHT = 1.0
    BIG_WEIGHT = 1.0

    best_val_loss_overall = float("inf")

    for epoch in range(args.epochs):
        print(f"Epoch [{epoch+1}/{args.epochs}]")
        model.train()

        train_losses_small = []
        train_losses_medium = []
        train_losses_big = []
        train_losses_small_dice = []
        train_losses_medium_dice = []
        train_losses_big_dice = []
        train_losses_small_ce = []
        train_losses_medium_ce = []
        train_losses_big_ce = []

        # ---------------------------
        # Training loop
        # ---------------------------
        for images, labels in train_dataloader:
            images = images.to(device)
            labels_trainid = convert_to_train_id(labels)

            optimizer.zero_grad()
            out_small, out_medium, out_big = model(images)

            # Prepare sub-head labels
            labels_small = remap_small(labels_trainid).long().squeeze(1).to(device)
            labels_medium = remap_medium(labels_trainid).long().squeeze(1).to(device)
            labels_big = remap_big(labels_trainid).long().squeeze(1).to(device)

            # Create ignore masks
            ignore_mask_small  = (labels_small  != 255).unsqueeze(1).float()
            ignore_mask_medium = (labels_medium != 255).unsqueeze(1).float()
            ignore_mask_big    = (labels_big    != 255).unsqueeze(1).float()

            # For dice, set 255->0 but won't matter because ignore mask = 0
            labels_small_dice = labels_small.clone()
            labels_small_dice[labels_small_dice == 255] = 0
            labels_medium_dice = labels_medium.clone()
            labels_medium_dice[labels_medium_dice == 255] = 0
            labels_big_dice = labels_big.clone()
            labels_big_dice[labels_big_dice == 255] = 0

            # Upsample if needed
            out_small_up = F.interpolate(out_small, size=labels_small.shape[-2:], mode='bilinear', align_corners=True)
            out_medium_up = F.interpolate(out_medium, size=labels_medium.shape[-2:], mode='bilinear', align_corners=True)
            out_big_up = F.interpolate(out_big, size=labels_big.shape[-2:], mode='bilinear', align_corners=True)

            # One-hot for dice
            n_s = out_small_up.shape[1]
            n_m = out_medium_up.shape[1]
            n_b = out_big_up.shape[1]
            oh_small = F.one_hot(labels_small_dice, n_s).permute(0,3,1,2).float()
            oh_medium = F.one_hot(labels_medium_dice, n_m).permute(0,3,1,2).float()
            oh_big = F.one_hot(labels_big_dice, n_b).permute(0,3,1,2).float()

            # Multiply by ignore masks
            s_probs = F.softmax(out_small_up, dim=1) * ignore_mask_small
            oh_s = oh_small * ignore_mask_small
            m_probs = F.softmax(out_medium_up, dim=1) * ignore_mask_medium
            oh_m = oh_medium * ignore_mask_medium
            b_probs = F.softmax(out_big_up, dim=1) * ignore_mask_big
            oh_b = oh_big * ignore_mask_big

            # dice losses
            loss_s_dice = multiclass_dice_loss_small(s_probs, oh_s)
            loss_m_dice = multiclass_dice_loss_medium(m_probs, oh_m)
            loss_b_dice = multiclass_dice_loss_big(b_probs, oh_b)

            # CE losses
            loss_s_ce = criterion_small_ce(out_small_up, labels_small)
            loss_m_ce = criterion_medium_ce(out_medium_up, labels_medium)
            loss_b_ce = criterion_big_ce(out_big_up, labels_big)

            # Weighted sum
            loss_s = dice_weight*loss_s_dice + ce_weight*loss_s_ce
            loss_m = dice_weight*loss_m_dice + ce_weight*loss_m_ce
            loss_b = dice_weight*loss_b_dice + ce_weight*loss_b_ce

            loss_total = (SMALL_WEIGHT*loss_s +
                          MEDIUM_WEIGHT*loss_m +
                          BIG_WEIGHT*loss_b)
            loss_total.backward()
            optimizer.step()

            # Bookkeeping
            train_losses_small.append(loss_s.item())
            train_losses_small_dice.append(loss_s_dice.item())
            train_losses_small_ce.append(loss_s_ce.item())

            train_losses_medium.append(loss_m.item())
            train_losses_medium_dice.append(loss_m_dice.item())
            train_losses_medium_ce.append(loss_m_ce.item())

            train_losses_big.append(loss_b.item())
            train_losses_big_dice.append(loss_b_dice.item())
            train_losses_big_ce.append(loss_b_ce.item())

        # Average training losses
        avg_loss_small = sum(train_losses_small)/len(train_losses_small)
        avg_loss_medium = sum(train_losses_medium)/len(train_losses_medium)
        avg_loss_big = sum(train_losses_big)/len(train_losses_big)

        avg_loss_small_dice = sum(train_losses_small_dice)/len(train_losses_small_dice)
        avg_loss_small_ce   = sum(train_losses_small_ce)/len(train_losses_small_ce)
        avg_loss_medium_dice = sum(train_losses_medium_dice)/len(train_losses_medium_dice)
        avg_loss_medium_ce   = sum(train_losses_medium_ce)/len(train_losses_medium_ce)
        avg_loss_big_dice    = sum(train_losses_big_dice)/len(train_losses_big_dice)
        avg_loss_big_ce      = sum(train_losses_big_ce)/len(train_losses_big_ce)

        # Validation
        model.eval()
        val_losses_small, val_losses_medium, val_losses_big = [], [], []
        val_losses_small_dice, val_losses_medium_dice, val_losses_big_dice = [], [], []
        val_losses_small_ce, val_losses_medium_ce, val_losses_big_ce = [], [], []

        # For confusion matrices
        conf_mat_small = torch.zeros(8, 8, dtype=torch.int64)
        conf_mat_medium = torch.zeros(5, 5, dtype=torch.int64)
        conf_mat_big = torch.zeros(9, 9, dtype=torch.int64)
        conf_mat_composed = torch.zeros(19, 19, dtype=torch.int64)

        logged_images = False

        with torch.no_grad():
            for i, (images, labels) in enumerate(valid_dataloader):
                images = images.to(device)
                labels_trainid = convert_to_train_id(labels).to(device)

                out_small, out_medium, out_big = model(images)

                # Evaluate the sub-head predictions for small, medium, big
                # (similar dice+CE logic as training)
                lab_s = remap_small(labels_trainid).long().squeeze(1)
                lab_m = remap_medium(labels_trainid).long().squeeze(1)
                lab_b = remap_big(labels_trainid).long().squeeze(1)

                ignore_s = (lab_s != 255).unsqueeze(1).float()
                ignore_m = (lab_m != 255).unsqueeze(1).float()
                ignore_b = (lab_b != 255).unsqueeze(1).float()

                out_s_up = F.interpolate(out_small, size=lab_s.shape[-2:], mode='bilinear')
                out_m_up = F.interpolate(out_medium, size=lab_m.shape[-2:], mode='bilinear')
                out_b_up = F.interpolate(out_big,    size=lab_b.shape[-2:], mode='bilinear')

                # dice
                lab_s_dice = lab_s.clone()
                lab_s_dice[lab_s_dice == 255] = 0
                oh_s = F.one_hot(lab_s_dice, out_s_up.shape[1]).permute(0,3,1,2).float()
                s_probs = F.softmax(out_s_up, dim=1)*ignore_s
                oh_s = oh_s*ignore_s
                loss_s_dice_val = multiclass_dice_loss_small(s_probs, oh_s)

                loss_s_ce_val = criterion_small_ce(out_s_up, lab_s)
                val_s = dice_weight*loss_s_dice_val + ce_weight*loss_s_ce_val
                val_losses_small_dice.append(loss_s_dice_val.item())
                val_losses_small_ce.append(loss_s_ce_val.item())
                val_losses_small.append(val_s.item())

                # update conf_mat for small alone
                preds_small = out_s_up.softmax(dim=1).argmax(dim=1)
                conf_mat_small = update_confusion_matrix(conf_mat_small, preds_small, lab_s)

                # medium
                lab_m_dice = lab_m.clone()
                lab_m_dice[lab_m_dice == 255] = 0
                oh_m = F.one_hot(lab_m_dice, out_m_up.shape[1]).permute(0,3,1,2).float()
                m_probs = F.softmax(out_m_up, dim=1)*ignore_m
                oh_m = oh_m*ignore_m
                loss_m_dice_val = multiclass_dice_loss_medium(m_probs, oh_m)

                loss_m_ce_val = criterion_medium_ce(out_m_up, lab_m)
                val_m = dice_weight*loss_m_dice_val + ce_weight*loss_m_ce_val
                val_losses_medium_dice.append(loss_m_dice_val.item())
                val_losses_medium_ce.append(loss_m_ce_val.item())
                val_losses_medium.append(val_m.item())

                preds_medium = out_m_up.softmax(dim=1).argmax(dim=1)
                conf_mat_medium = update_confusion_matrix(conf_mat_medium, preds_medium, lab_m)

                # big
                lab_b_dice = lab_b.clone()
                lab_b_dice[lab_b_dice == 255] = 0
                oh_b = F.one_hot(lab_b_dice, out_b_up.shape[1]).permute(0,3,1,2).float()
                b_probs = F.softmax(out_b_up, dim=1)*ignore_b
                oh_b = oh_b*ignore_b
                loss_b_dice_val = multiclass_dice_loss_big(b_probs, oh_b)

                loss_b_ce_val = criterion_big_ce(out_b_up, lab_b)
                val_b = dice_weight*loss_b_dice_val + ce_weight*loss_b_ce_val
                val_losses_big_dice.append(loss_b_dice_val.item())
                val_losses_big_ce.append(loss_b_ce_val.item())
                val_losses_big.append(val_b.item())

                preds_big = out_b_up.softmax(dim=1).argmax(dim=1)
                conf_mat_big = update_confusion_matrix(conf_mat_big, preds_big, lab_b)

                # --- Now the COMPOSED MODEL using confidence_composition ---
                composed_pred = confidence_composition(
                    out_s_up, out_m_up, out_b_up,
                    inverse_small, inverse_medium, inverse_big
                )
                conf_mat_composed = update_confusion_matrix(
                    conf_mat_composed,
                    composed_pred,
                    labels_trainid.squeeze(1),
                    ignore_index=255
                )

                # optional colorizing "who contributed"
                model_contribution = torch.zeros_like(composed_pred, dtype=torch.uint8)
                # Mark small=1 where preds_small != background ID
                model_contribution[preds_small != 0] = 1
                # Mark medium=2 where it's still 0
                still_zero = (model_contribution == 0)
                model_contribution[still_zero & (preds_medium != 0)] = 2
                # Mark big=3
                still_zero = (model_contribution == 0)
                model_contribution[still_zero & (preds_big != 0)] = 3

                # Optionally log images
                if not logged_images:
                    # decode predictions for each sub-head
                    dec_small = decode_label_small(preds_small)
                    dec_medium = decode_label_medium(preds_medium)
                    dec_big = decode_label_big(preds_big)

                    composed_pred_color = convert_train_id_to_color(composed_pred.unsqueeze(1))
                    composed_pred_img = make_grid(composed_pred_color.cpu(), nrow=4).permute(1, 2, 0).numpy()

                    pred_small_color = convert_train_id_to_color(dec_small.unsqueeze(1))
                    pred_small_img = make_grid(pred_small_color.cpu(), nrow=4).permute(1, 2, 0).numpy()

                    pred_medium_color = convert_train_id_to_color(dec_medium.unsqueeze(1))
                    pred_medium_img = make_grid(pred_medium_color.cpu(), nrow=4).permute(1, 2, 0).numpy()

                    pred_big_color = convert_train_id_to_color(dec_big.unsqueeze(1))
                    pred_big_img = make_grid(pred_big_color.cpu(), nrow=4).permute(1, 2, 0).numpy()

                    gt_color = convert_train_id_to_color(labels_trainid)
                    gt_img = make_grid(gt_color.cpu(), nrow=4).permute(1, 2, 0).numpy()

                    contrib_color = colorize_contribution(model_contribution)
                    contrib_color_img = make_grid(contrib_color.cpu(), nrow=4).permute(1, 2, 0).numpy()

                    if epoch % 5 == 0:
                        wandb.log({
                            "val_small_prediction": [wandb.Image(pred_small_img)],
                            "val_medium_prediction": [wandb.Image(pred_medium_img)],
                            "val_big_prediction": [wandb.Image(pred_big_img)],
                            "val_composed_prediction": [wandb.Image(composed_pred_img)],
                            "val_ground_truth": [wandb.Image(gt_img)],
                            "val_contribution_map": [wandb.Image(contrib_color_img)],
                        })
                    logged_images = True

        # Summaries
        avg_val_small = (sum(val_losses_small)/len(val_losses_small)) if val_losses_small else 0
        avg_val_medium = (sum(val_losses_medium)/len(val_losses_medium)) if val_losses_medium else 0
        avg_val_big = (sum(val_losses_big)/len(val_losses_big)) if val_losses_big else 0

        avg_val_small_dice = (sum(val_losses_small_dice)/len(val_losses_small_dice)) if val_losses_small_dice else 0
        avg_val_small_ce = (sum(val_losses_small_ce)/len(val_losses_small_ce)) if val_losses_small_ce else 0
        avg_val_medium_dice = (sum(val_losses_medium_dice)/len(val_losses_medium_dice)) if val_losses_medium_dice else 0
        avg_val_medium_ce = (sum(val_losses_medium_ce)/len(val_losses_medium_ce)) if val_losses_medium_ce else 0
        avg_val_big_dice = (sum(val_losses_big_dice)/len(val_losses_big_dice)) if val_losses_big_dice else 0
        avg_val_big_ce = (sum(val_losses_big_ce)/len(val_losses_big_ce)) if val_losses_big_ce else 0

        val_loss = (avg_val_small + avg_val_medium + avg_val_big) / 3.0
        miou_small = compute_miou(conf_mat_small)
        miou_medium = compute_miou(conf_mat_medium)
        miou_big = compute_miou(conf_mat_big)
        miou_composed = compute_miou(conf_mat_composed)

        wandb.log({
            "train_loss_small": avg_loss_small,
            "train_loss_medium": avg_loss_medium,
            "train_loss_big": avg_loss_big,
            "train_loss_small_dice": avg_loss_small_dice,
            "train_loss_small_ce": avg_loss_small_ce,
            "train_loss_medium_dice": avg_loss_medium_dice,
            "train_loss_medium_ce": avg_loss_medium_ce,
            "train_loss_big_dice": avg_loss_big_dice,
            "train_loss_big_ce": avg_loss_big_ce,

            "val_loss": val_loss,
            "val_loss_small": avg_val_small,
            "val_loss_medium": avg_val_medium,
            "val_loss_big": avg_val_big,
            "val_loss_small_dice": avg_val_small_dice,
            "val_loss_small_ce": avg_val_small_ce,
            "val_loss_medium_dice": avg_val_medium_dice,
            "val_loss_medium_ce": avg_val_medium_ce,
            "val_loss_big_dice": avg_val_big_dice,
            "val_loss_big_ce": avg_val_big_ce,

            "val_mIoU_small": miou_small,
            "val_mIoU_medium": miou_medium,
            "val_mIoU_big": miou_big,
            "val_mIoU_composed": miou_composed,
            "epoch": epoch+1,
        })

        # Save if best
        if val_loss < best_val_loss_overall:
            best_val_loss_overall = val_loss
            ckpt_path = os.path.join("checkpoints", "best_model_overall.pth")
            torch.save(model.state_dict(), ckpt_path)
            print(f"New best OVERALL model saved at epoch {epoch+1} with avg_val_loss={val_loss:.4f} -> {ckpt_path}")

    wandb.finish()
    print("Training complete.")

if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)
