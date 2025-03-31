import os
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
import wandb

from torchvision.datasets import Cityscapes, wrap_dataset_for_transforms_v2
from torchvision.transforms.v2 import Compose, Normalize, Resize, ToImage, ToDtype
from torchvision.utils import make_grid

# Dice losses for each model variant
from dice_loss import (
    multiclass_dice_loss_small,
    multiclass_dice_loss_medium,
    multiclass_dice_loss_big
)

# Utility functions (remapping and decoding functions remain the same)
from utils import (
    convert_to_train_id, 
    convert_train_id_to_color, 
    compose_predictions
)
import os
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
import wandb

from torchvision.datasets import Cityscapes, wrap_dataset_for_transforms_v2
from torchvision.transforms.v2 import Compose, Normalize, Resize, ToImage, ToDtype
from torchvision.utils import make_grid

# Dice losses for each model variant
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
    convert_train_id_to_color, 
    compose_predictions
)

from argparse import ArgumentParser

# --- UNet Building Blocks ---
class DoubleConv(nn.Module):
    """
    (convolution => [BN] => ReLU) * 2
    """
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
    """
    Downscaling with maxpool then double conv
    """
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """
    Upscaling then double conv
    """
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
    """
    1x1 convolution for the final output
    """
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

# --- MultiHead UNet Definition ---
class MultiHeadUNet(nn.Module):
    """
    A multi-head segmentation network with a shared UNet encoder and three separate decoder heads.
    - Small head predicts 8 classes.
    - Medium head predicts 5 classes.
    - Big head predicts 9 classes.
    
    Each decoder follows the standard UNet pattern:
      up1: Up(512+512, 256)
      up2: Up(256+256, 128)
      up3: Up(128+128, 64)
      up4: Up(64+64, 64)
    """
    def __init__(self, in_channels=3):
        super(MultiHeadUNet, self).__init__()
        # Shared encoder (same as UNet encoder)
        self.inc   = DoubleConv(in_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)  # Bottleneck

        # --- SMALL HEAD (8 classes) ---
        self.up1_small = Up(512 + 512, 256)   # x5 and x4: 512+512
        self.up2_small = Up(256 + 256, 128)     # x from up1_small (256) and x3 (256)
        self.up3_small = Up(128 + 128, 64)      # up2_small (128) and x2 (128)
        self.up4_small = Up(64 + 64, 64)        # up3_small (64) and x1 (64)
        self.outc_small = OutConv(64, 8)

        # --- MEDIUM HEAD (5 classes) ---
        self.up1_medium = Up(512 + 512, 256)
        self.up2_medium = Up(256 + 256, 128)
        self.up3_medium = Up(128 + 128, 64)
        self.up4_medium = Up(64 + 64, 64)
        self.outc_medium = OutConv(64, 5)

        # --- BIG HEAD (9 classes) ---
        self.up1_big = Up(512 + 512, 256)
        self.up2_big = Up(256 + 256, 128)
        self.up3_big = Up(128 + 128, 64)
        self.up4_big = Up(64 + 64, 64)
        self.outc_big = OutConv(64, 9)

    def forward(self, x):
        # Encoder
        x1 = self.inc(x)    # [B, 64, H, W]
        x2 = self.down1(x1) # [B, 128, H/2, W/2]
        x3 = self.down2(x2) # [B, 256, H/4, W/4]
        x4 = self.down3(x3) # [B, 512, H/8, W/8]
        x5 = self.down4(x4) # [B, 512, H/16, W/16]

        # SMALL HEAD decoder:
        x_small = self.up1_small(x5, x4)
        x_small = self.up2_small(x_small, x3)
        x_small = self.up3_small(x_small, x2)
        x_small = self.up4_small(x_small, x1)
        logits_small = self.outc_small(x_small)

        # MEDIUM HEAD decoder:
        x_medium = self.up1_medium(x5, x4)
        x_medium = self.up2_medium(x_medium, x3)
        x_medium = self.up3_medium(x_medium, x2)
        x_medium = self.up4_medium(x_medium, x1)
        logits_medium = self.outc_medium(x_medium)

        # BIG HEAD decoder:
        x_big = self.up1_big(x5, x4)
        x_big = self.up2_big(x_big, x3)
        x_big = self.up3_big(x_big, x2)
        x_big = self.up4_big(x_big, x1)
        logits_big = self.outc_big(x_big)

        return logits_small, logits_medium, logits_big

def get_args_parser():
    parser = ArgumentParser("Training script for multi-head UNet segmentation.")
    parser.add_argument("--data-dir", type=str, default="./data/cityscapes", help="Path to Cityscapes data")
    parser.add_argument("--batch-size", type=int, default=4, help="Training batch size")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.005, help="Learning rate")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader num_workers")
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

    transform = Compose([
        ToImage(),
        Resize((512, 1024)),  # Adjust resolution as needed
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

    # Instantiate the multi-head UNet model and move to device
    model = MultiHeadUNet(in_channels=3)
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr)

    # Define criterion for each head
    criterion_small_ce = nn.CrossEntropyLoss(ignore_index=255)
    criterion_medium_ce = nn.CrossEntropyLoss(ignore_index=255)
    criterion_big_ce = nn.CrossEntropyLoss(ignore_index=255)

    dice_weight = 1.0
    ce_weight = 1.0

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

        for images, labels in train_dataloader:
            images = images.to(device)
            labels_trainid = convert_to_train_id(labels)

            optimizer.zero_grad()
            # Forward pass through the multi-head network
            out_small, out_medium, out_big = model(images)

            # Prepare labels for each head
            labels_small = remap_small(labels_trainid).long().squeeze(1).to(device)
            labels_medium = remap_medium(labels_trainid).long().squeeze(1).to(device)
            labels_big = remap_big(labels_trainid).long().squeeze(1).to(device)

            # For dice loss targets: set ignore index to 0
            labels_small_dice = labels_small.clone()
            labels_small_dice[labels_small_dice == 255] = 0
            labels_medium_dice = labels_medium.clone()
            labels_medium_dice[labels_medium_dice == 255] = 0
            labels_big_dice = labels_big.clone()
            labels_big_dice[labels_big_dice == 255] = 0

            # Upsample outputs if needed to match label resolution
            out_small_upsampled = F.interpolate(out_small, size=labels_small.shape[-2:], mode='bilinear', align_corners=True)
            out_medium_upsampled = F.interpolate(out_medium, size=labels_medium.shape[-2:], mode='bilinear', align_corners=True)
            out_big_upsampled = F.interpolate(out_big, size=labels_big.shape[-2:], mode='bilinear', align_corners=True)

            # Create one-hot targets for dice loss
            num_classes_small = out_small_upsampled.shape[1]
            num_classes_medium = out_medium_upsampled.shape[1]
            num_classes_big = out_big_upsampled.shape[1]
            labels_small_one_hot = F.one_hot(labels_small_dice, num_classes_small).permute(0, 3, 1, 2).float()
            labels_medium_one_hot = F.one_hot(labels_medium_dice, num_classes_medium).permute(0, 3, 1, 2).float()
            labels_big_one_hot = F.one_hot(labels_big_dice, num_classes_big).permute(0, 3, 1, 2).float()

            # Compute losses for each head
            loss_small_dice = multiclass_dice_loss_small(out_small_upsampled, labels_small_one_hot)
            loss_small_ce   = criterion_small_ce(out_small_upsampled, labels_small)
            loss_small      = dice_weight * loss_small_dice + ce_weight * loss_small_ce

            loss_medium_dice = multiclass_dice_loss_medium(out_medium_upsampled, labels_medium_one_hot)
            loss_medium_ce   = criterion_medium_ce(out_medium_upsampled, labels_medium)
            loss_medium      = dice_weight * loss_medium_dice + ce_weight * loss_medium_ce

            loss_big_dice = multiclass_dice_loss_big(out_big_upsampled, labels_big_one_hot)
            loss_big_ce   = criterion_big_ce(out_big_upsampled, labels_big)
            loss_big      = dice_weight * loss_big_dice + ce_weight * loss_big_ce

            # Total loss is sum of all head losses
            loss_total = loss_small + loss_medium + loss_big
            loss_total.backward()
            optimizer.step()

            train_losses_small.append(loss_small.item())
            train_losses_small_dice.append(loss_small_dice.item())
            train_losses_small_ce.append(loss_small_ce.item())

            train_losses_medium.append(loss_medium.item())
            train_losses_medium_dice.append(loss_medium_dice.item())
            train_losses_medium_ce.append(loss_medium_ce.item())

            train_losses_big.append(loss_big.item())
            train_losses_big_dice.append(loss_big_dice.item())
            train_losses_big_ce.append(loss_big_ce.item())

        avg_loss_small = sum(train_losses_small) / len(train_losses_small)
        avg_loss_medium = sum(train_losses_medium) / len(train_losses_medium)
        avg_loss_big = sum(train_losses_big) / len(train_losses_big)

        avg_loss_small_dice = sum(train_losses_small_dice) / len(train_losses_small_dice)
        avg_loss_small_ce = sum(train_losses_small_ce) / len(train_losses_small_ce)
        avg_loss_medium_dice = sum(train_losses_medium_dice) / len(train_losses_medium_dice)
        avg_loss_medium_ce = sum(train_losses_medium_ce) / len(train_losses_medium_ce)
        avg_loss_big_dice = sum(train_losses_big_dice) / len(train_losses_big_dice)
        avg_loss_big_ce = sum(train_losses_big_ce) / len(train_losses_big_ce)

        # Validation
        model.eval()
        val_losses_small = []
        val_losses_medium = []
        val_losses_big = []

        val_losses_small_dice = []
        val_losses_medium_dice = []
        val_losses_big_dice = []
        val_losses_small_ce = []
        val_losses_medium_ce = []
        val_losses_big_ce = []

        # For confusion matrices: small (8 classes), medium (5 classes), big (9 classes) and composed (19 classes)
        num_classes_small = 8
        num_classes_medium = 5
        num_classes_big = 9
        num_classes_composed = 19

        conf_mat_small = torch.zeros(num_classes_small, num_classes_small, dtype=torch.int64)
        conf_mat_medium = torch.zeros(num_classes_medium, num_classes_medium, dtype=torch.int64)
        conf_mat_big = torch.zeros(num_classes_big, num_classes_big, dtype=torch.int64)
        conf_mat_composed = torch.zeros(num_classes_composed, num_classes_composed, dtype=torch.int64)

        logged_images = False

        with torch.no_grad():
            for i, (images, labels) in enumerate(valid_dataloader):
                images = images.to(device)
                labels_trainid = convert_to_train_id(labels)

                # --- SMALL HEAD ---
                labels_small = remap_small(labels_trainid).long().squeeze(1).to(device)
                labels_small_dice = labels_small.clone()
                labels_small_dice[labels_small_dice == 255] = 0
                labels_small_ce = labels_small.clone()

                out_small, out_medium, out_big = model(images)
                out_small_upsampled = F.interpolate(out_small, size=labels_small.shape[-2:], mode='bilinear', align_corners=True)
                # For small head, since it already outputs 8 classes, we directly compute predictions.
                pred_small = out_small_upsampled.softmax(dim=1).argmax(dim=1)
                # (If needed, you can remap predictions; here we assume remap_small is still applicable)
                pred_small = remap_small(pred_small)

                num_classes_small_val = out_small_upsampled.shape[1]
                labels_small_one_hot = F.one_hot(labels_small_dice, num_classes_small_val).permute(0, 3, 1, 2).float()
                val_small_dice = multiclass_dice_loss_small(out_small_upsampled, labels_small_one_hot)
                val_losses_small_dice.append(val_small_dice.item())
                val_small_ce = criterion_small_ce(out_small_upsampled, labels_small_ce)
                val_losses_small_ce.append(val_small_ce.item())
                val_loss_small = dice_weight * val_small_dice + ce_weight * val_small_ce
                val_losses_small.append(val_loss_small.item())
                conf_mat_small = update_confusion_matrix(conf_mat_small, pred_small, labels_small, ignore_index=255)

                # --- MEDIUM HEAD ---
                labels_medium = remap_medium(labels_trainid).long().squeeze(1).to(device)
                labels_medium_dice = labels_medium.clone()
                labels_medium_dice[labels_medium_dice == 255] = 0
                labels_medium_ce = labels_medium.clone()

                out_medium_upsampled = F.interpolate(out_medium, size=labels_medium.shape[-2:], mode='bilinear', align_corners=True)
                num_classes_medium_val = out_medium_upsampled.shape[1]
                labels_medium_one_hot = F.one_hot(labels_medium_dice, num_classes_medium_val).permute(0, 3, 1, 2).float()
                val_medium_dice = multiclass_dice_loss_medium(out_medium_upsampled, labels_medium_one_hot)
                val_losses_medium_dice.append(val_medium_dice.item())
                val_medium_ce = criterion_medium_ce(out_medium_upsampled, labels_medium_ce)
                val_losses_medium_ce.append(val_medium_ce.item())
                val_loss_medium = dice_weight * val_medium_dice + ce_weight * val_medium_ce
                val_losses_medium.append(val_loss_medium.item())
                pred_medium = out_medium_upsampled.softmax(dim=1).argmax(dim=1)
                conf_mat_medium = update_confusion_matrix(conf_mat_medium, pred_medium, labels_medium, ignore_index=255)

                # --- BIG HEAD ---
                labels_big = remap_big(labels_trainid).long().squeeze(1).to(device)
                labels_big_dice = labels_big.clone()
                labels_big_dice[labels_big_dice == 255] = 0
                labels_big_ce = labels_big.clone()

                out_big_upsampled = F.interpolate(out_big, size=labels_big.shape[-2:], mode='bilinear', align_corners=True)
                num_classes_big_val = out_big_upsampled.shape[1]
                labels_big_one_hot = F.one_hot(labels_big_dice, num_classes_big_val).permute(0, 3, 1, 2).float()
                val_big_dice = multiclass_dice_loss_big(out_big_upsampled, labels_big_one_hot)
                val_losses_big_dice.append(val_big_dice.item())
                val_big_ce = criterion_big_ce(out_big_upsampled, labels_big_ce)
                val_losses_big_ce.append(val_big_ce.item())
                val_loss_big = dice_weight * val_big_dice + ce_weight * val_big_ce
                val_losses_big.append(val_loss_big.item())
                pred_big = out_big_upsampled.softmax(dim=1).argmax(dim=1)
                conf_mat_big = update_confusion_matrix(conf_mat_big, pred_big, labels_big, ignore_index=255)

                # --- COMPOSED MODEL ---
                composed_pred = compose_predictions(
                    pred_small, pred_medium, pred_big,
                    bg_small=0, bg_medium=0, bg_big=0
                )
                ground_truth_full = labels_trainid.squeeze(1).to(device)
                conf_mat_composed = update_confusion_matrix(
                    conf_mat_composed, composed_pred, ground_truth_full, ignore_index=255
                )

                if not logged_images:
                    decoded_small = decode_label_small(pred_small)
                    decoded_medium = decode_label_medium(pred_medium)
                    decoded_big = decode_label_big(pred_big)

                    composed_pred_color = convert_train_id_to_color(composed_pred.unsqueeze(1))
                    composed_pred_img = make_grid(composed_pred_color.cpu(), nrow=4).permute(1, 2, 0).numpy()

                    pred_small_color = convert_train_id_to_color(decoded_small.unsqueeze(1))
                    pred_small_img = make_grid(pred_small_color.cpu(), nrow=4).permute(1, 2, 0).numpy()

                    pred_medium_color = convert_train_id_to_color(decoded_medium.unsqueeze(1))
                    pred_medium_img = make_grid(pred_medium_color.cpu(), nrow=4).permute(1, 2, 0).numpy()

                    pred_big_color = convert_train_id_to_color(decoded_big.unsqueeze(1))
                    pred_big_img = make_grid(pred_big_color.cpu(), nrow=4).permute(1, 2, 0).numpy()

                    gt_color = convert_train_id_to_color(labels_trainid)
                    gt_img = make_grid(gt_color.cpu(), nrow=4).permute(1, 2, 0).numpy()

                    if epoch % 5 == 0:
                        wandb.log({
                            "val_small_prediction": [wandb.Image(pred_small_img)],
                            "val_medium_prediction": [wandb.Image(pred_medium_img)],
                            "val_big_prediction": [wandb.Image(pred_big_img)],
                            "val_composed_prediction": [wandb.Image(composed_pred_img)],
                            "val_ground_truth": [wandb.Image(gt_img)],
                        })
                    logged_images = True

        avg_val_small = sum(val_losses_small) / len(val_losses_small) if val_losses_small else 0
        avg_val_medium = sum(val_losses_medium) / len(val_losses_medium) if val_losses_medium else 0
        avg_val_big = sum(val_losses_big) / len(val_losses_big) if val_losses_big else 0

        avg_val_small_dice = sum(val_losses_small_dice) / len(val_losses_small_dice) if val_losses_small_dice else 0
        avg_val_small_ce = sum(val_losses_small_ce) / len(val_losses_small_ce) if val_losses_small_ce else 0
        avg_val_medium_dice = sum(val_losses_medium_dice) / len(val_losses_medium_dice) if val_losses_medium_dice else 0
        avg_val_medium_ce = sum(val_losses_medium_ce) / len(val_losses_medium_ce) if val_losses_medium_ce else 0
        avg_val_big_dice = sum(val_losses_big_dice) / len(val_losses_big_dice) if val_losses_big_dice else 0
        avg_val_big_ce = sum(val_losses_big_ce) / len(val_losses_big_ce) if val_losses_big_ce else 0

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
            "epoch": epoch + 1,
        })

        # Save overall best model checkpoint
        if val_loss < best_val_loss_overall:
            best_val_loss_overall = val_loss
            best_checkpoint_path = os.path.join("checkpoints", "best_model_overall.pth")
            torch.save(model.state_dict(), best_checkpoint_path)
            print(f"New best OVERALL model saved at epoch {epoch+1} with avg_val_loss={val_loss:.4f} -> {best_checkpoint_path}")

    wandb.finish()
    print("Training complete.")

if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)
