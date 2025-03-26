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

class Model(nn.Module):
    """
    A single PyTorch model that contains the small, medium, and big submodels.
    During forward, it returns all three logits. You can also compose them externally.
    """
    def __init__(self):
        super().__init__()
        # Instantiate submodels on CPU by default; we will move them to GPU later in main().
        self.model_small = get_small_model(device="cpu")   # 8 output channels
        self.model_medium = get_medium_model(device="cpu") # 5 output channels
        self.model_big = get_big_model(device="cpu")       # 9 output channels

    def forward(self, x):
        out_small = self.model_small(x)
        out_medium = self.model_medium(x)
        out_big = self.model_big(x)
        return out_small, out_medium, out_big

def get_args_parser():
    parser = ArgumentParser("Training script for multiple models (small/medium/big).")
    parser.add_argument("--data-dir", type=str, default="./data/cityscapes", help="Path to Cityscapes data")
    parser.add_argument("--batch-size", type=int, default=4, help="Training batch size")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.005, help="Learning rate")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader num_workers")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--experiment-id", type=str, default="unet-training", help="Experiment ID for Weights & Biases")
    return parser

def update_confusion_matrix(conf_mat: torch.Tensor, 
                            pred: torch.Tensor, 
                            target: torch.Tensor, 
                            ignore_index=255) -> torch.Tensor:
    """
    Updates the provided confusion matrix (conf_mat) given the current batch of
    predictions (pred) and ground-truth labels (target).
    """
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
    # Check dataset path
    print(f"Checking dataset path: {args.data_dir}")
    if os.path.exists(args.data_dir):
        print("Contents:", os.listdir(args.data_dir))
    else:
        print("Data directory not found!")
        return

    # Set seed
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Initialize W&B for overall experiment
    wandb.finish()
    wandb.init(
        project="5lsm0-cityscapes-segmentation",
        name=args.experiment_id,
        config=vars(args),
        reinit=True,
    )

    # Define transforms
    transform = Compose([
        ToImage(),
        Resize((16, 16)),  # You can adjust the resolution as needed
        ToDtype(torch.float32, scale=True),
        Normalize((0.5,), (0.5,)),
    ])

    # Datasets and dataloaders
    train_dataset = Cityscapes(args.data_dir, split="train", mode="fine", target_type="semantic", transforms=transform)
    valid_dataset = Cityscapes(args.data_dir, split="val", mode="fine", target_type="semantic", transforms=transform)
    train_dataset = wrap_dataset_for_transforms_v2(train_dataset)
    valid_dataset = wrap_dataset_for_transforms_v2(valid_dataset)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # Create checkpoint directory
    os.makedirs("checkpoints", exist_ok=True)

    model = Model()
    model.to(device)

    # For convenience, separate references to each submodel
    model_small = model.model_small
    model_medium = model.model_medium
    model_big = model.model_big

    # Define optimizers for each submodel
    optimizer_small = AdamW(model_small.parameters(), lr=args.lr)
    optimizer_medium = AdamW(model_medium.parameters(), lr=args.lr)
    optimizer_big = AdamW(model_big.parameters(), lr=args.lr)

    # -------------------------------------------------------------------------
    # NEW: Define Cross Entropy for each submodel (ignore_index=255).
    #      We will combine it with Dice loss in training.
    # -------------------------------------------------------------------------
    criterion_small_ce = nn.CrossEntropyLoss(ignore_index=255)
    criterion_medium_ce = nn.CrossEntropyLoss(ignore_index=255)
    criterion_big_ce = nn.CrossEntropyLoss(ignore_index=255)

    # You can tune these weights to give more or less importance to each loss
    dice_weight = 1.0
    ce_weight = 1.0
    # -------------------------------------------------------------------------

    # Track separate best losses for each submodel
    best_val_loss_small = float("inf")
    best_val_loss_medium = float("inf")
    best_val_loss_big = float("inf")

    # Also track best combined loss
    best_val_loss_overall = float("inf")

    for epoch in range(args.epochs):
        print(f"Epoch [{epoch+1}/{args.epochs}]")

        model_small.train()
        model_medium.train()
        model_big.train()

        train_losses_small = []
        train_losses_medium = []
        train_losses_big = []

        # (Optional) Track separate dice/ce if desired:
        train_losses_small_dice = []
        train_losses_medium_dice = []
        train_losses_big_dice = []
        train_losses_small_ce = []
        train_losses_medium_ce = []
        train_losses_big_ce = []

        # ------------------ TRAINING LOOP ------------------
        for images, labels in train_dataloader:
            images = images.to(device)
            labels_trainid = convert_to_train_id(labels)

            # ---- Small model ----
            labels_small = remap_small(labels_trainid).long().squeeze(1).to(device)

            # For Dice: we replace 255 with 0 (i.e. treat ignore as background).
            labels_small_dice = labels_small.clone()
            labels_small_dice[labels_small_dice == 255] = 0

            # For CE: we keep 255 so ignore_index can do its job.
            labels_small_ce = labels_small.clone()  # shape [B,H,W], still has 255

            optimizer_small.zero_grad()
            out_small = model_small(images)  # [B,8,H,W]
            out_small_upsampled = F.interpolate(
                out_small,
                size=labels_small.shape[-2:],  # match label's spatial size
                mode='bilinear',
                align_corners=True
            )

            # 1) Dice Loss
            num_classes_small = out_small_upsampled.shape[1]
            labels_small_one_hot = F.one_hot(labels_small_dice, num_classes_small).permute(0, 3, 1, 2).float()
            loss_small_dice = multiclass_dice_loss_small(out_small_upsampled, labels_small_one_hot)

            # -----------------------------------------------------------------
            # NEW: 2) Cross Entropy Loss
            # -----------------------------------------------------------------
            loss_small_ce = criterion_small_ce(out_small_upsampled, labels_small_ce)

            # Combined final loss
            loss_small = dice_weight * loss_small_dice + ce_weight * loss_small_ce
            loss_small.backward()
            optimizer_small.step()

            train_losses_small.append(loss_small.item())
            train_losses_small_dice.append(loss_small_dice.item())
            train_losses_small_ce.append(loss_small_ce.item())

            # ---- Medium model ----
            labels_medium = remap_medium(labels_trainid).long().squeeze(1).to(device)
            labels_medium_dice = labels_medium.clone()
            labels_medium_dice[labels_medium_dice == 255] = 0
            labels_medium_ce = labels_medium.clone()

            optimizer_medium.zero_grad()
            out_medium = model_medium(images)  # [B,5,H,W]
            out_medium_upsampled = F.interpolate(
                out_medium,
                size=labels_medium.shape[-2:],
                mode='bilinear',
                align_corners=True
            )

            num_classes_medium = out_medium_upsampled.shape[1]
            labels_medium_one_hot = F.one_hot(labels_medium_dice, num_classes_medium).permute(0, 3, 1, 2).float()
            loss_medium_dice = multiclass_dice_loss_medium(out_medium_upsampled, labels_medium_one_hot)

            # NEW: cross entropy
            loss_medium_ce = criterion_medium_ce(out_medium_upsampled, labels_medium_ce)

            # Combined
            loss_medium = dice_weight * loss_medium_dice + ce_weight * loss_medium_ce
            loss_medium.backward()
            optimizer_medium.step()

            train_losses_medium.append(loss_medium.item())
            train_losses_medium_dice.append(loss_medium_dice.item())
            train_losses_medium_ce.append(loss_medium_ce.item())

            # ---- Big model ----
            labels_big = remap_big(labels_trainid).long().squeeze(1).to(device)
            labels_big_dice = labels_big.clone()
            labels_big_dice[labels_big_dice == 255] = 0
            labels_big_ce = labels_big.clone()

            optimizer_big.zero_grad()
            out_big = model_big(images)  # [B,9,H,W]
            out_big_upsampled = F.interpolate(
                out_big,
                size=labels_big.shape[-2:],
                mode='bilinear',
                align_corners=True
            )

            num_classes_big = out_big_upsampled.shape[1]
            labels_big_one_hot = F.one_hot(labels_big_dice, num_classes_big).permute(0, 3, 1, 2).float()
            loss_big_dice = multiclass_dice_loss_big(out_big_upsampled, labels_big_one_hot)

            # NEW: cross entropy
            loss_big_ce = criterion_big_ce(out_big_upsampled, labels_big_ce)

            # Combined
            loss_big = dice_weight * loss_big_dice + ce_weight * loss_big_ce
            loss_big.backward()
            optimizer_big.step()

            train_losses_big.append(loss_big.item())
            train_losses_big_dice.append(loss_big_dice.item())
            train_losses_big_ce.append(loss_big_ce.item())

        # Average training losses (combined)
        avg_loss_small = sum(train_losses_small) / len(train_losses_small)
        avg_loss_medium = sum(train_losses_medium) / len(train_losses_medium)
        avg_loss_big = sum(train_losses_big) / len(train_losses_big)

        # Also get average separate Dice/CE if you want to log them
        avg_loss_small_dice = sum(train_losses_small_dice)/len(train_losses_small_dice)
        avg_loss_small_ce = sum(train_losses_small_ce)/len(train_losses_small_ce)

        avg_loss_medium_dice = sum(train_losses_medium_dice)/len(train_losses_medium_dice)
        avg_loss_medium_ce = sum(train_losses_medium_ce)/len(train_losses_medium_ce)

        avg_loss_big_dice = sum(train_losses_big_dice)/len(train_losses_big_dice)
        avg_loss_big_ce = sum(train_losses_big_ce)/len(train_losses_big_ce)

        # ------------------ VALIDATION LOOP ------------------
        model_small.eval()
        model_medium.eval()
        model_big.eval()

        val_losses_small = []
        val_losses_medium = []
        val_losses_big = []

        # (Optional) track val dice/CE separately
        val_losses_small_dice = []
        val_losses_medium_dice = []
        val_losses_big_dice = []
        val_losses_small_ce = []
        val_losses_medium_ce = []
        val_losses_big_ce = []

        # For confusion matrices
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
                labels_trainid = convert_to_train_id(labels)  # [B,1,H,W]

                # --------- SMALL MODEL ---------
                labels_small = remap_small(labels_trainid).long().squeeze(1).to(device)

                labels_small_dice = labels_small.clone()
                labels_small_dice[labels_small_dice == 255] = 0
                labels_small_ce = labels_small.clone()  # keep 255

                out_small = model_small(images)
                out_small_upsampled = F.interpolate(
                    out_small,
                    size=labels_small.shape[-2:],
                    mode='bilinear',
                    align_corners=True
                )

                # Dice
                num_classes_small = out_small_upsampled.shape[1]
                labels_small_one_hot = F.one_hot(labels_small_dice, num_classes_small).permute(0, 3, 1, 2).float()
                val_small_dice = multiclass_dice_loss_small(out_small_upsampled, labels_small_one_hot)
                val_losses_small_dice.append(val_small_dice.item())

                # CE
                val_small_ce = criterion_small_ce(out_small_upsampled, labels_small_ce)
                val_losses_small_ce.append(val_small_ce.item())

                # Combine
                val_loss_small = dice_weight * val_small_dice + ce_weight * val_small_ce
                val_losses_small.append(val_loss_small.item())

                # Confusion matrix
                pred_small = out_small_upsampled.softmax(dim=1).argmax(dim=1)
                conf_mat_small = update_confusion_matrix(conf_mat_small, pred_small, labels_small, ignore_index=255)

                # --------- MEDIUM MODEL ---------
                labels_medium = remap_medium(labels_trainid).long().squeeze(1).to(device)
                labels_medium_dice = labels_medium.clone()
                labels_medium_dice[labels_medium_dice == 255] = 0
                labels_medium_ce = labels_medium.clone()

                out_medium = model_medium(images)
                out_medium_upsampled = F.interpolate(
                    out_medium,
                    size=labels_medium.shape[-2:],
                    mode='bilinear',
                    align_corners=True
                )

                num_classes_medium = out_medium_upsampled.shape[1]
                labels_medium_one_hot = F.one_hot(labels_medium_dice, num_classes_medium).permute(0, 3, 1, 2).float()
                val_medium_dice = multiclass_dice_loss_medium(out_medium_upsampled, labels_medium_one_hot)
                val_losses_medium_dice.append(val_medium_dice.item())

                val_medium_ce = criterion_medium_ce(out_medium_upsampled, labels_medium_ce)
                val_losses_medium_ce.append(val_medium_ce.item())

                val_loss_medium = dice_weight * val_medium_dice + ce_weight * val_medium_ce
                val_losses_medium.append(val_loss_medium.item())

                pred_medium = out_medium_upsampled.softmax(dim=1).argmax(dim=1)
                conf_mat_medium = update_confusion_matrix(conf_mat_medium, pred_medium, labels_medium, ignore_index=255)

                # --------- BIG MODEL ---------
                labels_big = remap_big(labels_trainid).long().squeeze(1).to(device)
                labels_big_dice = labels_big.clone()
                labels_big_dice[labels_big_dice == 255] = 0
                labels_big_ce = labels_big.clone()

                out_big = model_big(images)
                out_big_upsampled = F.interpolate(
                    out_big,
                    size=labels_big.shape[-2:],
                    mode='bilinear',
                    align_corners=True
                )

                num_classes_big = out_big_upsampled.shape[1]
                labels_big_one_hot = F.one_hot(labels_big_dice, num_classes_big).permute(0, 3, 1, 2).float()
                val_big_dice = multiclass_dice_loss_big(out_big_upsampled, labels_big_one_hot)
                val_losses_big_dice.append(val_big_dice.item())

                val_big_ce = criterion_big_ce(out_big_upsampled, labels_big_ce)
                val_losses_big_ce.append(val_big_ce.item())

                val_loss_big = dice_weight * val_big_dice + ce_weight * val_big_ce
                val_losses_big.append(val_loss_big.item())

                pred_big = out_big_upsampled.softmax(dim=1).argmax(dim=1)
                conf_mat_big = update_confusion_matrix(conf_mat_big, pred_big, labels_big, ignore_index=255)

                # --------- COMPOSED MODEL ---------
                composed_pred = compose_predictions(
                    pred_small, pred_medium, pred_big,
                    bg_small=0, bg_medium=0, bg_big=0
                )
                ground_truth_full = labels_trainid.squeeze(1).to(device)
                conf_mat_composed = update_confusion_matrix(
                    conf_mat_composed, composed_pred, ground_truth_full, ignore_index=255
                )

                # Log sample predictions (once per epoch) to W&B
                if not logged_images:
                    decoded_small = decode_label_small(pred_small)
                    decoded_medium = decode_label_medium(pred_medium)
                    decoded_big = decode_label_big(pred_big)

                    from utils import convert_train_id_to_color
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

                    # Log images every 5 epochs, for instance
                    if epoch % 5 == 0:
                        wandb.log({
                            "val_small_prediction": [wandb.Image(pred_small_img)],
                            "val_medium_prediction": [wandb.Image(pred_medium_img)],
                            "val_big_prediction": [wandb.Image(pred_big_img)],
                            "val_composed_prediction": [wandb.Image(composed_pred_img)],
                            "val_ground_truth": [wandb.Image(gt_img)],
                        })
                    logged_images = True

        # Compute validation losses (combined)
        avg_val_small = sum(val_losses_small) / len(val_losses_small) if val_losses_small else 0
        avg_val_medium = sum(val_losses_medium) / len(val_losses_medium) if val_losses_medium else 0
        avg_val_big = sum(val_losses_big) / len(val_losses_big) if val_losses_big else 0

        # (Optional) separate average Dice/CE
        avg_val_small_dice = sum(val_losses_small_dice)/len(val_losses_small_dice) if val_losses_small_dice else 0
        avg_val_small_ce = sum(val_losses_small_ce)/len(val_losses_small_ce) if val_losses_small_ce else 0

        avg_val_medium_dice = sum(val_losses_medium_dice)/len(val_losses_medium_dice) if val_losses_medium_dice else 0
        avg_val_medium_ce = sum(val_losses_medium_ce)/len(val_losses_medium_ce) if val_losses_medium_ce else 0

        avg_val_big_dice = sum(val_losses_big_dice)/len(val_losses_big_dice) if val_losses_big_dice else 0
        avg_val_big_ce = sum(val_losses_big_ce)/len(val_losses_big_ce) if val_losses_big_ce else 0

        # Combined val loss
        val_loss = (avg_val_small + avg_val_medium + avg_val_big) / 3.0

        # Compute mIoU
        miou_small = compute_miou(conf_mat_small)
        miou_medium = compute_miou(conf_mat_medium)
        miou_big = compute_miou(conf_mat_big)
        miou_composed = compute_miou(conf_mat_composed)

        # Log metrics
        wandb.log({
            # Training combined
            "train_loss_small": avg_loss_small,
            "train_loss_medium": avg_loss_medium,
            "train_loss_big": avg_loss_big,

            # Training separate dice/CE
            "train_loss_small_dice": avg_loss_small_dice,
            "train_loss_small_ce": avg_loss_small_ce,
            "train_loss_medium_dice": avg_loss_medium_dice,
            "train_loss_medium_ce": avg_loss_medium_ce,
            "train_loss_big_dice": avg_loss_big_dice,
            "train_loss_big_ce": avg_loss_big_ce,

            # Validation combined
            "val_loss": val_loss,
            "val_loss_small": avg_val_small,
            "val_loss_medium": avg_val_medium,
            "val_loss_big": avg_val_big,

            # Validation separate dice/CE
            "val_loss_small_dice": avg_val_small_dice,
            "val_loss_small_ce": avg_val_small_ce,
            "val_loss_medium_dice": avg_val_medium_dice,
            "val_loss_medium_ce": avg_val_medium_ce,
            "val_loss_big_dice": avg_val_big_dice,
            "val_loss_big_ce": avg_val_big_ce,

            # mIoU
            "val_mIoU_small": miou_small,
            "val_mIoU_medium": miou_medium,
            "val_mIoU_big": miou_big,
            "val_mIoU_composed": miou_composed,
            "epoch": epoch + 1,
        })

        # ------------------ SAVE BEST CHECKPOINTS ------------------
        # Check if the SMALL model improved
        if avg_val_small < best_val_loss_small:
            best_val_loss_small = avg_val_small
            checkpoint_path = os.path.join("checkpoints", f"best_model_small_epoch{epoch+1}.pth")
            torch.save(model_small.state_dict(), checkpoint_path)
            print(f"New best SMALL model saved at epoch {epoch+1} with val_loss_small={avg_val_small:.4f} -> {checkpoint_path}")

        # Check if the MEDIUM model improved
        if avg_val_medium < best_val_loss_medium:
            best_val_loss_medium = avg_val_medium
            checkpoint_path = os.path.join("checkpoints", f"best_model_medium_epoch{epoch+1}.pth")
            torch.save(model_medium.state_dict(), checkpoint_path)
            print(f"New best MEDIUM model saved at epoch {epoch+1} with val_loss_medium={avg_val_medium:.4f} -> {checkpoint_path}")

        # Check if the BIG model improved
        if avg_val_big < best_val_loss_big:
            best_val_loss_big = avg_val_big
            checkpoint_path = os.path.join("checkpoints", f"best_model_big_epoch{epoch+1}.pth")
            torch.save(model_big.state_dict(), checkpoint_path)
            print(f"New best BIG model saved at epoch {epoch+1} with val_loss_big={avg_val_big:.4f} -> {checkpoint_path}")

        # (Optional) Also save a single “best overall” combined model
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
