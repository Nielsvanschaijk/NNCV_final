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
from torchvision.transforms.v2 import InterpolationMode
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

from utils import convert_to_train_id, convert_train_id_to_color, compose_predictions

from argparse import ArgumentParser

class EnsembleModel(nn.Module):
    """
    A single PyTorch model that contains the small, medium, and big submodels.
    During forward, it returns all three logits (or you can do composition).
    """
    def __init__(self, device='cpu'):
        super().__init__()
        # Instantiate each submodel on CPU or GPU as needed
        self.model_small = get_small_model(device=device)
        self.model_medium = get_medium_model(device=device)
        self.model_big = get_big_model(device=device)

    def forward(self, x):
        # We simply forward through each submodel and return all three results
        out_small = self.model_small(x)
        out_medium = self.model_medium(x)
        out_big = self.model_big(x)
        return out_small, out_medium, out_big


def get_args_parser():
    parser = ArgumentParser("Training script for multiple models (small/medium/big).")
    parser.add_argument("--data-dir", type=str, default="./data/cityscapes", help="Path to Cityscapes data")
    parser.add_argument("--batch-size", type=int, default=1, help="Training batch size")
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

    Args:
        conf_mat:       shape [num_classes, num_classes], int64
        pred:           shape [B,H,W], predictions in [0..num_classes-1]
        target:         shape [B,H,W], ground-truth in [0..num_classes-1 or ignore_index]
        ignore_index:   label to ignore from evaluation (e.g. 255)

    Returns:
        Updated conf_mat in-place (also returned for convenience).
        NOTE: The returned `conf_mat` is now on the same device as `pred`.
    """
    # Make sure conf_mat is on the same device as pred
    if conf_mat.device != pred.device:
        conf_mat = conf_mat.to(pred.device)

    # Flatten for simplicity
    pred = pred.view(-1)
    target = target.view(-1)

    # Filter out ignore_index
    valid_mask = (target != ignore_index)
    pred = pred[valid_mask]
    target = target[valid_mask]

    # Now accumulate counts into confusion matrix
    indices = target * conf_mat.shape[0] + pred
    conf_mat += torch.bincount(indices, minlength=conf_mat.numel()).view(conf_mat.size())

    return conf_mat


def compute_miou(conf_mat: torch.Tensor) -> float:
    """
    Computes the mean IoU from the provided confusion matrix.
    Ignores any class that has 0 pixels total.
    """
    # The diagonal is intersection
    intersection = torch.diag(conf_mat)
    # For each row (class), the row sum is the total GT for that class
    # For each column (class), the column sum is the total predictions
    gt_plus_pred = conf_mat.sum(dim=1) + conf_mat.sum(dim=0) - intersection  # denominator for IoU

    # Avoid division by zero
    valid_mask = (gt_plus_pred > 0)
    iou_per_class = torch.zeros_like(intersection, dtype=torch.float)
    iou_per_class[valid_mask] = intersection[valid_mask].float() / gt_plus_pred[valid_mask].float()

    # Mean over valid classes
    return iou_per_class.mean().item()

def main(args):
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

    # Transforms
    transform = Compose([
        ToImage(),
        Resize((512, 512)),
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

    # Initialize models
    model_small = get_small_model(device)    # 8 output channels
    model_medium = get_medium_model(device)  # 5 output channels
    model_big = get_big_model(device)        # 9 output channels

    # Define optimizers
    optimizer_small = AdamW(model_small.parameters(), lr=args.lr)
    optimizer_medium = AdamW(model_medium.parameters(), lr=args.lr)
    optimizer_big = AdamW(model_big.parameters(), lr=args.lr)

    # Loss function (ignore_index=255 to skip originally ignored pixels)
    criterion = nn.CrossEntropyLoss(ignore_index=255)

    best_val_loss = float("inf")
    best_checkpoint_path = None

    for epoch in range(args.epochs):
        print(f"Epoch [{epoch+1}/{args.epochs}]")

        model_small.train()
        model_medium.train()
        model_big.train()

        train_losses_small = []
        train_losses_medium = []
        train_losses_big = []

        # ------------------ TRAINING LOOP ------------------
        for images, labels in train_dataloader:
            images = images.to(device)
            # Convert official IDs to train IDs, then remap to small/medium/big
            labels_trainid = convert_to_train_id(labels)

            # ---- Small model ----
            labels_small = remap_small(labels_trainid).long().squeeze(1).to(device)
            optimizer_small.zero_grad()
            out_small = model_small(images)  # shape [B,8,H,W]
            out_small_upsampled = F.interpolate(
                out_small,
                size=labels_small.shape[-2:],  # match label resolution
                mode='bilinear',
                align_corners=True
            )
            loss_small = criterion(out_small_upsampled, labels_small)
            loss_small.backward()
            optimizer_small.step()
            train_losses_small.append(loss_small.item())

            # ---- Medium model ----
            labels_medium = remap_medium(labels_trainid).long().squeeze(1).to(device)
            optimizer_medium.zero_grad()
            out_medium = model_medium(images)  # shape [B,5,H,W]
            out_medium_upsampled = F.interpolate(
                out_medium,
                size=labels_medium.shape[-2:],
                mode='bilinear',
                align_corners=True
            )
            loss_medium = criterion(out_medium_upsampled, labels_medium)
            loss_medium.backward()
            optimizer_medium.step()
            train_losses_medium.append(loss_medium.item())

            # ---- Big model ----
            labels_big = remap_big(labels_trainid).long().squeeze(1).to(device)
            optimizer_big.zero_grad()
            out_big = model_big(images)  # shape [B,9,H,W]
            out_big_upsampled = F.interpolate(
                out_big,
                size=labels_big.shape[-2:],
                mode='bilinear',
                align_corners=True
            )
            loss_big = criterion(out_big_upsampled, labels_big)
            loss_big.backward()
            optimizer_big.step()
            train_losses_big.append(loss_big.item())

        # Compute average training losses
        avg_loss_small = sum(train_losses_small) / len(train_losses_small)
        avg_loss_medium = sum(train_losses_medium) / len(train_losses_medium)
        avg_loss_big = sum(train_losses_big) / len(train_losses_big)

        # ------------------ VALIDATION LOOP ------------------
        model_small.eval()
        model_medium.eval()
        model_big.eval()

        val_losses_small = []
        val_losses_medium = []
        val_losses_big = []

        # For confusion matrices
        num_classes_small = 8   # sub-model classes
        num_classes_medium = 5
        num_classes_big = 9
        num_classes_composed = 19  # if compose outputs [0..18]
        conf_mat_small = torch.zeros(num_classes_small, num_classes_small, dtype=torch.int64)
        conf_mat_medium = torch.zeros(num_classes_medium, num_classes_medium, dtype=torch.int64)
        conf_mat_big = torch.zeros(num_classes_big, num_classes_big, dtype=torch.int64)
        conf_mat_composed = torch.zeros(num_classes_composed, num_classes_composed, dtype=torch.int64)

        logged_images = False

        with torch.no_grad():
            for i, (images, labels) in enumerate(valid_dataloader):
                images = images.to(device)

                # Convert official IDs to train IDs
                labels_trainid = convert_to_train_id(labels)  # shape [B,1,H,W]

                # --------- SMALL MODEL ---------
                labels_small = remap_small(labels_trainid).long().squeeze(1).to(device)
                out_small = model_small(images)
                out_small_upsampled = F.interpolate(
                    out_small,
                    size=labels_small.shape[-2:],
                    mode='bilinear',
                    align_corners=True
                )
                loss_small = criterion(out_small_upsampled, labels_small)
                val_losses_small.append(loss_small.item())
                pred_small = out_small_upsampled.softmax(dim=1).argmax(dim=1)  # [B,H,W]
                conf_mat_small = update_confusion_matrix(conf_mat_small, pred_small, labels_small, ignore_index=255)

                # --------- MEDIUM MODEL ---------
                labels_medium = remap_medium(labels_trainid).long().squeeze(1).to(device)
                out_medium = model_medium(images)
                out_medium_upsampled = F.interpolate(
                    out_medium,
                    size=labels_medium.shape[-2:],
                    mode='bilinear',
                    align_corners=True
                )
                loss_medium = criterion(out_medium_upsampled, labels_medium)
                val_losses_medium.append(loss_medium.item())
                pred_medium = out_medium_upsampled.softmax(dim=1).argmax(dim=1)
                conf_mat_medium = update_confusion_matrix(conf_mat_medium, pred_medium, labels_medium, ignore_index=255)

                # --------- BIG MODEL ---------
                labels_big = remap_big(labels_trainid).long().squeeze(1).to(device)
                out_big = model_big(images)
                out_big_upsampled = F.interpolate(
                    out_big,
                    size=labels_big.shape[-2:],
                    mode='bilinear',
                    align_corners=True
                )
                loss_big = criterion(out_big_upsampled, labels_big)
                val_losses_big.append(loss_big.item())
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

                # --------- LOG IMAGES (ONCE PER EPOCH) ---------
                if not logged_images:
                    # 1) Decode sub-model predictions to Cityscapes IDs
                    decoded_small = decode_label_small(pred_small)    # shape [B,H,W], 0..18
                    decoded_medium = decode_label_medium(pred_medium) # shape [B,H,W], 0..18
                    decoded_big = decode_label_big(pred_big)          # shape [B,H,W], 0..18

                    # If compose_predictions already yields 0..18, we can color it directly
                    composed_pred_color = convert_train_id_to_color(composed_pred.unsqueeze(1))
                    composed_pred_img = make_grid(composed_pred_color.cpu(), nrow=4).permute(1,2,0).numpy()

                    # 2) Convert each decoded mask to color
                    pred_small_color = convert_train_id_to_color(decoded_small.unsqueeze(1))
                    pred_small_img = make_grid(pred_small_color.cpu(), nrow=4).permute(1, 2, 0).numpy()

                    pred_medium_color = convert_train_id_to_color(decoded_medium.unsqueeze(1))
                    pred_medium_img = make_grid(pred_medium_color.cpu(), nrow=4).permute(1, 2, 0).numpy()

                    pred_big_color = convert_train_id_to_color(decoded_big.unsqueeze(1))
                    pred_big_img = make_grid(pred_big_color.cpu(), nrow=4).permute(1, 2, 0).numpy()

                    # 3) Convert ground truth (train IDs) to color
                    gt_color = convert_train_id_to_color(labels_trainid)  # [B,3,H,W]
                    gt_img = make_grid(gt_color.cpu(), nrow=4).permute(1, 2, 0).numpy()

                    # Log to wandb
                    if epoch % 5 == 0:
                        wandb.log({
                            "val_small_prediction": [wandb.Image(pred_small_img)],
                            "val_medium_prediction": [wandb.Image(pred_medium_img)],
                            "val_big_prediction": [wandb.Image(pred_big_img)],
                            "val_composed_prediction": [wandb.Image(composed_pred_img)],
                            "val_ground_truth": [wandb.Image(gt_img)]
                        })

                    logged_images = True
                # --------- END validation batch loop ---------

        # Averages
        avg_val_small = sum(val_losses_small) / len(val_losses_small) if val_losses_small else 0
        avg_val_medium = sum(val_losses_medium) / len(val_losses_medium) if val_losses_medium else 0
        avg_val_big = sum(val_losses_big) / len(val_losses_big) if val_losses_big else 0
        val_loss = (avg_val_small + avg_val_medium + avg_val_big) / 3.0

        # Compute mIoU in sub-model space
        miou_small = compute_miou(conf_mat_small)
        miou_medium = compute_miou(conf_mat_medium)
        miou_big = compute_miou(conf_mat_big)
        miou_composed = compute_miou(conf_mat_composed)

        # Log metrics
        wandb.log({
            "train_loss_small": avg_loss_small,
            "train_loss_medium": avg_loss_medium,
            "train_loss_big": avg_loss_big,
            "val_loss": val_loss,
            "val_mIoU_small": miou_small,
            "val_mIoU_medium": miou_medium,
            "val_mIoU_big": miou_big,
            "val_mIoU_composed": miou_composed,
            "epoch": epoch + 1,
        })

        # Save best model checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_name = f"best_models_epoch={epoch+1}_val={val_loss:.4f}.pth"
            best_checkpoint_path = os.path.join("checkpoints", checkpoint_name)

            torch.save({
                "model_small": model_small.state_dict(),
                "model_medium": model_medium.state_dict(),
                "model_big": model_big.state_dict(),
            }, best_checkpoint_path)

            print(f"New best model saved at epoch {epoch+1} with val_loss {val_loss:.4f}")
            
            # 3) Build single-file ensemble checkpoint
            print("-> Building single-file ensemble checkpoint...")
            ensemble = EnsembleModel(device="cpu")

            # Load each submodel’s weights
            checkpoint_dict = torch.load(best_checkpoint_path, map_location="cpu")
            ensemble.model_small.load_state_dict(checkpoint_dict["model_small"])
            ensemble.model_medium.load_state_dict(checkpoint_dict["model_medium"])
            ensemble.model_big.load_state_dict(checkpoint_dict["model_big"])

            # Save entire ensemble’s state_dict as ONE file
            ensemble_path = os.path.join("checkpoints", f"best_ensemble_epoch={epoch+1}_val={val_loss:.4f}.pth")
            torch.save(ensemble.state_dict(), ensemble_path)

            print(f"-> Single-file ensemble saved: {ensemble_path}")

    wandb.finish()
    print("Training complete.")


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)
