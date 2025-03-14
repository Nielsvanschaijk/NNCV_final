import os
from argparse import ArgumentParser

import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchvision.datasets import Cityscapes
from torchvision.utils import make_grid
from torchvision.transforms.v2 import (
    Compose,
    Normalize,
    Resize,
    ToImage,
    ToDtype
)
from torchvision.transforms.functional import InterpolationMode
from unet import UNet

def remap_label_small(mask, background_label=0):
    new_mask = torch.full_like(mask, fill_value=background_label)
    small_classes = [5, 6, 7, 11, 12, 17, 18]
    for c in small_classes:
        new_mask[mask == c] = c
    return new_mask

def remap_label_medium(mask, ignore_index=255):
    new_mask = torch.full_like(mask, fill_value=ignore_index)
    medium_classes = [1, 3, 4, 13]
    for c in medium_classes:
        new_mask[mask == c] = c
    new_mask[mask == 255] = ignore_index
    return new_mask

def remap_label_big(mask, ignore_index=255):
    new_mask = torch.full_like(mask, fill_value=ignore_index)
    big_classes = [0, 2, 8, 9, 10, 14, 15, 16]
    for c in big_classes:
        new_mask[mask == c] = c
    new_mask[mask == 255] = ignore_index
    return new_mask

# Image transform: convert to image, resize, convert to float and normalize
image_transform = Compose([
    ToImage(),
    Resize((256, 256)),
    ToDtype(torch.float32, scale=True),
])

# Target transform: convert to image and resize without normalization/scaling
target_transform = Compose([
    ToImage(),
    Resize((256, 256), interpolation=InterpolationMode.NEAREST),
    ToDtype(torch.int64),
])

# Convert raw Cityscapes "ID" labels to "train IDs"
id_to_trainid = {cls.id: cls.train_id for cls in Cityscapes.classes}
def convert_to_train_id(label_img: torch.Tensor) -> torch.Tensor:
    label_img = label_img.to(torch.int64)
    return label_img.apply_(lambda x: id_to_trainid.get(int(x), 255))

train_id_to_color = {cls.train_id: cls.color for cls in Cityscapes.classes if cls.train_id != 255}
train_id_to_color[255] = (0, 0, 0)
def convert_train_id_to_color(prediction: torch.Tensor) -> torch.Tensor:
    batch, _, height, width = prediction.shape
    color_image = torch.zeros((batch, 3, height, width), dtype=torch.uint8)
    for train_id, color in train_id_to_color.items():
        mask = prediction[:, 0] == train_id
        for i in range(3):
            color_image[:, i][mask] = color[i]
    return color_image

def compose_predictions(pred_small: torch.Tensor, 
                        pred_medium: torch.Tensor,
                        pred_big: torch.Tensor,
                        ignore_index=255):
    final = torch.full_like(pred_small, fill_value=ignore_index)
    mask_small = (pred_small != ignore_index)
    final[mask_small] = pred_small[mask_small]
    mask_medium = (final == ignore_index) & (pred_medium != ignore_index)
    final[mask_medium] = pred_medium[mask_medium]
    mask_big = (final == ignore_index) & (pred_big != ignore_index)
    final[mask_big] = pred_big[mask_big]
    return final

def compute_iou(pred: torch.Tensor, target: torch.Tensor, ignore_index=255, num_classes=19):
    ious = []
    for cls in range(num_classes):
        if cls == ignore_index:
            continue
        pred_mask = (pred == cls)
        target_mask = (target == cls)
        intersection = (pred_mask & target_mask).sum().item()
        union = (pred_mask | target_mask).sum().item()
        if union == 0:
            continue
        ious.append(intersection / union)
    if len(ious) == 0:
        return 0.0
    return sum(ious) / len(ious)

def custom_loss(outputs, labels, target_classes, ce_loss_fn, penalty_coef, ignore_index=255):
    """
    Combines the cross entropy loss with an extra penalty that punishes
    the model if it predicts non-target (e.g. background) for pixels that should
    belong to the target classes.
    """
    ce_loss = ce_loss_fn(outputs, labels)
    probs = torch.softmax(outputs, dim=1)
    mask = labels != ignore_index  # only consider valid pixels
    if mask.sum() > 0:
        target_classes_tensor = torch.tensor(target_classes, device=outputs.device)
        target_prob = probs.index_select(1, target_classes_tensor).sum(dim=1)
        penalty = -torch.log(target_prob + 1e-6)
        penalty = penalty[mask]
        penalty = penalty.mean()
    else:
        penalty = 0.0
    return ce_loss + penalty_coef * penalty

def get_args_parser():
    parser = ArgumentParser("Training script for multiple models (small/medium/big) with custom loss.")
    parser.add_argument("--data-dir", type=str, default="./data/cityscapes", help="Path to Cityscapes data")
    parser.add_argument("--batch-size", type=int, default=16, help="Training batch size")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader num_workers")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--experiment-id", type=str, default="unet-training", help="Experiment ID for Weights & Biases")
    return parser

def main(args):
    print(f"Checking dataset path: {args.data_dir}")
    if os.path.exists(args.data_dir):
        print("Contents:", os.listdir(args.data_dir))
        if "leftImg8bit" in os.listdir(args.data_dir) and "gtFine" in os.listdir(args.data_dir):
            print("✅ Required folders (leftImg8bit & gtFine) found!")
        else:
            print("❌ Missing required folders (leftImg8bit or gtFine).")
    else:
        print("Data directory does not exist.")

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
        settings=wandb.Settings(init_timeout=300)
    )
    wandb.define_metric("val_iou_small", summary="max")
    wandb.define_metric("val_iou_medium", summary="max")
    wandb.define_metric("val_iou_big", summary="max")
    wandb.define_metric("val_iou_composed", summary="max")

    from torchvision.datasets import Cityscapes  # Ensure Cityscapes classes are available
    train_dataset = Cityscapes(
        args.data_dir, 
        split="train", 
        mode="fine", 
        target_type="semantic",
        transform=image_transform,
        target_transform=target_transform
    )
    valid_dataset = Cityscapes(
        args.data_dir, 
        split="val",   
        mode="fine", 
        target_type="semantic",
        transform=image_transform,
        target_transform=target_transform
    )

    # For quick debugging, use small subsets:
    train_dataset = torch.utils.data.Subset(train_dataset, list(range(10)))
    valid_dataset = torch.utils.data.Subset(valid_dataset, list(range(5)))

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # Initialize models
    model_small = UNet(in_channels=3, n_classes=19).to(device)
    model_medium = UNet(in_channels=3, n_classes=19).to(device)
    model_big = UNet(in_channels=3, n_classes=19).to(device)

    optimizer_small  = AdamW(model_small.parameters(),  lr=args.lr)
    optimizer_medium = AdamW(model_medium.parameters(), lr=args.lr)
    optimizer_big    = AdamW(model_big.parameters(),    lr=args.lr)
    
    ce_criterion = nn.CrossEntropyLoss(ignore_index=255)
    penalty_coef = 0.5

    os.makedirs("checkpoints", exist_ok=True)
    best_val_loss = float("inf")

    # Define target classes for each model
    target_classes_small  = [5, 6, 7, 11, 12, 17, 18]
    target_classes_medium = [1, 3, 4, 13]
    target_classes_big    = [0, 2, 8, 9, 10, 14, 15, 16]

    global_step = 0  # Global step counter

    for epoch in range(args.epochs):
        print(f"Epoch [{epoch+1}/{args.epochs}]")
        model_small.train()
        model_medium.train()
        model_big.train()

        train_losses_small = []
        train_losses_medium = []
        train_losses_big = []

        # Training loop
        for images, labels in train_dataloader:
            images = images.to(device)
            labels_trainid = convert_to_train_id(labels)

            # --- SMALL model training ---
            labels_small = remap_label_small(labels_trainid).long().squeeze(1).to(device)
            optimizer_small.zero_grad()
            out_small = model_small(images)
            loss_small = custom_loss(out_small, labels_small, target_classes_small, ce_criterion, penalty_coef)
            loss_small.backward()
            optimizer_small.step()
            train_losses_small.append(loss_small.item())

            # --- MEDIUM model training ---
            labels_medium = remap_label_medium(labels_trainid).long().squeeze(1).to(device)
            optimizer_medium.zero_grad()
            out_medium = model_medium(images)
            loss_medium = custom_loss(out_medium, labels_medium, target_classes_medium, ce_criterion, penalty_coef)
            loss_medium.backward()
            optimizer_medium.step()
            train_losses_medium.append(loss_medium.item())

            # --- BIG model training ---
            labels_big = remap_label_big(labels_trainid).long().squeeze(1).to(device)
            optimizer_big.zero_grad()
            out_big = model_big(images)
            loss_big = custom_loss(out_big, labels_big, target_classes_big, ce_criterion, penalty_coef)
            loss_big.backward()
            optimizer_big.step()
            train_losses_big.append(loss_big.item())

            global_step += 1  # Increment global step

        avg_loss_small  = sum(train_losses_small)  / len(train_losses_small)
        avg_loss_medium = sum(train_losses_medium) / len(train_losses_medium)
        avg_loss_big    = sum(train_losses_big)    / len(train_losses_big)
        avg_train_loss = (avg_loss_small + avg_loss_medium + avg_loss_big) / 3.0

        wandb.log({
            "train_loss_small": avg_loss_small,
            "train_loss_medium": avg_loss_medium,
            "train_loss_big": avg_loss_big,
            "train_loss_avg": avg_train_loss,
            "epoch": epoch + 1
        }, step=global_step)

        # ---- VALIDATION ----
        model_small.eval()
        model_medium.eval()
        model_big.eval()

        val_losses_small = []
        val_losses_medium = []
        val_losses_big = []
        iou_small_list = []
        iou_medium_list = []
        iou_big_list = []
        iou_composed_list = []
        logged_images = False

        with torch.no_grad():
            for i, (images, labels) in enumerate(valid_dataloader):
                images = images.to(device)
                labels_trainid = convert_to_train_id(labels)

                # SMALL model validation
                labels_small = remap_label_small(labels_trainid).long().squeeze(1).to(device)
                out_small = model_small(images)
                loss_small = ce_criterion(out_small, labels_small)
                val_losses_small.append(loss_small.item())
                pred_small = out_small.softmax(dim=1).argmax(dim=1)
                iou_small = compute_iou(pred_small, labels_small, ignore_index=255, num_classes=19)
                iou_small_list.append(iou_small)
                if i == 0:
                    pred_small_color = convert_train_id_to_color(pred_small.unsqueeze(1))
                    pred_small_img = make_grid(pred_small_color.cpu(), nrow=4).permute(1, 2, 0).numpy()
                    wandb.log({ "predictions_small": [wandb.Image(pred_small_img)] }, step=global_step)

                # MEDIUM model validation
                labels_medium = remap_label_medium(labels_trainid).long().squeeze(1).to(device)
                out_medium = model_medium(images)
                loss_medium = ce_criterion(out_medium, labels_medium)
                val_losses_medium.append(loss_medium.item())
                pred_medium = out_medium.softmax(dim=1).argmax(dim=1)
                iou_medium = compute_iou(pred_medium, labels_medium, ignore_index=255, num_classes=19)
                iou_medium_list.append(iou_medium)
                if i == 0:
                    pred_medium_color = convert_train_id_to_color(pred_medium.unsqueeze(1))
                    pred_medium_img = make_grid(pred_medium_color.cpu(), nrow=4).permute(1, 2, 0).numpy()
                    wandb.log({ "predictions_medium": [wandb.Image(pred_medium_img)] }, step=global_step)

                # BIG model validation
                labels_big = remap_label_big(labels_trainid).long().squeeze(1).to(device)
                out_big = model_big(images)
                loss_big = ce_criterion(out_big, labels_big)
                val_losses_big.append(loss_big.item())
                pred_big = out_big.softmax(dim=1).argmax(dim=1)
                iou_big = compute_iou(pred_big, labels_big, ignore_index=255, num_classes=19)
                iou_big_list.append(iou_big)
                if i == 0:
                    pred_big_color = convert_train_id_to_color(pred_big.unsqueeze(1))
                    pred_big_img = make_grid(pred_big_color.cpu(), nrow=4).permute(1, 2, 0).numpy()
                    wandb.log({ "predictions_big": [wandb.Image(pred_big_img)] }, step=global_step)

                # Composite prediction
                composed_pred = compose_predictions(pred_small, pred_medium, pred_big, ignore_index=255)
                iou_composed = compute_iou(composed_pred, labels_trainid.squeeze(1).to(device), ignore_index=255, num_classes=19)
                iou_composed_list.append(iou_composed)
                if not logged_images:
                    composed_pred_color = convert_train_id_to_color(composed_pred.unsqueeze(1))
                    composed_pred_img = make_grid(composed_pred_color.cpu(), nrow=4).permute(1, 2, 0).numpy()
                    labels_color = convert_train_id_to_color(labels_trainid)
                    labels_color_grid = make_grid(labels_color.cpu(), nrow=4).permute(1, 2, 0).numpy()
                    wandb.log({
                        "val_composed_prediction": [wandb.Image(composed_pred_img)],
                        "val_ground_truth": [wandb.Image(labels_color_grid)]
                    }, step=global_step)
                    logged_images = True

        avg_val_small  = sum(val_losses_small)  / len(val_losses_small)
        avg_val_medium = sum(val_losses_medium) / len(val_losses_medium)
        avg_val_big    = sum(val_losses_big)    / len(val_losses_big)
        avg_val_loss   = (avg_val_small + avg_val_medium + avg_val_big) / 3.0

        avg_iou_small    = sum(iou_small_list)    / len(iou_small_list)
        avg_iou_medium   = sum(iou_medium_list)   / len(iou_medium_list)
        avg_iou_big      = sum(iou_big_list)      / len(iou_big_list)
        avg_iou_composed = sum(iou_composed_list) / len(iou_composed_list)

        print(f"Epoch {epoch+1}: IoU Small {avg_iou_small:.4f}, IoU Medium {avg_iou_medium:.4f}, IoU Big {avg_iou_big:.4f}, IoU Composed {avg_iou_composed:.4f}")
        
        wandb.log({
            "val_loss_small": avg_val_small,
            "val_loss_medium": avg_val_medium,
            "val_loss_big": avg_val_big,
            "val_loss_avg": avg_val_loss,
            "val_iou_small": avg_iou_small,
            "val_iou_medium": avg_iou_medium,
            "val_iou_big": avg_iou_big,
            "val_iou_composed": avg_iou_composed,
            "epoch": epoch + 1
        }, step=global_step)

        wandb.run.summary["val_iou_small"] = avg_iou_small
        wandb.run.summary["val_iou_medium"] = avg_iou_medium
        wandb.run.summary["val_iou_big"] = avg_iou_big
        wandb.run.summary["val_iou_composed"] = avg_iou_composed

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            checkpoint_path = os.path.join("checkpoints", f"best_models_epoch={epoch+1}_val={avg_val_loss:.4f}.pth")
            torch.save({
                "model_small": model_small.state_dict(),
                "model_medium": model_medium.state_dict(),
                "model_big": model_big.state_dict(),
            }, checkpoint_path)
            print(f"New best model saved at epoch {epoch+1} with val_loss {avg_val_loss:.4f}")

    wandb.finish()
    print("Training complete.")


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)
