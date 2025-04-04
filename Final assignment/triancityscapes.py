import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

# These imports come from the cloned DeepLabV3Plus-Pytorch repository
# Adjust the paths if your repo folders differ
import network
from utils import ext_transforms as et
from metrics import StreamSegMecdtrics
from datasets import Cityscapes  # The Cityscapes Dataset class in the repo

def get_cityscapes_loaders(data_root, batch_size=4, crop_size=768):
    """
    Create DataLoader objects for Cityscapes train/val.
    Adjust transforms (crop, resize, etc.) as you wish.
    """

    # Example transforms for augmentation + normalization
    train_transform = et.ExtCompose([
        et.ExtResize( (crop_size, crop_size) ),
        et.ExtRandomCrop(size=(crop_size, crop_size)),
        et.ExtRandomHorizontalFlip(),
        et.ExtToTensor(),
        et.ExtNormalize(mean=(0.485, 0.456, 0.406),
                        std=(0.229, 0.224, 0.225))
    ])

    val_transform = et.ExtCompose([
        et.ExtResize( (crop_size, crop_size) ),
        et.ExtToTensor(),
        et.ExtNormalize(mean=(0.485, 0.456, 0.406),
                        std=(0.229, 0.224, 0.225))
    ])

    # Create Datasets
    train_dst = Cityscapes(root=data_root, split='train', transform=train_transform)
    val_dst   = Cityscapes(root=data_root, split='val',   transform=val_transform)
    
    train_loader = DataLoader(train_dst, batch_size=batch_size,
                              shuffle=True, num_workers=4, drop_last=True)
    val_loader   = DataLoader(val_dst, batch_size=1,
                              shuffle=False, num_workers=4)

    return train_loader, val_loader

def main():
    ############################################################################
    # 1) Hyperparameters & Config
    ############################################################################
    data_root       = "./datasets/data/cityscapes"   # Path to Cityscapes data
    ckpt_path       = "checkpoints/best_deeplabv3plus_resnet101_cityscapes_os16.pth" 
    num_classes     = 19
    output_stride   = 16
    batch_size      = 4    # Adjust based on your GPU memory
    lr              = 0.01 # Example learning rate
    num_epochs      = 50   # Set as you need
    crop_size       = 768  # Typical for Cityscapes training
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ############################################################################
    # 2) Create Model & Load Pretrained Weights
    ############################################################################
    print("Loading DeeplabV3+ (ResNet101) model...")
    model = network.modeling.__dict__['deeplabv3plus_resnet101'](
        num_classes=num_classes, output_stride=output_stride
    )

    # Load the pretrained checkpoint
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state'])
    model.to(device)

    ############################################################################
    # 3) Create Dataloaders
    ############################################################################
    print("Preparing Cityscapes dataloaders...")
    train_loader, val_loader = get_cityscapes_loaders(
        data_root=data_root,
        batch_size=batch_size,
        crop_size=crop_size
    )

    ############################################################################
    # 4) Define Loss, Optimizer, Metrics
    ############################################################################
    criterion = nn.CrossEntropyLoss(ignore_index=255)  # standard for segmentation
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)

    # For optional validation metrics
    metrics = StreamSegMetrics(num_classes=num_classes)

    ############################################################################
    # 5) Training Loop
    ############################################################################
    print("Start training ...")
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backprop
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        
        # Print average training loss of the epoch
        epoch_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}]  Train Loss: {epoch_loss:.4f}")

        # ----------------------------------------------------------------------
        # Optional: Evaluate on validation set each epoch
        # ----------------------------------------------------------------------
        model.eval()
        metrics.reset()
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                preds = outputs.max(1)[1].cpu().numpy()   # argmax
                targets = labels.cpu().numpy()

                metrics.update(targets, preds)
        
        score = metrics.get_results()
        print(f"Val mIOU: {score['Mean IoU']:.4f}, Acc: {score['Overall Acc']:.4f}")

    print("Training Complete.")

if __name__ == '__main__':
    main()
