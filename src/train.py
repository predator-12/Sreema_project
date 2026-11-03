"""
Training pipeline for StitchSense.
Trains the lightweight U-Net / MobileNetV2 segmentation model using
Combined Focal + Dice Loss with Cosine Annealing learning rate schedule.
"""

import os
import json
import time
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.utils import load_config, set_seed, get_device
from src.dataset import SewingDefectDataset, get_training_transforms, get_validation_transforms
from src.model import build_model, export_to_onnx
from src.loss import StitchSenseCompoundLoss
from src.evaluate import evaluate_model


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Execute one training epoch."""
    model.train()
    running_loss = 0.0
    running_focal = 0.0
    running_dice = 0.0

    pbar = tqdm(dataloader, desc="Training", leave=False)
    for images, masks in pbar:
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss, loss_metrics = criterion(logits, masks)
        
        loss.backward()
        optimizer.step()

        running_loss += loss_metrics["total_loss"]
        running_focal += loss_metrics["focal_loss"]
        running_dice += loss_metrics["dice_loss"]

        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    n = max(len(dataloader), 1)
    return {
        "loss": running_loss / n,
        "focal_loss": running_focal / n,
        "dice_loss": running_dice / n
    }


def validate_epoch(model, dataloader, criterion, device):
    """Execute one validation epoch."""
    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        for images, masks in tqdm(dataloader, desc="Validating", leave=False):
            images = images.to(device)
            masks = masks.to(device)

            logits = model(images)
            loss, loss_metrics = criterion(logits, masks)
            running_loss += loss_metrics["total_loss"]

    n = max(len(dataloader), 1)
    return {"val_loss": running_loss / n}


def main():
    print("=" * 60)
    print("StitchSense Model Training Pipeline")
    print("=" * 60)

    config = load_config("configs/config.yaml")
    set_seed(config.get("seed", 42))
    device = get_device(config)
    print(f"[Info] Running on device: {device}")

    # Dataset directories
    images_dir = config["dataset"]["images_dir"]
    masks_dir = config["dataset"]["masks_dir"]
    img_size = tuple(config["dataset"]["image_size"])
    batch_size = config["dataset"]["batch_size"]

    if not os.path.exists(images_dir) or len(os.listdir(images_dir)) == 0:
        print(f"[Warning] No training images found in '{images_dir}'.")
        print("[Tip] Generate synthetic training samples first using: python src/synthetic_generator.py")
        return

    # Datasets and loaders
    train_dataset = SewingDefectDataset(
        images_dir=images_dir,
        masks_dir=masks_dir,
        transform=get_training_transforms(img_size),
        img_size=img_size
    )

    # 80/20 train/val split
    total_len = len(train_dataset)
    val_len = max(int(0.2 * total_len), 1)
    train_len = total_len - val_len

    train_subset, val_subset = torch.utils.data.random_split(
        train_dataset, [train_len, val_len],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=config["dataset"]["num_workers"],
        pin_memory=False
    )

    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config["dataset"]["num_workers"],
        pin_memory=False
    )

    print(f"[Info] Training samples: {train_len} | Validation samples: {val_len}")

    # Model, Loss, Optimizer
    model = build_model(config).to(device)
    criterion = StitchSenseCompoundLoss(
        focal_weight=config["training"]["focal_loss_weight"],
        dice_weight=config["training"]["dice_loss_weight"]
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"]
    )

    epochs = config["training"]["epochs"]
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    save_dir = config["training"]["save_dir"]
    os.makedirs(save_dir, exist_ok=True)
    best_weights_path = os.path.join(save_dir, config["training"]["checkpoint_name"])

    best_val_loss = float("inf")
    patience = config["training"]["early_stopping_patience"]
    patience_counter = 0
    history = []

    print(f"\n[Info] Starting training for {epochs} epochs...")
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        train_res = train_epoch(model, train_loader, criterion, optimizer, device)
        val_res = validate_epoch(model, val_loader, criterion, device)
        scheduler.step()

        val_loss = val_res["val_loss"]
        epoch_duration = time.time() - epoch_start

        log_entry = {
            "epoch": epoch,
            "train_loss": round(train_res["loss"], 4),
            "val_loss": round(val_loss, 4),
            "lr": round(scheduler.get_last_lr()[0], 6),
            "duration_sec": round(epoch_duration, 1)
        }
        history.append(log_entry)

        print(f"Epoch [{epoch:02d}/{epochs:02d}] | Train Loss: {train_res['loss']:.4f} | Val Loss: {val_loss:.4f} | Time: {epoch_duration:.1f}s")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "config": config
            }, best_weights_path)
            print(f"  --> Checkpoint saved: {best_weights_path} (Val Loss: {val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n[Early Stopping] No improvement for {patience} epochs. Stopping training.")
                break

    total_training_time = time.time() - start_time
    print(f"\n[Done] Training completed in {total_training_time / 60:.2f} minutes.")

    # Save training history
    history_path = os.path.join(save_dir, "train_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    # Optional ONNX export for Intel Iris Xe deployment
    try:
        onnx_path = os.path.join(save_dir, "stitchsense.onnx")
        export_to_onnx(model, onnx_path, img_size)
        print(f"[Info] Exported accelerated ONNX model to {onnx_path}")
    except Exception as e:
        print(f"[Notice] ONNX export skipped: {e}")


if __name__ == "__main__":
    main()

