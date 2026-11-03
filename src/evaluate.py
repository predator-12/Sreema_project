"""
Evaluation metrics module for StitchSense.
Calculates Mean IoU (mIoU), Dice coefficient, precision, and recall
across garment defect classes.
"""

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm


def calculate_iou_and_dice(
    preds: np.ndarray,
    targets: np.ndarray,
    num_classes: int = 6
) -> dict:
    """
    Compute intersection over union (IoU) and Dice score for each class.

    Args:
        preds: 2D or 3D integer array of class predictions.
        targets: 2D or 3D integer array of ground truth class labels.
        num_classes: Total number of classes including background.

    Returns:
        dict containing per-class and mean metrics.
    """
    ious = []
    dices = []
    class_results = {}

    for c in range(num_classes):
        pred_c = (preds == c)
        target_c = (targets == c)

        intersection = np.logical_and(pred_c, target_c).sum()
        union = np.logical_or(pred_c, target_c).sum()
        cardinality = pred_c.sum() + target_c.sum()

        if union == 0:
            iou = 1.0 if cardinality == 0 else 0.0
            dice = 1.0 if cardinality == 0 else 0.0
        else:
            iou = intersection / union
            dice = (2.0 * intersection) / cardinality if cardinality > 0 else 0.0

        ious.append(iou)
        dices.append(dice)
        class_results[c] = {"iou": float(iou), "dice": float(dice)}

    # Defect-only averages (classes 1 to num_classes-1)
    defect_ious = ious[1:]
    defect_dices = dices[1:]

    return {
        "mean_iou_all": float(np.mean(ious)),
        "mean_dice_all": float(np.mean(dices)),
        "mean_iou_defects": float(np.mean(defect_ious)) if defect_ious else 0.0,
        "mean_dice_defects": float(np.mean(defect_dices)) if defect_dices else 0.0,
        "class_breakdown": class_results
    }


def evaluate_model(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    num_classes: int = 6
) -> dict:
    """
    Run full evaluation loop over a dataset.
    """
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, masks in tqdm(dataloader, desc="Evaluating", leave=False):
            images = images.to(device)
            logits = model(images)
            probs = F.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1).cpu().numpy()

            all_preds.append(preds)
            all_targets.append(masks.numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    return calculate_iou_and_dice(all_preds, all_targets, num_classes=num_classes)

