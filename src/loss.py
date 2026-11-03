"""
Loss functions for StitchSense.
Implements Combined Dice Loss and Focal Loss to address extreme class imbalance
in sewing defect segmentation (thin stitches vs vast background fabric).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MulticlassDiceLoss(nn.Module):
    """
    Computes Soft Dice Loss across multiple defect classes.
    """
    def __init__(self, smooth: float = 1e-6, ignore_index: int = -100):
        super().__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # logits: (B, C, H, W)
        # targets: (B, H, W)
        num_classes = logits.shape[1]
        probs = F.softmax(logits, dim=1)

        # One-hot encode targets
        targets_one_hot = F.one_hot(torch.clamp(targets, min=0, max=num_classes - 1), num_classes=num_classes)
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()  # (B, C, H, W)

        # Ignore background (class 0) in dice computation if desired, or include all classes
        dims = (0, 2, 3)
        intersection = torch.sum(probs * targets_one_hot, dim=dims)
        cardinality = torch.sum(probs + targets_one_hot, dim=dims)

        dice_score = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        # Focus dice loss on defect classes (classes 1 to C-1)
        defect_dice = dice_score[1:]
        if len(defect_dice) > 0:
            return 1.0 - torch.mean(defect_dice)
        return 1.0 - torch.mean(dice_score)


class FocalLoss(nn.Module):
    """
    Multi-class Focal Loss to down-weight easy background pixels and focus on ambiguous seam edges.
    """
    def __init__(self, gamma: float = 2.0, alpha: torch.Tensor = None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Cross entropy loss per pixel
        ce_loss = F.cross_entropy(logits, targets, reduction="none", weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1.0 - pt) ** self.gamma) * ce_loss
        return torch.mean(focal_loss)


class StitchSenseCompoundLoss(nn.Module):
    """
    Combines Focal Loss and Dice Loss with configurable weights.
    Total Loss = alpha * FocalLoss + beta * DiceLoss
    """
    def __init__(self, focal_weight: float = 0.6, dice_weight: float = 0.4, gamma: float = 2.0):
        super().__init__()
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
        self.focal = FocalLoss(gamma=gamma)
        self.dice = MulticlassDiceLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, dict]:
        focal_loss = self.focal(logits, targets)
        dice_loss = self.dice(logits, targets)
        total_loss = (self.focal_weight * focal_loss) + (self.dice_weight * dice_loss)
        
        metrics = {
            "total_loss": total_loss.item(),
            "focal_loss": focal_loss.item(),
            "dice_loss": dice_loss.item()
        }
        return total_loss, metrics

