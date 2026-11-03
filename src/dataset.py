"""
Dataset module for StitchSense.
Provides PyTorch Dataset and Albumentations augmentations tailored for
garment seam defect segmentation.
"""

import os
from typing import Optional, Callable
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2


class SewingDefectDataset(Dataset):
    """
    PyTorch Dataset for sewing defect segmentation.
    Loads garment RGB images and corresponding single-channel segmentation masks.
    """

    def __init__(
        self,
        images_dir: str,
        masks_dir: Optional[str] = None,
        image_filenames: Optional[list[str]] = None,
        transform: Optional[Callable] = None,
        img_size: tuple[int, int] = (512, 512)
    ):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.img_size = img_size

        if image_filenames is not None:
            self.image_filenames = image_filenames
        else:
            if os.path.exists(images_dir):
                self.image_filenames = sorted([
                    f for f in os.listdir(images_dir)
                    if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))
                ])
            else:
                self.image_filenames = []

    def __len__(self) -> int:
        return len(self.image_filenames)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        filename = self.image_filenames[idx]
        img_path = os.path.join(self.images_dir, filename)

        # Load RGB image
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Failed to load image from {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Load mask if available
        mask = None
        if self.masks_dir and os.path.exists(self.masks_dir):
            base_name = os.path.splitext(filename)[0]
            # Try png first, then same extension
            mask_path = os.path.join(self.masks_dir, f"{base_name}.png")
            if not os.path.exists(mask_path):
                mask_path = os.path.join(self.masks_dir, filename)

            if os.path.exists(mask_path):
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if mask is None:
            # Clean sample (all background = 0)
            mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

        # Apply augmentations / preprocessing
        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image_tensor = augmented["image"]
            mask_tensor = augmented["mask"].long()
        else:
            # Fallback resizing and manual tensor conversion
            image_resized = cv2.resize(image, (self.img_size[1], self.img_size[0]))
            mask_resized = cv2.resize(mask, (self.img_size[1], self.img_size[0]), interpolation=cv2.INTER_NEAREST)
            
            image_norm = image_resized.astype(np.float32) / 255.0
            image_tensor = torch.from_numpy(image_norm).permute(2, 0, 1).float()
            mask_tensor = torch.from_numpy(mask_resized).long()

        return image_tensor, mask_tensor


def get_training_transforms(img_size: tuple[int, int] = (512, 512)) -> A.Compose:
    """
    Data augmentations for training seam defect models:
    Flips, rotations, color shifts, and lighting variations commonly seen in tailor shops.
    """
    h, w = img_size
    return A.Compose([
        A.Resize(h, w, interpolation=cv2.INTER_LINEAR),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),
        A.RandomRotate90(p=0.3),
        A.ShiftScaleRotate(
            shift_limit=0.06,
            scale_limit=0.1,
            rotate_limit=15,
            border_mode=cv2.BORDER_REFLECT,
            p=0.5
        ),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.4),
        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        ),
        ToTensorV2()
    ])


def get_validation_transforms(img_size: tuple[int, int] = (512, 512)) -> A.Compose:
    """Deterministic transforms for validation and evaluation."""
    h, w = img_size
    return A.Compose([
        A.Resize(h, w, interpolation=cv2.INTER_LINEAR),
        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        ),
        ToTensorV2()
    ])

