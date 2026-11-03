"""
Utility functions for StitchSense:
Configuration loading, reproducibility seed setting, CPU optimization,
and visual overlay generation for tailoring defect inspection.
"""

import os
import json
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFont

try:
    import yaml
except ImportError:
    yaml = None

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import torch
except ImportError:
    torch = None


def load_config(config_path: str = "configs/config.yaml") -> dict:
    """Load configuration file safely (supports YAML or JSON fallback)."""
    # Check if specified file exists
    if not os.path.exists(config_path):
        json_alt = config_path.replace(".yaml", ".json").replace(".yml", ".json")
        if os.path.exists(json_alt):
            config_path = json_alt
        else:
            raise FileNotFoundError(f"Configuration file not found at: {config_path}")

    if config_path.endswith(".json") or yaml is None:
        # If yaml is not installed or file is json, read json fallback
        json_path = config_path if config_path.endswith(".json") else config_path.replace(".yaml", ".json").replace(".yml", ".json")
        if os.path.exists(json_path):
            with open(json_path, "r", encoding="utf-8") as f:
                return json.load(f)

    if yaml is not None:
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    raise ImportError("PyYAML is not installed and no JSON fallback was found.")


def set_seed(seed: int = 42) -> None:
    """Set random seed for full reproducibility across NumPy, Random, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def get_device(config: dict):
    """
    Configure and return PyTorch compute device.
    Optimized for Intel Iris Xe CPU execution with multi-threading.
    """
    if torch is None:
        return "cpu"

    if config.get("device", {}).get("use_cuda", False) and torch.cuda.is_available():
        return torch.device("cuda")
    
    num_threads = config.get("device", {}).get("num_threads", 4)
    torch.set_num_threads(num_threads)
    return torch.device("cpu")


def create_defect_overlay(
    image: np.ndarray,
    mask: np.ndarray,
    class_names: dict,
    class_colors: dict,
    alpha: float = 0.45,
    draw_boxes: bool = True
) -> tuple[np.ndarray, list[dict]]:
    """
    Blend color-coded defect segmentation masks onto the original garment image.
    Optionally draws bounding boxes and tags around identified defect regions.
    """
    h, w = mask.shape[:2]
    if image.shape[:2] != (h, w):
        if cv2 is not None:
            image = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)
        else:
            image = np.array(Image.fromarray(image).resize((w, h), Image.Resampling.BILINEAR))

    overlay = image.copy()
    color_mask = np.zeros_like(image, dtype=np.uint8)
    detected_defects = []

    unique_classes = np.unique(mask)

    for class_id in unique_classes:
        if class_id == 0:
            continue

        cid_str = str(class_id)
        cid_int = int(class_id)
        color = class_colors.get(cid_int) or class_colors.get(cid_str, [255, 0, 0])
        name = class_names.get(cid_int) or class_names.get(cid_str, f"Defect {cid_int}")

        binary_mask = (mask == cid_int).astype(np.uint8)
        color_mask[binary_mask == 1] = color

        if cv2 is not None:
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < 20:
                    continue
                x, y, bw, bh = cv2.boundingRect(cnt)
                detected_defects.append({
                    "class_id": cid_int,
                    "name": name,
                    "bbox": [int(x), int(y), int(bw), int(bh)],
                    "area_px": int(area),
                    "area_percent": round((area / (h * w)) * 100, 2),
                    "color_rgb": color
                })
        else:
            # Fallback bounding box calculation using NumPy
            rows = np.any(binary_mask, axis=1)
            cols = np.any(binary_mask, axis=0)
            if np.any(rows) and np.any(cols):
                ymin, ymax = np.where(rows)[0][[0, -1]]
                xmin, xmax = np.where(cols)[0][[0, -1]]
                area = int(np.sum(binary_mask))
                if area >= 20:
                    detected_defects.append({
                        "class_id": cid_int,
                        "name": name,
                        "bbox": [int(xmin), int(ymin), int(xmax - xmin), int(ymax - ymin)],
                        "area_px": area,
                        "area_percent": round((area / (h * w)) * 100, 2),
                        "color_rgb": color
                    })

    # Blend color mask onto image
    mask_pixels = (mask > 0)
    blended = image.copy().astype(np.float32)
    blended[mask_pixels] = (
        (1 - alpha) * image[mask_pixels].astype(np.float32) +
        alpha * color_mask[mask_pixels].astype(np.float32)
    )
    blended = np.clip(blended, 0, 255).astype(np.uint8)

    # Draw bounding boxes and text labels
    if draw_boxes and detected_defects:
        if cv2 is not None:
            for defect in detected_defects:
                x, y, bw, bh = defect["bbox"]
                color = [int(c) for c in defect["color_rgb"]]
                cv2.rectangle(blended, (x, y), (x + bw, y + bh), color, 2)
                label = f"{defect['name']} ({defect['area_percent']}%)"
                (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                badge_y1 = max(0, y - text_h - 6)
                badge_x2 = min(w, x + text_w + 6)
                cv2.rectangle(blended, (x, badge_y1), (badge_x2, y), color, -1)
                cv2.putText(
                    blended, label, (x + 3, y - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA
                )
        else:
            pil_img = Image.fromarray(blended)
            draw = ImageDraw.Draw(pil_img)
            for defect in detected_defects:
                x, y, bw, bh = defect["bbox"]
                color = tuple(defect["color_rgb"])
                draw.rectangle([x, y, x + bw, y + bh], outline=color, width=2)
                label = f"{defect['name']} ({defect['area_percent']}%)"
                draw.rectangle([x, max(0, y - 16), x + 150, y], fill=color)
                draw.text((x + 2, max(0, y - 14)), label, fill=(255, 255, 255))
            blended = np.array(pil_img)

    return blended, detected_defects
