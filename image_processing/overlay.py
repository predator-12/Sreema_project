"""
Defect mask and overlay rendering module.
Generates color-coded overlays with bounding boxes and defect labels.
"""

import os
import numpy as np
from PIL import Image

try:
    from src.utils import create_defect_overlay
except ImportError:
    create_defect_overlay = None


def render_defect_overlay(
    image_path: str,
    mask: np.ndarray,
    output_dir: str,
    class_names: dict,
    class_colors: dict,
    alpha: float = 0.45,
    draw_boxes: bool = True
) -> tuple[str, list[dict]]:
    """
    Renders defect overlay on top of the original image and saves the result.

    Returns:
        tuple (overlay_filename: str, detected_defects: list[dict])
    """
    os.makedirs(output_dir, exist_ok=True)
    pil_img = Image.open(image_path).convert("RGB")
    img_np = np.array(pil_img)

    if create_defect_overlay is not None:
        blended, detected_defects = create_defect_overlay(
            image=img_np,
            mask=mask,
            class_names=class_names,
            class_colors=class_colors,
            alpha=alpha,
            draw_boxes=draw_boxes
        )
    else:
        blended = img_np
        detected_defects = []

    base_name = os.path.basename(image_path)
    overlay_filename = f"overlay_{base_name}"
    overlay_path = os.path.join(output_dir, overlay_filename)

    Image.fromarray(blended).save(overlay_path, format="JPEG", quality=92)
    return overlay_filename, detected_defects

