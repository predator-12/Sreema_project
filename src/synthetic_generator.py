"""
Procedural Synthetic Seam Defect Generator for StitchSense.
Synthesizes realistic fabric weave textures, seams, and precisely annotated
ground truth masks for all 5 sewing defect categories + normal seams.

Uses pure Pillow + NumPy so it can execute out-of-the-box in any standard Python environment.
"""

import os
import random
import numpy as np
from PIL import Image, ImageDraw


def create_fabric_texture(h: int = 512, w: int = 512) -> np.ndarray:
    """Procedurally generate textured fabric canvas with realistic weave noise."""
    base_color_palette = [
        [220, 215, 205],  # Cream / Off-white cotton
        [70, 95, 140],    # Denim blue
        [180, 160, 140],  # Khaki / Linen
        [190, 70, 80],    # Crimson tailor cloth
        [75, 120, 95],    # Sage green
        [230, 230, 235],  # White poplin
        [140, 140, 150]   # Slate grey
    ]
    base_rgb = random.choice(base_color_palette)
    canvas = np.full((h, w, 3), base_rgb, dtype=np.float32)

    x = np.arange(w)
    y = np.arange(h)
    xx, yy = np.meshgrid(x, y)

    # Micro weave pattern
    weave = np.sin(xx * 0.8) * np.cos(yy * 0.8) * 8.0
    noise = np.random.normal(0, 4.0, (h, w, 1))

    canvas = canvas + weave[:, :, None] + noise
    return np.clip(canvas, 0, 255).astype(np.uint8)


def generate_seam_sample(
    h: int = 512,
    w: int = 512,
    defect_class: int = 0
) -> tuple[Image.Image, Image.Image, str]:
    """
    Generate an image and corresponding single-channel segmentation mask for a specified defect class.
    
    Classes:
        0: Normal / Background (Clean Seam)
        1: Broken / Skipped Stitch
        2: Seam Puckering
        3: Open Seam
        4: Misaligned Seam
        5: Needle Cut / Fabric Hole
    """
    fabric_np = create_fabric_texture(h, w)
    image = Image.fromarray(fabric_np, mode="RGB")
    mask = Image.new("L", (w, h), 0)

    draw_img = ImageDraw.Draw(image)
    draw_mask = ImageDraw.Draw(mask)

    # Seam parameters
    seam_x = w // 2 + random.randint(-30, 30)
    stitch_color = (245, 245, 240) if np.mean(fabric_np) < 160 else (35, 35, 40)
    shadow_color = (max(0, fabric_np[0, 0, 0] - 40), max(0, fabric_np[0, 0, 1] - 40), max(0, fabric_np[0, 0, 2] - 40))

    # Seam groove
    draw_img.line([(seam_x, 0), (seam_x, h)], fill=tuple(shadow_color), width=2)
    draw_img.line([(seam_x + 1, 0), (seam_x + 1, h)], fill=(250, 250, 250), width=1)

    stitch_length = random.randint(14, 20)
    stitch_gap = random.randint(6, 10)
    step = stitch_length + stitch_gap

    if defect_class == 0:
        # Class 0: Clean normal seam
        y = 10
        while y < h - 20:
            draw_img.line([(seam_x, y), (seam_x, y + stitch_length)], fill=stitch_color, width=2)
            draw_img.ellipse([(seam_x - 1, y - 1), (seam_x + 1, y + 1)], fill=(20, 20, 20))
            draw_img.ellipse([(seam_x - 1, y + stitch_length - 1), (seam_x + 1, y + stitch_length + 1)], fill=(20, 20, 20))
            y += step
        desc = "Normal Seam"

    elif defect_class == 1:
        # Class 1: Broken / Skipped Stitch
        skip_start = random.randint(h // 4, h // 2)
        skip_end = skip_start + random.randint(80, 140)

        y = 10
        while y < h - 20:
            if skip_start <= y <= skip_end:
                # Needle puncture holes without thread
                draw_img.ellipse([(seam_x - 2, y - 2), (seam_x + 2, y + 2)], fill=(25, 25, 25))
                # Mark mask as Class 1
                draw_mask.ellipse([(seam_x - 6, y - 6), (seam_x + 6, y + 6)], fill=1)
                draw_mask.line([(seam_x, skip_start), (seam_x, skip_end)], fill=1, width=6)
            else:
                draw_img.line([(seam_x, y), (seam_x, y + stitch_length)], fill=stitch_color, width=2)
                draw_img.ellipse([(seam_x - 1, y - 1), (seam_x + 1, y + 1)], fill=(20, 20, 20))
            y += step
        desc = "Skipped Stitch"

    elif defect_class == 2:
        # Class 2: Seam Puckering
        puck_start = random.randint(h // 4, h // 2)
        puck_end = puck_start + random.randint(120, 200)

        # Draw wrinkling waves radiating from seam
        for py in range(puck_start, puck_end, 14):
            for side in [-1, 1]:
                pts = []
                for dx in range(0, 80, 8):
                    offset_y = int(np.sin(dx * 0.08) * 8.0)
                    pts.append((seam_x + side * dx, py + offset_y))
                draw_img.line(pts, fill=tuple(shadow_color), width=2)

        draw_mask.rectangle([(seam_x - 30, puck_start), (seam_x + 30, puck_end)], fill=2)

        y = 10
        while y < h - 20:
            draw_img.line([(seam_x, y), (seam_x, y + stitch_length - 4)], fill=stitch_color, width=2)
            y += step - 3
        desc = "Seam Puckering"

    elif defect_class == 3:
        # Class 3: Open Seam
        open_start = random.randint(h // 4, h // 2)
        open_end = open_start + random.randint(90, 160)
        gap_w = random.randint(18, 28)

        polygon = [
            (seam_x, open_start),
            (seam_x - gap_w // 2, (open_start + open_end) // 2),
            (seam_x, open_end),
            (seam_x + gap_w // 2, (open_start + open_end) // 2)
        ]
        draw_img.polygon(polygon, fill=(20, 18, 20))
        draw_mask.polygon(polygon, fill=3)

        y = 10
        while y < h - 20:
            if not (open_start <= y <= open_end):
                draw_img.line([(seam_x, y), (seam_x, y + stitch_length)], fill=stitch_color, width=2)
            y += step
        desc = "Open Seam"

    elif defect_class == 4:
        # Class 4: Misaligned / Wavy Seam
        drift_start = random.randint(60, 120)
        drift_end = random.randint(h // 2, h - 80)

        y = 10
        while y < h - 20:
            if drift_start <= y <= drift_end:
                drift_x = int(seam_x + np.sin((y - drift_start) * 0.04) * 35.0)
                draw_img.line([(drift_x, y), (drift_x, y + stitch_length)], fill=stitch_color, width=2)
                draw_mask.ellipse([(drift_x - 7, y - 7), (drift_x + 7, y + 7)], fill=4)
                draw_mask.line([(seam_x, y), (drift_x, y)], fill=4, width=5)
            else:
                draw_img.line([(seam_x, y), (seam_x, y + stitch_length)], fill=stitch_color, width=2)
            y += step
        desc = "Misaligned Seam"

    elif defect_class == 5:
        # Class 5: Needle Cut / Chewed Holes
        cut_center_y = random.randint(h // 3, 2 * h // 3)
        num_holes = random.randint(5, 10)

        y = 10
        while y < h - 20:
            draw_img.line([(seam_x, y), (seam_x, y + stitch_length)], fill=stitch_color, width=2)
            y += step

        for _ in range(num_holes):
            hx = seam_x + random.randint(-12, 12)
            hy = cut_center_y + random.randint(-40, 40)
            radius = random.randint(4, 7)
            draw_img.ellipse([(hx - radius, hy - radius), (hx + radius, hy + radius)], fill=(20, 20, 25))
            draw_img.ellipse([(hx - radius - 2, hy - radius - 2), (hx + radius + 2, hy + radius + 2)], outline=(180, 160, 140), width=1)
            draw_mask.ellipse([(hx - radius - 3, hy - radius - 3), (hx + radius + 3, hy + radius + 3)], fill=5)
        desc = "Needle Cut"

    return image, mask, desc


def generate_dataset(
    output_images_dir: str = "data/processed/images",
    output_masks_dir: str = "data/processed/masks",
    samples_per_class: int = 10,
    sample_preview_dir: str = "sample_test_images"
) -> None:
    """Generate balanced synthetic dataset for training and interactive UI demonstration."""
    os.makedirs(output_images_dir, exist_ok=True)
    os.makedirs(output_masks_dir, exist_ok=True)
    os.makedirs(sample_preview_dir, exist_ok=True)

    print(f"[Generator] Generating synthetic seam dataset ({samples_per_class} per class)...")
    total_generated = 0

    for c in range(6):
        for idx in range(samples_per_class):
            img, mask, desc = generate_seam_sample(h=512, w=512, defect_class=c)
            fname = f"seam_c{c}_{idx:03d}"

            img_path = os.path.join(output_images_dir, f"{fname}.jpg")
            mask_path = os.path.join(output_masks_dir, f"{fname}.png")

            img.save(img_path, quality=95)
            mask.save(mask_path)
            total_generated += 1

            if idx == 0:
                demo_path = os.path.join(sample_preview_dir, f"demo_{desc.lower().replace(' ', '_')}.jpg")
                img.save(demo_path, quality=95)

    print(f"[Generator] Successfully created {total_generated} samples in '{output_images_dir}' and masks in '{output_masks_dir}'.")
    print(f"[Generator] Sample test images ready in '{sample_preview_dir}'.")


if __name__ == "__main__":
    generate_dataset()