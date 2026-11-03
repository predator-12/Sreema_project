"""
Post-processing module for StitchSense.
Applies morphological cleanup to raw segmentation masks, extracts defect contours,
computes bounding boxes, calculates severity metrics, and packages findings for the repair engine.
Compatible with OpenCV (cv2) or Scipy/NumPy fallback.
"""

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

try:
    from scipy import ndimage
except ImportError:
    ndimage = None


def clean_segmentation_mask(
    mask: np.ndarray,
    kernel_size: int = 3,
    min_area_px: int = 25
) -> np.ndarray:
    """
    Remove stray single-pixel artifacts and bridge small gaps in detected seams.
    """
    cleaned_mask = np.zeros_like(mask)
    unique_classes = np.unique(mask)

    if cv2 is not None:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        for cid in unique_classes:
            if cid == 0:
                continue

            binary = (mask == cid).astype(np.uint8)
            closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)

            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(opened, connectivity=8)
            for i in range(1, num_labels):
                if stats[i, cv2.CC_STAT_AREA] >= min_area_px:
                    cleaned_mask[labels == i] = cid
    elif ndimage is not None:
        for cid in unique_classes:
            if cid == 0:
                continue
            binary = (mask == cid)
            structure = np.ones((kernel_size, kernel_size), dtype=bool)
            closed = ndimage.binary_closing(binary, structure=structure)
            opened = ndimage.binary_opening(closed, structure=structure)

            labeled, num_features = ndimage.label(opened)
            for i in range(1, num_features + 1):
                area = np.sum(labeled == i)
                if area >= min_area_px:
                    cleaned_mask[labeled == i] = cid
    else:
        # Pass through without morph operations
        cleaned_mask = mask.copy()

    return cleaned_mask


def extract_defect_instances(
    mask: np.ndarray,
    class_names: dict,
    min_area_px: int = 25
) -> list[dict]:
    """
    Find individual defect clusters in the segmentation mask and compute
    their geometric and positional properties.
    """
    h, w = mask.shape[:2]
    total_pixels = h * w
    defects = []

    unique_classes = np.unique(mask)

    for cid in unique_classes:
        if cid == 0:
            continue

        cid_int = int(cid)
        cid_str = str(cid)
        defect_name = class_names.get(cid_int) or class_names.get(cid_str, f"Defect Class {cid_int}")

        binary = (mask == cid_int).astype(np.uint8)

        if cv2 is not None:
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < min_area_px:
                    continue

                x, y, bw, bh = cv2.boundingRect(cnt)
                moments = cv2.moments(cnt)
                if moments["m00"] != 0:
                    cx = int(moments["m10"] / moments["m00"])
                    cy = int(moments["m01"] / moments["m00"])
                else:
                    cx = x + bw // 2
                    cy = y + bh // 2

                area_pct = round((area / total_pixels) * 100, 3)

                defects.append({
                    "class_id": cid_int,
                    "defect_name": defect_name,
                    "bbox": [int(x), int(y), int(bw), int(bh)],
                    "centroid": [int(cx), int(cy)],
                    "area_px": int(area),
                    "area_percent": area_pct
                })
        else:
            # Fallback using scipy.ndimage if available
            if ndimage is not None:
                labeled, num_features = ndimage.label(binary)
                for i in range(1, num_features + 1):
                    component = (labeled == i)
                    area = int(np.sum(component))
                    if area < min_area_px:
                        continue
                    rows = np.any(component, axis=1)
                    cols = np.any(component, axis=0)
                    ymin, ymax = np.where(rows)[0][[0, -1]]
                    xmin, xmax = np.where(cols)[0][[0, -1]]
                    cx = int((xmin + xmax) / 2)
                    cy = int((ymin + ymax) / 2)
                    area_pct = round((area / total_pixels) * 100, 3)
                    defects.append({
                        "class_id": cid_int,
                        "defect_name": defect_name,
                        "bbox": [int(xmin), int(ymin), int(xmax - xmin + 1), int(ymax - ymin + 1)],
                        "centroid": [cx, cy],
                        "area_px": area,
                        "area_percent": area_pct
                    })
            else:
                # Pure NumPy fallback
                area = int(np.sum(binary))
                if area >= min_area_px:
                    rows = np.any(binary, axis=1)
                    cols = np.any(binary, axis=0)
                    if np.any(rows) and np.any(cols):
                        ymin, ymax = np.where(rows)[0][[0, -1]]
                        xmin, xmax = np.where(cols)[0][[0, -1]]
                        cx = int((xmin + xmax) / 2)
                        cy = int((ymin + ymax) / 2)
                        area_pct = round((area / total_pixels) * 100, 3)
                        defects.append({
                            "class_id": cid_int,
                            "defect_name": defect_name,
                            "bbox": [int(xmin), int(ymin), int(xmax - xmin + 1), int(ymax - ymin + 1)],
                            "centroid": [cx, cy],
                            "area_px": area,
                            "area_percent": area_pct
                        })

    defects.sort(key=lambda d: d["area_px"], reverse=True)
    return defects
