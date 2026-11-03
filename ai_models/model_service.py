"""
AI Model Service for StitchSense Flask Application.
Bridges Flask HTTP request endpoints with the segmentation model and Tailoring Repair Engine.
Preserves and utilizes the research pipeline (StitchingNet-Seg / src modules).
"""

import os
from PIL import Image
import numpy as np

# Import from existing research codebase
from src.inference import StitchSensePipeline
from src.utils import load_config


class ModelService:
    """Singleton service manager for defect segmentation and repair inference."""
    _instance = None
    _pipeline = None

    @classmethod
    def get_instance(cls, config=None):
        if cls._instance is None:
            cls._instance = cls(config)
        return cls._instance

    def __init__(self, app_config=None):
        config_path = "configs/config.yaml"
        weights_path = "weights/stitchsense_best.pth"
        kb_path = "configs/repair_rules.json"

        if app_config:
            config_path = app_config.get("MODEL_CONFIG_PATH", config_path)
            weights_path = app_config.get("MODEL_WEIGHTS_PATH", weights_path)
            kb_path = app_config.get("REPAIR_RULES_PATH", kb_path)

        self.pipeline = StitchSensePipeline(
            config_path=config_path,
            weights_path=weights_path,
            kb_path=kb_path
        )

    def analyze_image(self, image_path: str, uploads_dir: str) -> dict:
        """
        Execute full defect detection and repair diagnosis on an image.

        Args:
            image_path: Path to the sanitized garment image.
            uploads_dir: Static uploads folder for saving annotated overlay.

        Returns:
            dict containing inspection results, overlay path, and tailor guide.
        """
        results = self.pipeline.predict(image_path)

        # Save visual overlay image
        base_name = os.path.basename(image_path)
        overlay_filename = f"overlay_{base_name}"
        overlay_full_path = os.path.join(uploads_dir, overlay_filename)

        overlay_np = results["overlay_image"]
        Image.fromarray(overlay_np).save(overlay_full_path, format="JPEG", quality=92)

        diagnosis = results["diagnosis"]
        defects = results["defects"]

        return {
            "status": "PASSED" if diagnosis["status"] == "SEAM_PASSED" else "DEFECTS FOUND",
            "is_passed": diagnosis["status"] == "SEAM_PASSED",
            "total_defect_count": diagnosis["total_defect_count"],
            "primary_defect": diagnosis["primary_defect"],
            "overall_severity": diagnosis["overall_severity"],
            "summary": diagnosis["summary"],
            "defect_reports": diagnosis["defect_reports"],
            "overlay_filename": overlay_filename,
            "defects_list": defects
        }

