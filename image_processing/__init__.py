"""
Image processing package for StitchSense.
Handles file validation, preprocessing, morphological mask refinement,
and visual defect overlay generation.
"""

from image_processing.preprocessor import validate_and_save_upload, allowed_file
from image_processing.overlay import render_defect_overlay

__all__ = ["validate_and_save_upload", "allowed_file", "render_defect_overlay"]

