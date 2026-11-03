"""
Secure image upload and preprocessing handler.
Protects against directory traversal, malicious payloads, and invalid image formats.
"""

import os
import uuid
from PIL import Image, ImageOps

try:
    from werkzeug.utils import secure_filename
except ImportError:
    import re
    def secure_filename(filename: str) -> str:
        """Fallback secure filename generator."""
        clean = re.sub(r'[^a-zA-Z0-9_.-]', '_', os.path.basename(filename))
        return clean.strip('_') or 'upload'


def allowed_file(filename: str, allowed_extensions: set) -> bool:
    """Check if uploaded file has an allowed image extension."""
    if not filename or "." not in filename:
        return False
    ext = filename.rsplit(".", 1)[1].lower()
    return ext in allowed_extensions


def validate_and_save_upload(
    file_storage,
    upload_folder: str,
    allowed_extensions: set = {"png", "jpg", "jpeg", "webp"},
    max_dimension: int = 1024
) -> tuple[bool, str, str]:
    """
    Validate, sanitize, strip metadata, and securely save an uploaded image file.

    Args:
        file_storage: Werkzeug FileStorage or file-like object.
        upload_folder: Target directory to save the file.
        allowed_extensions: Set of permitted file extensions.
        max_dimension: Maximum width or height constraint.

    Returns:
        tuple (success: bool, saved_filename: str, error_message: str)
    """
    if not file_storage or file_storage.filename == "":
        return False, "", "No file selected for upload."

    raw_filename = file_storage.filename
    if not allowed_file(raw_filename, allowed_extensions):
        exts_str = ", ".join(sorted(list(allowed_extensions)))
        return False, "", f"Unsupported file type. Permitted formats: {exts_str}"

    os.makedirs(upload_folder, exist_ok=True)

    # Sanitize and assign collision-proof unique filename
    safe_name = secure_filename(raw_filename)
    unique_prefix = uuid.uuid4().hex[:10]
    final_filename = f"{unique_prefix}_{safe_name}"
    save_path = os.path.join(upload_folder, final_filename)

    try:
        # Verify genuine image integrity using Pillow
        image = Image.open(file_storage)
        image.verify()  # Check for header corruptions or malformed files

        # Re-open for actual processing (verify closes file pointer)
        file_storage.seek(0)
        image = Image.open(file_storage)
        
        # Handle camera EXIF auto-rotation (e.g. smartphone portraits)
        image = ImageOps.exif_transpose(image)
        image = image.convert("RGB")

        # Downscale gracefully if massive resolution to preserve memory
        if max(image.size) > max_dimension:
            image.thumbnail((max_dimension, max_dimension), Image.Resampling.LANCZOS)

        # Save sanitized RGB JPEG (strips any hidden EXIF exploit scripts)
        image.save(save_path, format="JPEG", quality=92)
        return True, final_filename, ""

    except Exception as e:
        return False, "", f"Corrupted or invalid image file: {str(e)}"

