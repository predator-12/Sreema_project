"""
Model definition and factory module for StitchSense.
Wraps Segmentation Models PyTorch (SMP) with CPU-optimized architectures (U-Net + MobileNetV2).
Includes ONNX export for Intel Iris Xe hardware acceleration.
"""

import os
import torch
import torch.nn as nn

try:
    import segmentation_models_pytorch as smp
    SMP_AVAILABLE = True
except ImportError:
    SMP_AVAILABLE = False


def build_model(config: dict) -> nn.Module:
    """
    Build and return the semantic segmentation neural network.
    Default: U-Net with MobileNetV2 backbone (ideal for CPU inference).

    Args:
        config: Configuration dictionary loaded from config.yaml.

    Returns:
        PyTorch nn.Module ready for training or evaluation.
    """
    if not SMP_AVAILABLE:
        raise ImportError(
            "segmentation_models_pytorch is not installed. "
            "Please install it using: pip install segmentation-models-pytorch"
        )

    model_cfg = config.get("model", {})
    arch_name = model_cfg.get("architecture", "Unet")
    encoder_name = model_cfg.get("encoder_name", "mobilenet_v2")
    encoder_weights = model_cfg.get("encoder_weights", "imagenet")
    in_channels = model_cfg.get("in_channels", 3)
    classes = model_cfg.get("classes", 6)

    # SMP model factory mapping
    arch_constructors = {
        "Unet": smp.Unet,
        "DeepLabV3Plus": smp.DeepLabV3Plus,
        "FPN": smp.FPN,
        "PAN": smp.PAN,
        "Linknet": smp.Linknet
    }

    if arch_name not in arch_constructors:
        raise ValueError(f"Unsupported architecture: '{arch_name}'. Choose from {list(arch_constructors.keys())}")

    constructor = arch_constructors[arch_name]

    try:
        model = constructor(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
            activation=None  # Output raw logits for numerical stability with Loss functions
        )
    except Exception as e:
        # Fallback if offline and ImageNet weights cannot be downloaded from GitHub
        print(f"[Warning] Failed to download encoder weights '{encoder_weights}' ({e}). Falling back to randomly initialized encoder.")
        model = constructor(
            encoder_name=encoder_name,
            encoder_weights=None,
            in_channels=in_channels,
            classes=classes,
            activation=None
        )

    return model


def export_to_onnx(
    model: nn.Module,
    save_path: str = "weights/stitchsense.onnx",
    img_size: tuple[int, int] = (512, 512)
) -> str:
    """
    Export PyTorch model to ONNX format for accelerated Intel Iris Xe CPU runtime.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.eval()
    dummy_input = torch.randn(1, 3, img_size[0], img_size[1], dtype=torch.float32)

    torch.onnx.export(
        model,
        dummy_input,
        save_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["input_image"],
        output_names=["defect_logits"],
        dynamic_axes={
            "input_image": {0: "batch_size"},
            "defect_logits": {0: "batch_size"}
        }
    )
    return save_path

