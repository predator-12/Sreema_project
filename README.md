# 🧵 StitchSense
### AI-Powered Sewing Defect Detection and Repair Assistant
**Master of Computer Applications (MCA) Project**

---

## 📌 1. Project Overview & Motivation

Garment manufacturing in boutique studios and home tailoring shops frequently suffers from quality inconsistencies due to manual operation. Sewing machine calibration issues—such as improper thread tension, blunt needles, incorrect presser foot pressure, or fabric feeding slippage—lead to seam failures that waste fabric and compromise garment durability.

**StitchSense** is an AI-powered quality inspection and remediation assistant designed specifically for tailors, boutique owners, and quality assurance checkers. The system:
1. **Accepts Garment Images:** High-resolution close-ups of seams and stitch lines captured via smartphone or inspection camera.
2. **Performs Pixel-Level Defect Segmentation:** Employs a lightweight **U-Net** architecture with a **MobileNetV2** encoder via `segmentation-models-pytorch (SMP)`.
3. **Highlights Defective Regions:** Generates color-coded translucent segmentation overlays and bounding boxes.
4. **Classifies Sewing Defects:** Accurately distinguishes between 5 common tailoring defect categories.
5. **Recommends Actionable Tailoring Repairs:** An embedded expert repair engine translates detected machine defects into immediate mechanical adjustments (needle gauge, tension dial, presser foot, stabilizer) and repair methods.
6. **CPU-Optimized Inference:** Engineered to run smoothly on standard Windows 11 laptops with Intel Iris Xe integrated graphics without requiring expensive discrete NVIDIA GPUs.

---

## 🎯 2. Sewing Defect Taxonomy

StitchSense classifies and segments 6 distinct semantic classes:

| Class ID | Defect Category | Visual Characteristics | Mechanical Root Cause |
| :---: | :--- | :--- | :--- |
| **0** | **Normal / Background** | Uniform lockstitch line, balanced tension, zero skips. | Properly calibrated machine. |
| **1** | **Broken / Skipped Stitch** | Gaps in thread line with empty needle puncture holes. | Blunt/bent needle, wrong needle point for fabric (e.g., universal needle on stretch knit), tension snag. |
| **2** | **Seam Puckering** | Sinusoidal fabric wrinkles and ripples radiating from seam. | Excessive upper thread tension, blunt needle, feed dog differential mismatch, tight stitch length. |
| **3** | **Open Seam** | Complete parting of fabric panels exposing raw interior edges. | Omission of reverse lockstitches (backtacking) at seam termination, weak thread breakage. |
| **4** | **Misaligned Seam** | Meandering, crooked stitch line drifting from seam allowance. | Operator pulling/pushing fabric instead of guiding, lack of seam guide, high machine speed. |
| **5** | **Needle Cut / Fabric Hole** | Punctured yarns, chewed knit ladders, fabric fraying at seam. | Damaged burred needle point, using leather cutting point on woven cloth, strike against needle plate. |

---

## 🏗️ 3. System Architecture

```
StitchSense/
│
├── configs/
│   ├── config.yaml            # Master configuration (hyperparameters, paths, thresholds)
│   ├── config.json            # Synchronized JSON configuration fallback
│   └── repair_rules.json      # Tailoring Domain Knowledge Base & Remediation Engine
│
├── data/
│   ├── raw/                   # Unprocessed garment photos
│   ├── processed/
│   │   ├── images/            # Standardized 512x512 RGB training images
│   │   └── masks/             # Single-channel 8-bit ground truth segmentation masks
│   └── splits/                # Train, validation, and test split manifests
│
├── src/
│   ├── __init__.py            # Package initializer
│   ├── dataset.py             # PyTorch Dataset + Albumentations data augmentation
│   ├── model.py               # SMP U-Net with MobileNetV2 + ONNX export
│   ├── loss.py                # Compound Loss: Multiclass Dice Loss + Focal Loss
│   ├── evaluate.py            # Mean IoU (Jaccard Index) & Dice (F1) metric evaluators
│   ├── train.py               # Training loop with validation & early stopping
│   ├── postprocess.py         # Morphological filtering & OpenCV contour instance extraction
│   ├── repair_engine.py       # Tailoring Repair Advisor expert system
│   ├── inference.py           # End-to-end CPU inference pipeline
│   ├── utils.py               # CPU threading setup, overlay blender & config loaders
│   └── synthetic_generator.py # Procedural fabric & seam defect synthesizer
│
├── app/
│   └── app.py                 # Interactive Streamlit Web UI for tailors & boutiques
│
├── sample_test_images/        # Pre-loaded seam defect samples for instant UI evaluation
├── weights/                   # Trained model weights (.pth) and ONNX exports (.onnx)
├── tests/
│   └── test_pipeline.py       # Core unit & integration tests
├── requirements.txt           # Python dependency manifest
└── README.md                  # Project documentation & execution guide
```

---

## ⚙️ 4. Mathematical & Algorithmic Design

### Segmentation Model
- **Decoder:** U-Net preserves high-resolution edge details via long skip connections between encoder and decoder blocks.
- **Encoder:** MobileNetV2 uses depthwise separable convolutions and inverted residual linear bottlenecks, reducing FLOPs by ~80% compared to standard ResNet architectures.
- **Inference Speed:** ~50ms – 90ms on modern Intel Iris Xe CPUs for $512 \times 512$ resolution.

### Compound Loss Formulation
Defect pixels comprise only 1% to 5% of a typical garment photograph. Standard Cross-Entropy produces trivial solutions (predicting background everywhere). StitchSense uses a balanced compound loss:

$$\mathcal{L}_{\text{total}} = \alpha \mathcal{L}_{\text{Focal}} + \beta \mathcal{L}_{\text{Dice}}$$

$$\mathcal{L}_{\text{Focal}} = - (1 - p_t)^\gamma \log(p_t)$$

$$\mathcal{L}_{\text{Dice}} = 1 - \frac{2 \sum (p \cdot y) + \epsilon}{\sum p + \sum y + \epsilon}$$

Where:
- $\gamma = 2.0$ down-weights easily classified background pixels.
- $\alpha = 0.6, \beta = 0.4$ balance boundary sharpness and class overlap.

---

## 🚀 5. Installation & Execution Guide

### Prerequisites
- Windows 11
- Python 3.12.x
- VS Code

### Step 1: Initialize Virtual Environment (`.venv`)
Open PowerShell in the project directory:
```powershell
# Create .venv using Python 3.12
py -3.12 -m venv .venv

# Activate virtual environment
.\.venv\Scripts\Activate.ps1
```

### Step 2: Install PyTorch CPU & Dependencies
Install the official CPU-optimized PyTorch build (CUDA-free) followed by application requirements:
```powershell
# Install PyTorch CPU build
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install remaining libraries
pip install -r requirements.txt
```

### Step 3: Verify Core Functionality
Run the unit test suite to verify configuration loading and the repair engine:
```powershell
python tests/test_pipeline.py
```

### Step 4: Generate Synthetic Dataset (Optional / For Testing)
If you do not have hundreds of pre-annotated real garment images, run the procedural generator:
```powershell
python src/synthetic_generator.py
```
This generates 60 balanced samples with pixel-accurate ground truth masks in `data/processed/` and sample previews in `sample_test_images/`.

### Step 5: Train the Model
```powershell
python src/train.py
```
Checkpoints will be saved automatically to `weights/stitchsense_best.pth` along with an accelerated `weights/stitchsense.onnx` export.

### Step 6: Launch the Professional Flask Web Application
```powershell
python run.py
```
The application will launch at `http://127.0.0.1:5000` with the complete 7-page atelier interface (Home, About, Upload Studio, Dashboard, History, Repair Guide, and Contact).

### Step 7: Alternative Streamlit Dashboard (Optional)
```powershell
streamlit run app/app.py
```

---

## 💡 6. Tailor Repair Advisor Demonstration

When an image is analyzed, StitchSense outputs:
1. **Visual Overlay:** Highlights defects with distinct color borders (e.g., Red for Skipped Stitches, Orange for Puckering).
2. **Defect Severity:** Calculates percentage of seam affected (Minor < 3%, Moderate 3-10%, Severe > 10%).
3. **Machine Remediation Card:**
   - **Tension Dials:** Adjust upper tension by -1.5 for puckering.
   - **Needle Check:** Switch to Ballpoint #75/11 for jersey/spandex knits to eliminate skipped stitches.
   - **Operator Technique:** Use magnetic seam guides and avoid pulling fabric through feed dogs.
4. **Alteration Ticket:** Single-click download of a printable tailoring work order ticket.

---

## 🎓 7. MCA Project Defense & Viva Key Points

1. **Why semantic segmentation instead of object detection (YOLO)?**
   - Object detection draws rectangular bounding boxes. For thin, curving, and irregular defects like seam puckering or skipped stitch lines, bounding boxes encompass mostly non-defective background fabric. Semantic segmentation classifies every single pixel, providing exact defect boundary and seam area measurement.
2. **Why U-Net with MobileNetV2 over heavier backbones (e.g., ResNet-50 or Swin Transformer)?**
   - Home tailors and boutique owners operate standard laptops without dedicated GPUs. MobileNetV2 features inverted residuals and depthwise convolutions that run in under 100 milliseconds on CPU, avoiding latency and high memory overhead.
3. **How does the system handle class imbalance?**
   - Combined Focal Loss and Multiclass Dice Loss specifically prevent the background class from dominating gradient updates.

