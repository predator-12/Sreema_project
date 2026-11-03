"""
StitchSense – AI-Powered Sewing Defect Detection and Repair Assistant
Streamlit Web Application designed for Home Tailors and Boutique Owners.
"""

import os
import sys
from PIL import Image
import numpy as np
import streamlit as st

# Ensure project root is in python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils import load_config
from src.inference import StitchSensePipeline


st.set_page_config(
    page_title="StitchSense | AI Sewing Defect Assistant",
    page_icon="🧵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for tailored boutique aesthetics
st.markdown("""
<style>
    .main-header {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        color: #1E293B;
        margin-bottom: 0px;
    }
    .sub-header {
        color: #64748B;
        font-size: 1.1rem;
        margin-bottom: 25px;
    }
    .metric-card {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .metric-title {
        color: #64748B;
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
    }
    .metric-value {
        color: #0F172A;
        font-size: 1.5rem;
        font-weight: 700;
        margin-top: 5px;
    }
    .status-pass {
        color: #16A34A !important;
    }
    .status-fail {
        color: #DC2626 !important;
    }
    .repair-card {
        background-color: #FFFFFF;
        border-left: 5px solid #3B82F6;
        border-radius: 8px;
        padding: 18px;
        margin-bottom: 20px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.04);
        border-top: 1px solid #F1F5F9;
        border-right: 1px solid #F1F5F9;
        border-bottom: 1px solid #F1F5F9;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_pipeline():
    """Load and cache the StitchSense inference pipeline."""
    return StitchSensePipeline(
        config_path="configs/config.yaml",
        weights_path="weights/stitchsense_best.pth",
        kb_path="configs/repair_rules.json"
    )


def main():
    st.markdown('<h1 class="main-header">🧵 StitchSense</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-Powered Sewing Defect Detection & Repair Assistant for Tailors and Boutiques</p>', unsafe_allow_html=True)

    pipeline = get_pipeline()

    # Sidebar Controls
    st.sidebar.header("📍 Garment Input")
    input_mode = st.sidebar.radio("Select Input Method:", ["Upload Image", "Choose Sample Garment"])

    image_to_process = None
    image_title = ""

    sample_dir = "sample_test_images"

    if input_mode == "Upload Image":
        uploaded_file = st.sidebar.file_uploader(
            "Upload seam close-up (JPG, PNG)",
            type=["jpg", "jpeg", "png", "webp"]
        )
        if uploaded_file is not None:
            image_to_process = Image.open(uploaded_file)
            image_title = uploaded_file.name
    else:
        # Check if sample test directory exists
        if os.path.exists(sample_dir):
            sample_files = [f for f in os.listdir(sample_dir) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
        else:
            sample_files = []

        if sample_files:
            selected_sample = st.sidebar.selectbox("Choose a sample defect:", sample_files)
            sample_path = os.path.join(sample_dir, selected_sample)
            image_to_process = Image.open(sample_path)
            image_title = selected_sample
        else:
            st.sidebar.info("Generate sample images using:\n`python src/synthetic_generator.py`")

    # Inspection Settings
    st.sidebar.markdown("---")
    st.sidebar.header("⚙️ Inspection Settings")
    alpha = st.sidebar.slider("Mask Overlay Opacity", min_value=0.2, max_value=0.8, value=0.45, step=0.05)
    draw_boxes = st.sidebar.checkbox("Draw Defect Bounding Boxes", value=True)

    st.sidebar.markdown("---")
    st.sidebar.caption("StitchSense v1.0 | MCA Final Project")
    st.sidebar.caption("Deployment: Windows 11 CPU (Intel Iris Xe)")

    # Main Inspection View
    if image_to_process is None:
        st.info("👈 Upload a garment seam image or select a sample from the sidebar to start inspection.")
        
        # Display Feature Overview
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("### 🔍 Pixel-Level Localization")
            st.write("Identifies precise locations of skipped stitches, seam puckering, open seams, wavy stitching, and fabric damage.")
        with col2:
            st.markdown("### 📊 Severity Analysis")
            st.write("Measures affected seam area, computes severity scores, and determines structural risk.")
        with col3:
            st.markdown("### 🪡 Tailor Repair Advisor")
            st.write("Translates AI findings into machine adjustments (needle size, tension dials, feed dogs) and repair steps.")
        return

    # Run Prediction
    with st.spinner("Analyzing garment seam with StitchSense..."):
        results = pipeline.predict(image_to_process)

    orig_img = results["original_image"]
    overlay_img = results["overlay_image"]
    diagnosis = results["diagnosis"]
    defects = results["defects"]

    # Top KPI Metrics Cards
    col_kpi1, col_kpi2, col_kpi3, col_kpi4 = st.columns(4)

    is_passed = diagnosis["status"] == "SEAM_PASSED"
    status_label = "PASSED" if is_passed else "DEFECTS FOUND"
    status_class = "status-pass" if is_passed else "status-fail"

    with col_kpi1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Quality Inspection</div>
            <div class="metric-value {status_class}">{status_label}</div>
        </div>
        """, unsafe_allow_html=True)

    with col_kpi2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Defect Instances</div>
            <div class="metric-value">{diagnosis['total_defect_count']}</div>
        </div>
        """, unsafe_allow_html=True)

    with col_kpi3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Primary Issue</div>
            <div class="metric-value" style="font-size: 1.1rem; line-height: 2rem;">{diagnosis['primary_defect']}</div>
        </div>
        """, unsafe_allow_html=True)

    with col_kpi4:
        sev_color = "#16A34A" if is_passed else ("#EA580C" if diagnosis['overall_severity'] == "Low" else "#DC2626")
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Overall Severity</div>
            <div class="metric-value" style="color: {sev_color};">{diagnosis['overall_severity']}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Dual-View Inspection Images
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("📷 Original Garment Seam")
        st.image(orig_img, use_container_width=True, caption=f"Source: {image_title}")

    with col_right:
        st.subheader("🎯 StitchSense Defect Overlay")
        st.image(overlay_img, use_container_width=True, caption="Highlighted Defect Regions & Classifications")

    st.markdown("---")

    # Tailoring Repair & Action Plan
    st.subheader("🪡 Tailoring Repair & Remediation Guide")

    if is_passed:
        st.success("✅ **No sewing defects detected!** The stitch length is uniform, tension is balanced, and no thread breakage is observed. The garment is ready for client delivery.")
    else:
        st.warning(f"⚠️ **Attention Required:** {diagnosis['summary']}")

        for idx, report in enumerate(diagnosis["defect_reports"]):
            severity_badge = f"🔴 {report['severity_label']}" if report['severity_rank'] == 3 else (
                f"🟠 {report['severity_label']}" if report['severity_rank'] == 2 else f"🟡 {report['severity_label']}"
            )

            with st.expander(f"Defect {idx+1}: {report['category']} — {severity_badge}", expanded=True):
                col_diag, col_action = st.columns([1, 1.3])

                with col_diag:
                    st.markdown("#### 🔍 Machine Diagnosis")
                    st.write(f"**Affected Seam Area:** {report['total_area_percent']}% of inspection frame")
                    st.write(f"**Structural Impact:** {report['impact']}")
                    
                    st.markdown("**Probable Root Causes:**")
                    for cause in report["root_causes"]:
                        st.write(f"- {cause}")

                with col_action:
                    st.markdown("#### 🛠️ Step-by-Step Remediation Plan")
                    for step in report["action_steps"]:
                        st.markdown(step)

                    if report.get("preventive_tips"):
                        st.info(f"💡 **Quality Assurance Tip:** {report['preventive_tips']}")

        # Exportable Alteration Ticket
        st.markdown("---")
        st.subheader("📄 Boutique Alteration Ticket")
        ticket_text = f"""
STITCHSENSE QUALITY INSPECTION REPORT
--------------------------------------
Garment Reference: {image_title}
Status: {status_label}
Defects Detected: {diagnosis['total_defect_count']}
Primary Issue: {diagnosis['primary_defect']}
Overall Severity: {diagnosis['overall_severity']}

REPAIR INSTRUCTIONS:
"""
        for r in diagnosis["defect_reports"]:
            ticket_text += f"\n* {r['category']} ({r['severity_label']}):\n"
            for step in r["action_steps"]:
                ticket_text += f"   {step}\n"

        st.download_button(
            label="📥 Download Alteration Ticket (TXT)",
            data=ticket_text,
            file_name="stitchsense_alteration_ticket.txt",
            mime="text/plain"
        )


if __name__ == "__main__":
    main()

