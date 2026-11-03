"""
Flask Application Test Suite for StitchSense.
Tests all 7 application pages, navigation routes, file upload filtering,
demo pipelines, contact form, and error handling.
"""

import os
import sys
import unittest
from io import BytesIO
from PIL import Image

# Ensure project root is on Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from flask import Flask
    from app import create_app
    from database.models import InspectionRecord
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False


class TestStitchSenseFrontend(unittest.TestCase):
    """Test cases for the complete StitchSense frontend and navigation."""

    def setUp(self):
        if not FLASK_AVAILABLE:
            self.skipTest("Flask is not installed in the current environment.")
        self.app = create_app("testing")
        self.client = self.app.test_client()
        self.ctx = self.app.app_context()
        self.ctx.push()

    def tearDown(self):
        if FLASK_AVAILABLE:
            self.ctx.pop()
            db_uri = self.app.config.get("SQLALCHEMY_DATABASE_URI", "")
            if db_uri.startswith("sqlite:///"):
                db_path = db_uri.replace("sqlite:///", "")
                if "test_" in os.path.basename(db_path) and os.path.exists(db_path):
                    try:
                        os.remove(db_path)
                    except Exception:
                        pass

    # -------------------------------------------------------------------------
    # Route Navigation Tests (All 7 Pages)
    # -------------------------------------------------------------------------
    def test_page_1_home(self):
        """Page 1: Verify Home page loads with branding, hero, and features."""
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"StitchSense", response.data)
        self.assertIn(b"Elevate Boutique Craftsmanship", response.data)
        self.assertIn(b"How StitchSense Works", response.data)

    def test_page_2_about(self):
        """Page 2: Verify About StitchSense page loads with U-Net architecture."""
        response = self.client.get("/about")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"About StitchSense", response.data)
        self.assertIn(b"SMP U-Net Architecture", response.data)
        self.assertIn(b"MobileNetV2", response.data)

    def test_page_3_upload(self):
        """Page 3: Verify Upload Studio loads with dropzone and photography tips."""
        response = self.client.get("/upload")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Upload Garment Seam", response.data)
        self.assertIn(b"Tailor Photography Best Practices", response.data)

    def test_page_4_dashboard(self):
        """Page 4: Verify Analysis Dashboard loads properly."""
        response = self.client.get("/dashboard")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Analysis Dashboard", response.data)

    def test_page_5_history(self):
        """Page 5: Verify Analysis History loads."""
        response = self.client.get("/history")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Analysis History", response.data)

    def test_page_6_repair_guide(self):
        """Page 6: Verify Tailoring Repair Guide encyclopedia loads with all 5 defect categories."""
        response = self.client.get("/repair-guide")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Tailoring Repair Guide", response.data)
        self.assertIn(b"Skipped Stitches", response.data)
        self.assertIn(b"Seam Puckering", response.data)
        self.assertIn(b"Open Seam", response.data)
        self.assertIn(b"Misaligned", response.data)
        self.assertIn(b"Needle Cut", response.data)

    def test_page_7_contact_get_and_post(self):
        """Page 7: Verify Contact page loads and processes feedback submission."""
        # GET
        get_res = self.client.get("/contact")
        self.assertEqual(get_res.status_code, 200)
        self.assertIn(b"Contact", get_res.data)
        self.assertIn(b"Consultation", get_res.data)

        # POST
        post_data = {
            "name": "Sreema Boutique",
            "email": "owner@sreema.com",
            "organization": "Sreema Tailoring Studio",
            "topic": "Boutique Consultation",
            "message": "Testing the defect detection system for bridal gowns."
        }
        post_res = self.client.post("/contact", data=post_data, follow_redirects=True)
        self.assertEqual(post_res.status_code, 200)
        self.assertIn(b"Thank you, Sreema Boutique!", post_res.data)

    # -------------------------------------------------------------------------
    # Demo & Analysis Workflow Tests
    # -------------------------------------------------------------------------
    def test_demo_skipped_stitch(self):
        """Verify demo route processes sample and renders dashboard with repair cards."""
        response = self.client.get("/demo/skipped_stitch")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Inspection Studio", response.data)
        self.assertIn(b"Machine Remediation", response.data)
        self.assertIn(b"Skipped Stitch", response.data)

    def test_upload_security_filters(self):
        """Verify non-image uploads (.exe, .txt) are rejected."""
        data = {
            "garment_image": (BytesIO(b"malicious script"), "hack.exe")
        }
        response = self.client.post("/analyze", data=data, content_type="multipart/form-data", follow_redirects=True)
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Unsupported file type", response.data)

    def test_download_alteration_ticket(self):
        """Verify download alteration ticket endpoint returns plain text work ticket."""
        rec = InspectionRecord(
            filename="demo_ticket.jpg",
            original_image_path="demo_ticket.jpg",
            overlay_image_path="overlay_demo_ticket.jpg",
            status="DEFECTS FOUND",
            primary_defect="Seam Puckering",
            defect_count=1,
            severity="Moderate",
            summary="Seam puckering along right panel.",
            report_json=[{"category": "Seam Puckering", "severity_label": "Moderate", "action_steps": ["Loosen upper tension"]}]
        )
        rec_id = rec.save()

        res = self.client.get(f"/ticket/{rec_id}")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.mimetype, "text/plain")
        self.assertIn(b"STITCHSENSE - BOUTIQUE QUALITY INSPECTION TICKET", res.data)
        self.assertIn(b"Seam Puckering", res.data)


if __name__ == "__main__":
    unittest.main()
