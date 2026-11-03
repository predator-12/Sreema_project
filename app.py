"""
StitchSense Flask Application Factory & Core Controllers.
Implements secure endpoints for garment image analysis, defect segmentation,
tailoring repair guidance, and inspection history tracking.
"""

import os
import shutil
from flask import (
    Flask, render_template, request, redirect, url_for, flash,
    send_file, jsonify, Response
)

from config import config_by_name
from database.db import init_db, close_db
from database.models import InspectionRecord
from image_processing.preprocessor import validate_and_save_upload
from ai_models.model_service import ModelService


def create_app(config_name: str = "development") -> Flask:
    """
    Application factory pattern for StitchSense.

    Args:
        config_name: Configuration mode ('development', 'production', 'testing').

    Returns:
        Configured Flask WSGI application instance.
    """
    app = Flask(__name__, template_folder="templates", static_folder="static")

    # Load configuration
    cfg = config_by_name.get(config_name, config_by_name["default"])
    app.config.from_object(cfg)

    # Ensure required runtime folders exist
    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
    os.makedirs(os.path.join(app.root_path, "database"), exist_ok=True)

    # Initialize SQLite database
    init_db(app)

    # Register database connection teardown
    app.teardown_appcontext(close_db)

    # Initialize AI Model Service
    model_service = ModelService.get_instance(app.config)

    # =========================================================================
    # Error Handlers
    # =========================================================================
    @app.errorhandler(400)
    def bad_request_error(e):
        if request.path.startswith("/api/"):
            return jsonify({"error": "Bad request", "message": str(e)}), 400
        return render_template("errors/400.html", error_message=str(e)), 400

    @app.errorhandler(404)
    def not_found_error(e):
        if request.path.startswith("/api/"):
            return jsonify({"error": "Resource not found"}), 404
        return render_template("errors/404.html"), 404

    @app.errorhandler(413)
    def file_too_large_error(e):
        if request.path.startswith("/api/"):
            return jsonify({"error": "File exceeds maximum size limit (16MB)"}), 413
        return render_template("errors/413.html"), 413

    @app.errorhandler(500)
    def internal_server_error(e):
        if request.path.startswith("/api/"):
            return jsonify({"error": "Internal server error"}), 500
        return render_template("errors/500.html"), 500

    # =========================================================================
    # Application Routes
    # =========================================================================
    @app.route("/", methods=["GET"])
    def index():
        """Homepage: Displays hero banner, quick demo chips, how it works & features."""
        return render_template("index.html")

    @app.route("/about", methods=["GET"])
    def about():
        """About Page: Project background, U-Net architecture & MCA credentials."""
        return render_template("about.html")

    @app.route("/upload", methods=["GET"])
    def upload_page():
        """Upload Page: Dedicated studio upload dropzone & tailor photography tips."""
        return render_template("upload.html")

    @app.route("/dashboard", methods=["GET"])
    def dashboard():
        """Analysis Dashboard: Active inspection view with dual viewer and repair plan."""
        # Check if there is a recent inspection record to display
        recent_records = InspectionRecord.get_recent(limit=1)
        if recent_records:
            latest = recent_records[0]
            result_data = {
                "status": latest.status,
                "is_passed": latest.status == "PASSED",
                "total_defect_count": latest.defect_count,
                "primary_defect": latest.primary_defect,
                "overall_severity": latest.severity,
                "summary": latest.summary,
                "defect_reports": latest.get_parsed_report(),
                "overlay_filename": latest.overlay_image_path
            }
            return render_template(
                "dashboard.html",
                result=result_data,
                record=latest,
                original_filename=latest.original_image_path
            )
        return render_template("dashboard.html", result=None, record=None)

    @app.route("/repair-guide", methods=["GET"])
    def repair_guide():
        """Repair Guide: Comprehensive tailoring encyclopedia for all defect classes."""
        return render_template("repair_guide.html")

    @app.route("/contact", methods=["GET", "POST"])
    def contact():
        """Contact Page: Boutique inquiry & academic feedback form."""
        if request.method == "POST":
            name = request.form.get("name", "Tailor / Boutique User")
            topic = request.form.get("topic", "General Inquiry")
            flash(f"Thank you, {name}! Your inquiry regarding '{topic}' has been received successfully.", "success")
            return redirect(url_for("contact"))
        return render_template("contact.html")

    @app.route("/analyze", methods=["POST"])
    def analyze():
        """Process an uploaded garment image and display defect diagnosis."""
        if "garment_image" not in request.files:
            flash("No file part in the request.", "danger")
            return redirect(url_for("upload_page"))

        file = request.files["garment_image"]
        if file.filename == "":
            flash("Please select an image file to upload.", "warning")
            return redirect(url_for("upload_page"))

        # Securely validate and persist the uploaded image
        success, saved_filename, error_msg = validate_and_save_upload(
            file_storage=file,
            upload_folder=app.config["UPLOAD_FOLDER"],
            allowed_extensions=app.config["ALLOWED_EXTENSIONS"]
        )

        if not success:
            flash(error_msg, "danger")
            return redirect(url_for("upload_page"))

        image_path = os.path.join(app.config["UPLOAD_FOLDER"], saved_filename)

        try:
            # Perform AI Defect Segmentation & Remediation Diagnosis
            analysis_results = model_service.analyze_image(
                image_path=image_path,
                uploads_dir=app.config["UPLOAD_FOLDER"]
            )

            # Persist audit record in SQLite database
            record = InspectionRecord(
                filename=saved_filename,
                original_image_path=saved_filename,
                overlay_image_path=analysis_results["overlay_filename"],
                status=analysis_results["status"],
                primary_defect=analysis_results["primary_defect"],
                defect_count=analysis_results["total_defect_count"],
                severity=analysis_results["overall_severity"],
                summary=analysis_results["summary"],
                report_json=analysis_results["defect_reports"]
            )
            record.save()

            return render_template(
                "dashboard.html",
                result=analysis_results,
                record=record,
                original_filename=saved_filename
            )

        except Exception as e:
            flash(f"Analysis failed: {str(e)}", "danger")
            return redirect(url_for("upload_page"))

    @app.route("/demo/<defect_type>", methods=["GET"])
    def demo(defect_type: str):
        """Analyze pre-loaded boutique demo samples for instant demonstration."""
        sample_dir = os.path.join(app.root_path, "sample_test_images")
        sample_filename = f"demo_{defect_type}.jpg"
        source_path = os.path.join(sample_dir, sample_filename)

        if not os.path.exists(source_path):
            flash(f"Sample '{defect_type}' not found.", "warning")
            return redirect(url_for("upload_page"))

        # Copy sample into uploads folder for isolated processing
        dest_filename = f"sample_{sample_filename}"
        dest_path = os.path.join(app.config["UPLOAD_FOLDER"], dest_filename)
        shutil.copyfile(source_path, dest_path)

        analysis_results = model_service.analyze_image(
            image_path=dest_path,
            uploads_dir=app.config["UPLOAD_FOLDER"]
        )

        record = InspectionRecord(
            filename=f"[DEMO] {defect_type.replace('_', ' ').title()}",
            original_image_path=dest_filename,
            overlay_image_path=analysis_results["overlay_filename"],
            status=analysis_results["status"],
            primary_defect=analysis_results["primary_defect"],
            defect_count=analysis_results["total_defect_count"],
            severity=analysis_results["overall_severity"],
            summary=analysis_results["summary"],
            report_json=analysis_results["defect_reports"]
        )
        record.save()

        return render_template(
            "dashboard.html",
            result=analysis_results,
            record=record,
            original_filename=dest_filename
        )

    @app.route("/history", methods=["GET"])
    def history():
        """View historical garment inspection audits."""
        records = InspectionRecord.get_recent(limit=25)
        return render_template("history.html", records=records)

    @app.route("/record/<int:record_id>", methods=["GET"])
    def view_record(record_id: int):
        """View a specific historical inspection report in the dashboard."""
        record = InspectionRecord.get_by_id(record_id)
        if not record:
            flash(f"Inspection record #{record_id} not found.", "warning")
            return redirect(url_for("history"))

        result_data = {
            "status": record.status,
            "is_passed": record.status == "PASSED",
            "total_defect_count": record.defect_count,
            "primary_defect": record.primary_defect,
            "overall_severity": record.severity,
            "summary": record.summary,
            "defect_reports": record.get_parsed_report(),
            "overlay_filename": record.overlay_image_path
        }

        return render_template(
            "dashboard.html",
            result=result_data,
            record=record,
            original_filename=record.original_image_path
        )

    @app.route("/ticket/<int:record_id>", methods=["GET"])
    def download_ticket(record_id: int):
        """Generate and download a printable tailoring work order ticket."""
        record = InspectionRecord.get_by_id(record_id)
        if not record:
            flash("Record not found.", "danger")
            return redirect(url_for("history"))

        parsed_reports = record.get_parsed_report()
        ticket_content = f"""=======================================================
STITCHSENSE - BOUTIQUE QUALITY INSPECTION TICKET
=======================================================
Ticket ID:        #{record.id}
Date & Time:      {record.timestamp}
Garment File:     {record.filename}
Quality Status:   {record.status}
Primary Defect:   {record.primary_defect}
Severity Rating:  {record.severity}
Defect Count:     {record.defect_count}

SUMMARY:
{record.summary}

-------------------------------------------------------
TAILOR REMEDIATION INSTRUCTIONS:
-------------------------------------------------------
"""
        if not parsed_reports:
            ticket_content += "No alterations required. Garment approved for delivery.\n"
        else:
            for rep in parsed_reports:
                ticket_content += f"\n[{rep.get('category', 'Defect')}] - {rep.get('severity_label', 'Issue')}\n"
                ticket_content += f"Structural Impact: {rep.get('impact', 'N/A')}\n"
                ticket_content += "Required Actions:\n"
                for step in rep.get("action_steps", []):
                    ticket_content += f"  * {step}\n"
                if rep.get("preventive_tips"):
                    ticket_content += f"Quality Tip: {rep.get('preventive_tips')}\n"

        ticket_content += "\n=======================================================\n"
        ticket_content += "StitchSense AI Assistant | MCA Capstone Project\n"

        return Response(
            ticket_content,
            mimetype="text/plain",
            headers={"Content-Disposition": f"attachment;filename=alteration_ticket_{record.id}.txt"}
        )

    return app


# Default app instance for standard WSGI servers
app = create_app(os.environ.get("FLASK_ENV", "development"))

