"""
Data models and query helpers for StitchSense inspection records.
"""

import json
from datetime import datetime, timezone
from database.db import get_db


class InspectionRecord:
    """Represents a garment seam defect inspection record."""

    def __init__(
        self,
        id=None,
        timestamp=None,
        filename="",
        original_image_path="",
        overlay_image_path="",
        status="PASSED",
        primary_defect="None",
        defect_count=0,
        severity="Clean",
        summary="",
        report_json="{}"
    ):
        self.id = id
        self.timestamp = timestamp or datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        self.filename = filename
        self.original_image_path = original_image_path
        self.overlay_image_path = overlay_image_path
        self.status = status
        self.primary_defect = primary_defect
        self.defect_count = defect_count
        self.severity = severity
        self.summary = summary
        self.report_json = report_json if isinstance(report_json, str) else json.dumps(report_json)

    def save(self):
        """Persist inspection record to SQLite database."""
        db = get_db()
        cursor = db.cursor()
        cursor.execute("""
            INSERT INTO inspection_records (
                timestamp, filename, original_image_path, overlay_image_path,
                status, primary_defect, defect_count, severity, summary, report_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            self.timestamp, self.filename, self.original_image_path, self.overlay_image_path,
            self.status, self.primary_defect, self.defect_count, self.severity, self.summary, self.report_json
        ))
        db.commit()
        self.id = cursor.lastrowid
        return self.id

    @classmethod
    def get_by_id(cls, record_id):
        """Retrieve inspection record by primary key."""
        db = get_db()
        row = db.execute("SELECT * FROM inspection_records WHERE id = ?", (record_id,)).fetchone()
        if row:
            return cls(
                id=row["id"],
                timestamp=row["timestamp"],
                filename=row["filename"],
                original_image_path=row["original_image_path"],
                overlay_image_path=row["overlay_image_path"],
                status=row["status"],
                primary_defect=row["primary_defect"],
                defect_count=row["defect_count"],
                severity=row["severity"],
                summary=row["summary"],
                report_json=row["report_json"]
            )
        return None

    @classmethod
    def get_recent(cls, limit=10):
        """Retrieve recent inspection records."""
        db = get_db()
        rows = db.execute(
            "SELECT * FROM inspection_records ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        return [
            cls(
                id=r["id"],
                timestamp=r["timestamp"],
                filename=r["filename"],
                original_image_path=r["original_image_path"],
                overlay_image_path=r["overlay_image_path"],
                status=r["status"],
                primary_defect=r["primary_defect"],
                defect_count=r["defect_count"],
                severity=r["severity"],
                summary=r["summary"],
                report_json=r["report_json"]
            ) for r in rows
        ]

    def get_parsed_report(self):
        """Return deserialized diagnosis report dict."""
        try:
            return json.loads(self.report_json)
        except Exception:
            return {}
