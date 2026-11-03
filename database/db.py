"""
Database connection and lifecycle manager.
Implements robust SQLite database management with zero external dependencies,
ensuring seamless compatibility across all Python 3.12 installations.
"""

import os
import sqlite3
from flask import g, current_app


def get_db_path():
    """Retrieve database path from Flask configuration or default location."""
    if current_app:
        uri = current_app.config.get("SQLALCHEMY_DATABASE_URI", "")
        if uri.startswith("sqlite:///"):
            return uri.replace("sqlite:///", "")
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    return os.path.join(base_dir, "database", "stitchsense.db")


def get_db():
    """Get active database connection for the current Flask request context."""
    if "db" not in g:
        db_path = get_db_path()
        if db_path != ":memory:":
            dir_name = os.path.dirname(db_path)
            if dir_name:
                os.makedirs(dir_name, exist_ok=True)
        g.db = sqlite3.connect(db_path)
        g.db.row_factory = sqlite3.Row
    return g.db


def close_db(e=None):
    """Close active database connection on request teardown."""
    db = g.pop("db", None)
    if db is not None:
        db.close()


def init_db(app=None):
    """Initialize database tables."""
    db_path = get_db_path() if not app else (
        app.config.get("SQLALCHEMY_DATABASE_URI", "").replace("sqlite:///", "")
        if app.config.get("SQLALCHEMY_DATABASE_URI", "").startswith("sqlite:///")
        else os.path.join(app.config.get("BASE_DIR", os.path.dirname(os.path.abspath(__file__))), "database", "stitchsense.db")
    )
    if db_path != ":memory:":
        dir_name = os.path.dirname(db_path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS inspection_records (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        filename TEXT NOT NULL,
        original_image_path TEXT NOT NULL,
        overlay_image_path TEXT NOT NULL,
        status TEXT NOT NULL,
        primary_defect TEXT NOT NULL,
        defect_count INTEGER NOT NULL DEFAULT 0,
        severity TEXT NOT NULL DEFAULT 'Clean',
        summary TEXT,
        report_json TEXT
    );
    """)
    conn.commit()
    conn.close()
