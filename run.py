"""
StitchSense Application Entry Point.
Launches the Flask development server with configuration and environment checks.
"""

import os
import sys

# Ensure root directory is on Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from app import create_app


def main():
    env_mode = os.environ.get("FLASK_ENV", "development")
    host = os.environ.get("FLASK_HOST", "127.0.0.1")
    port = int(os.environ.get("FLASK_PORT", 5000))
    debug = env_mode == "development"

    app = create_app(env_mode)

    print("=" * 65)
    print(" 🧵 StitchSense - AI Sewing Defect Detection & Repair Assistant")
    print("=" * 65)
    print(f" * Environment : {env_mode.capitalize()}")
    print(f" * Local Server: http://{host}:{port}")
    print(f" * Hardware    : Intel Iris Xe CPU Execution")
    print(f" * Database    : SQLite ({app.config.get('SQLALCHEMY_DATABASE_URI')})")
    print(" * Press CTRL+C to stop the application")
    print("=" * 65)

    app.run(host=host, port=port, debug=debug)


if __name__ == "__main__":
    main()

