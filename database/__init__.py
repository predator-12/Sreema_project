"""
Database package for StitchSense.
Provides data persistence for garment inspection history and tailoring tickets.
"""

from database.db import get_db, init_db
from database.models import InspectionRecord

__all__ = ["get_db", "init_db", "InspectionRecord"]

