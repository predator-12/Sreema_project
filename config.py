"""
StitchSense Application Configuration.
Defines environment-specific configurations (Development, Production, Testing)
with strict security settings for file uploads and database management.
Compatible with Python 3.12.5.
"""

import os
from datetime import timedelta

BASE_DIR = os.path.abspath(os.path.dirname(__file__))


class BaseConfig:
    """Base configuration with default settings."""
    # Security
    SECRET_KEY = os.environ.get("SECRET_KEY", "stitchsense-secret-key-2026-mca-project")
    
    # File Upload Configuration
    UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB max upload size
    ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}
    
    # Database
    SQLALCHEMY_DATABASE_URI = os.environ.get(
        "DATABASE_URL",
        f"sqlite:///{os.path.join(BASE_DIR, 'database', 'stitchsense.db')}"
    )
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    # Model & AI Inference
    MODEL_CONFIG_PATH = os.path.join(BASE_DIR, "configs", "config.yaml")
    REPAIR_RULES_PATH = os.path.join(BASE_DIR, "configs", "repair_rules.json")
    MODEL_WEIGHTS_PATH = os.path.join(BASE_DIR, "weights", "stitchsense_best.pth")
    
    # CPU Hardware Optimization
    TORCH_NUM_THREADS = int(os.environ.get("TORCH_NUM_THREADS", 4))
    
    # Session Configuration
    PERMANENT_SESSION_LIFETIME = timedelta(hours=2)


class DevelopmentConfig(BaseConfig):
    """Development environment configuration."""
    DEBUG = True
    TESTING = False
    ENV = "development"


class ProductionConfig(BaseConfig):
    """Production environment configuration with hardened security."""
    DEBUG = False
    TESTING = False
    ENV = "production"
    # Ensure SECRET_KEY is set via environment variable in production
    SECRET_KEY = os.environ.get("SECRET_KEY")
    SESSION_COOKIE_SECURE = True
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = "Lax"


class TestingConfig(BaseConfig):
    """Testing environment configuration."""
    DEBUG = False
    TESTING = True
    ENV = "testing"
    SQLALCHEMY_DATABASE_URI = f"sqlite:///{os.path.join(BASE_DIR, 'database', 'test_stitchsense.db')}"
    WTF_CSRF_ENABLED = False


# Configuration dictionary lookup
config_by_name = {
    "development": DevelopmentConfig,
    "production": ProductionConfig,
    "testing": TestingConfig,
    "default": DevelopmentConfig
}
