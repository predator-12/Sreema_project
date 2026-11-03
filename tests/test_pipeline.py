"""
Unit and integration tests for StitchSense components.
Tests configuration parsing, repair knowledge base validation,
and end-to-end fallback inference flow.
"""

import os
import sys
import json
import numpy as np

# Ensure project root is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils import load_config, set_seed
from src.repair_engine import TailoringRepairAdvisor


def test_config_structure():
    """Verify config.yaml contains all essential settings."""
    config = load_config("configs/config.yaml")
    assert "project" in config
    assert "model" in config
    assert "classes" in config
    assert config["model"]["classes"] == 6
    assert len(config["classes"]) == 6


def test_knowledge_base():
    """Verify repair_rules.json contains all 5 defect categories and valid action steps."""
    assert os.path.exists("configs/repair_rules.json")
    with open("configs/repair_rules.json", "r", encoding="utf-8") as f:
        kb = json.load(f)

    assert "defects" in kb
    for defect_id in ["1", "2", "3", "4", "5"]:
        assert defect_id in kb["defects"]
        entry = kb["defects"][defect_id]
        assert "category" in entry
        assert "root_causes" in entry
        assert len(entry["root_causes"]) > 0
        assert "tailor_actions" in entry
        assert len(entry["tailor_actions"]) > 0


def test_repair_advisor_clean_seam():
    """Verify advisor produces PASSED verdict when no defect instances exist."""
    advisor = TailoringRepairAdvisor("configs/repair_rules.json")
    result = advisor.diagnose_and_recommend([])

    assert result["status"] == "SEAM_PASSED"
    assert result["total_defect_count"] == 0
    assert result["overall_severity"] == "Clean"


def test_repair_advisor_defect_diagnosis():
    """Verify advisor correctly identifies defect and assigns remediation steps."""
    advisor = TailoringRepairAdvisor("configs/repair_rules.json")
    dummy_defect = [{
        "class_id": 1,
        "defect_name": "Broken / Skipped Stitch",
        "bbox": [100, 150, 40, 80],
        "centroid": [120, 190],
        "area_px": 1200,
        "area_percent": 4.5
    }]

    result = advisor.diagnose_and_recommend(dummy_defect)
    assert result["status"] == "DEFECT_FOUND"
    assert result["total_defect_count"] == 1
    assert result["primary_defect"] == "Broken / Skipped Stitch"
    assert len(result["defect_reports"]) == 1
    assert len(result["defect_reports"][0]["action_steps"]) > 0


if __name__ == "__main__":
    print("Running StitchSense unit tests...")
    test_config_structure()
    print("  [PASS] test_config_structure passed.")
    test_knowledge_base()
    print("  [PASS] test_knowledge_base passed.")
    test_repair_advisor_clean_seam()
    print("  [PASS] test_repair_advisor_clean_seam passed.")
    test_repair_advisor_defect_diagnosis()
    print("  [PASS] test_repair_advisor_defect_diagnosis passed.")
    print("\nAll StitchSense core tests passed successfully!")
