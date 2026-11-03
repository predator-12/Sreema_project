"""
Repair Assistant Engine for StitchSense.
Connects computer vision segmentation outputs with tailoring domain expertise.
Translates detected defects and severity scores into practical machine adjustments and repair instructions.
"""

import os
import json


class TailoringRepairAdvisor:
    """
    Expert system for garment defect analysis and machine remediation.
    """

    def __init__(self, knowledge_base_path: str = "configs/repair_rules.json"):
        self.kb_path = knowledge_base_path
        self.rules = self._load_rules()

    def _load_rules(self) -> dict:
        if not os.path.exists(self.kb_path):
            raise FileNotFoundError(f"Knowledge base file not found at: {self.kb_path}")
        with open(self.kb_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def diagnose_and_recommend(self, detected_instances: list[dict]) -> dict:
        """
        Analyze detected defects and generate a comprehensive tailor recommendation report.

        Args:
            detected_instances: List of defect dicts from postprocess.extract_defect_instances.

        Returns:
            dict containing:
                - status: "DEFECT_FOUND" or "SEAM_PASSED"
                - total_defect_count: int
                - primary_defect: str
                - overall_severity: str (Clean, Low, Moderate, High)
                - defect_reports: list of detailed repair advice per category
        """
        if not detected_instances:
            return {
                "status": "SEAM_PASSED",
                "total_defect_count": 0,
                "primary_defect": "None",
                "overall_severity": "Clean",
                "summary": "Garment inspection passed. Stitch line is continuous, balanced, and free of visible defects.",
                "defect_reports": []
            }

        defect_rules = self.rules.get("defects", {})
        
        # Group instances by class_id
        grouped = {}
        for inst in detected_instances:
            cid = str(inst["class_id"])
            if cid not in grouped:
                grouped[cid] = []
            grouped[cid].append(inst)

        defect_reports = []
        max_severity_rank = 0  # 1: minor, 2: moderate, 3: severe

        for cid, instances in grouped.items():
            rule = defect_rules.get(cid, {})
            category_name = rule.get("category", f"Defect Class {cid}")
            
            # Compute total area percentage for this defect category
            total_pct = sum(item["area_percent"] for item in instances)
            total_px = sum(item["area_px"] for item in instances)
            count = len(instances)

            # Determine severity level
            severity_cfg = rule.get("severity_levels", {})
            minor_thresh = severity_cfg.get("minor", {}).get("threshold_percent", 3.0)
            mod_thresh = severity_cfg.get("moderate", {}).get("threshold_percent", 10.0)

            if total_pct <= minor_thresh:
                sev_key = "minor"
                rank = 1
            elif total_pct <= mod_thresh:
                sev_key = "moderate"
                rank = 2
            else:
                sev_key = "severe"
                rank = 3

            max_severity_rank = max(max_severity_rank, rank)
            sev_info = severity_cfg.get(sev_key, {})

            report = {
                "class_id": int(cid),
                "category": category_name,
                "short_name": rule.get("short_name", category_name),
                "instance_count": count,
                "total_area_percent": round(total_pct, 2),
                "severity_label": sev_info.get("label", sev_key.capitalize()),
                "severity_rank": rank,
                "impact": sev_info.get("impact", "Quality variance on seam."),
                "root_causes": rule.get("root_causes", []),
                "action_steps": rule.get("tailor_actions", []),
                "preventive_tips": rule.get("preventive_tips", "")
            }
            defect_reports.append(report)

        # Sort reports by severity rank descending
        defect_reports.sort(key=lambda r: (r["severity_rank"], r["total_area_percent"]), reverse=True)
        primary_defect = defect_reports[0]["category"]

        severity_map = {1: "Low", 2: "Moderate", 3: "High"}
        overall_severity = severity_map.get(max_severity_rank, "Moderate")

        return {
            "status": "DEFECT_FOUND",
            "total_defect_count": len(detected_instances),
            "primary_defect": primary_defect,
            "overall_severity": overall_severity,
            "summary": f"Detected {len(detected_instances)} defect instance(s). Primary issue: {primary_defect} ({overall_severity} severity).",
            "defect_reports": defect_reports
        }

