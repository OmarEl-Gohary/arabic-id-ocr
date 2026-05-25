"""Data drift detection for the OCR pipeline using Evidently."""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def build_prediction_dataframe(predictions: List[Dict]) -> pd.DataFrame:
    """Convert a list of OCR results to a flat DataFrame for drift analysis."""
    rows = []
    for pred in predictions:
        row = {
            "id_side": pred.get("id_side", "unknown"),
            "num_fields_detected": pred.get("metadata", {}).get("num_fields_detected", 0),
            "processing_time_s": pred.get("metadata", {}).get("processing_time_s", 0),
        }
        for field_name, value in pred.get("fields", {}).items():
            row[f"field_{field_name}_present"] = 1 if value else 0
            row[f"field_{field_name}_length"] = len(value) if value else 0
        rows.append(row)
    return pd.DataFrame(rows)


class DriftDetector:
    """
    Detects data drift between a reference prediction set (production baseline)
    and the current prediction window.
    Requires: pip install evidently
    """

    def __init__(self, reference_path: Optional[str] = None):
        self.reference_df: Optional[pd.DataFrame] = None
        if reference_path and Path(reference_path).exists():
            self.load_reference(reference_path)

    def load_reference(self, path: str):
        self.reference_df = pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path)
        logger.info(f"Loaded reference dataset with {len(self.reference_df)} rows")

    def save_reference(self, predictions: List[Dict], path: str):
        df = build_prediction_dataframe(predictions)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)
        self.reference_df = df
        logger.info(f"Saved reference dataset ({len(df)} rows) to {path}")

    def check_drift(self, current_predictions: List[Dict]) -> Dict:
        if self.reference_df is None:
            return {"error": "No reference dataset loaded. Call save_reference() first."}

        try:
            from evidently.metric_preset import DataDriftPreset
            from evidently.report import Report
        except ImportError:
            return {"error": "Install evidently: pip install evidently"}

        current_df = build_prediction_dataframe(current_predictions)

        # Align columns between reference and current
        common_cols = list(set(self.reference_df.columns) & set(current_df.columns))
        ref = self.reference_df[common_cols]
        cur = current_df[common_cols]

        report = Report(metrics=[DataDriftPreset()])
        report.run(reference_data=ref, current_data=cur)

        result_dict = report.as_dict()
        drift_summary = {
            "timestamp": datetime.utcnow().isoformat(),
            "reference_size": len(ref),
            "current_size": len(cur),
            "drift_detected": False,
            "drifted_columns": [],
        }

        for metric in result_dict.get("metrics", []):
            if metric.get("metric") == "DatasetDriftMetric":
                r = metric.get("result", {})
                drift_summary["drift_detected"] = r.get("dataset_drift", False)
                drift_summary["drift_share"] = r.get("drift_share", 0)
                drift_summary["drifted_columns"] = r.get("number_of_drifted_columns", 0)

        return drift_summary

    def simple_stat_check(self, current_predictions: List[Dict]) -> Dict:
        """Lightweight statistical check without Evidently dependency."""
        if self.reference_df is None:
            return {"error": "No reference dataset loaded"}

        current_df = build_prediction_dataframe(current_predictions)
        issues = []

        for col in ["num_fields_detected", "processing_time_s"]:
            if col not in self.reference_df.columns or col not in current_df.columns:
                continue
            ref_mean = self.reference_df[col].mean()
            cur_mean = current_df[col].mean()
            if ref_mean > 0:
                change_pct = abs(cur_mean - ref_mean) / ref_mean * 100
                if change_pct > 20:
                    issues.append(f"{col} changed by {change_pct:.1f}% (ref={ref_mean:.2f}, cur={cur_mean:.2f})")

        presence_cols = [c for c in self.reference_df.columns if c.startswith("field_") and c.endswith("_present")]
        for col in presence_cols:
            if col not in current_df.columns:
                continue
            ref_rate = self.reference_df[col].mean()
            cur_rate = current_df[col].mean()
            if ref_rate > 0 and abs(cur_rate - ref_rate) > 0.15:
                issues.append(f"{col}: presence rate changed from {ref_rate:.2f} to {cur_rate:.2f}")

        return {
            "timestamp": datetime.utcnow().isoformat(),
            "issues_found": len(issues),
            "issues": issues,
            "drift_suspected": len(issues) > 0,
        }
