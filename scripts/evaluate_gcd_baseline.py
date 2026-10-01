"""Evaluate the current India routing model on the separate GCD review queue."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "evaluation" / "india_gcd_reviewed.csv"
MODEL = ROOT / "models" / "india_departments"
OUTPUT = ROOT / "evaluation" / "india_gcd_baseline_metrics.json"


def evaluate(frame: pd.DataFrame, pipeline, encoder) -> dict:
    expected = frame["mapped_department"]
    predicted = encoder.inverse_transform(pipeline.predict(frame["grievance_text"]))
    return {
        "rows": int(len(frame)),
        "accuracy": float(accuracy_score(expected, predicted)),
        "macro_f1": float(f1_score(expected, predicted, average="macro")),
        "classification_report": classification_report(
            expected,
            predicted,
            labels=encoder.classes_,
            target_names=encoder.classes_,
            output_dict=True,
            zero_division=0,
        ),
    }


def main() -> None:
    frame = pd.read_csv(DATA)
    approved = frame[
        frame["review_status"] == "assistant_reviewed_approved"
    ].copy()
    pipeline = joblib.load(MODEL / "pipeline.joblib")
    encoder = joblib.load(MODEL / "label_encoder.joblib")
    metrics = {
        "dataset": "GCD Government Complaints Dataset",
        "dataset_role": "separate synthetic TTS baseline; not merged into training",
        "review_status": "assistant_reviewed_approved_only",
        "total_rows_before_review_filter": int(len(frame)),
        "rows_excluded_for_second_review": int(len(frame) - len(approved)),
        "mapping": "source_category_to_project_department",
        "overall": evaluate(approved, pipeline, encoder),
        "by_language": {
            language: evaluate(group, pipeline, encoder)
            for language, group in approved.groupby("language", sort=True)
        },
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
