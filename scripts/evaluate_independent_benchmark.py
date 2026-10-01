"""Evaluate the three routing systems on independently approved benchmark rows."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import joblib
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_gcd_systems import CLASSES, hybrid_predict, score
from text_normalization import normalize_grievance_text

BENCHMARK = ROOT / "data" / "evaluation" / "india_hindi_benchmark_candidates.csv"
CURRENT = ROOT / "models" / "india_departments"
AUXILIARY = ROOT / "models" / "india_gcd_auxiliary"
OUTPUT = ROOT / "evaluation" / "india_independent_benchmark_metrics.json"


def main() -> None:
    frame = pd.read_csv(BENCHMARK)
    approved = frame[
        frame["independent_review_status"].eq("independently_approved")
        & frame["final_department"].isin(CLASSES)
    ].copy()
    result = {
        "dataset": "independent six-class Hindi/English/Hinglish benchmark",
        "candidate_rows": int(len(frame)),
        "independently_approved_rows": int(len(approved)),
        "required_action": (
            "Independent reviewer must approve rows and populate final_department "
            "before benchmark metrics are production-decision evidence."
        ),
    }
    if approved.empty:
        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        OUTPUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result, indent=2))
        return

    current_pipeline = joblib.load(CURRENT / "pipeline.joblib")
    current_encoder = joblib.load(CURRENT / "label_encoder.joblib")
    auxiliary_pipeline = joblib.load(AUXILIARY / "pipeline.joblib")
    auxiliary_encoder = joblib.load(AUXILIARY / "label_encoder.joblib")
    texts = approved["grievance_text"].map(normalize_grievance_text)
    predictions = {
        "current_14_class_restricted": current_encoder.inverse_transform(
            current_pipeline.predict(texts)
        ),
        "six_class_auxiliary": auxiliary_encoder.inverse_transform(
            auxiliary_pipeline.predict(texts)
        ),
    }
    predictions["hybrid_rules_plus_auxiliary"] = hybrid_predict(
        texts, pd.Series(predictions["six_class_auxiliary"], index=approved.index)
    )
    expected = approved["final_department"]
    for name, predicted in predictions.items():
        predicted = pd.Series(predicted, index=approved.index)
        systems = {
            "overall": score(expected, predicted),
            "confusion_matrix": confusion_matrix(
                expected, predicted, labels=CLASSES
            ).tolist(),
            "by_language": {
                language: score(
                    group["final_department"], predicted.loc[group.index]
                )
                for language, group in approved.groupby("language", sort=True)
            },
        }
        result.setdefault("systems", {})[name] = systems
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
