"""Compare current 14-class, six-class auxiliary, and hybrid GCD systems."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from text_normalization import normalize_grievance_text

TEST = ROOT / "evaluation" / "india_gcd_auxiliary_split.csv"
CURRENT = ROOT / "models" / "india_departments"
AUXILIARY = ROOT / "models" / "india_gcd_auxiliary"
OUTPUT = ROOT / "evaluation" / "india_gcd_system_comparison.json"

CLASSES = [
    "Water Supply & Sewerage",
    "Electricity & Power",
    "Roads & Transport",
    "Public Health",
    "Environment & Pollution",
    "Public Distribution System",
]
RULES = {
    "Water Supply & Sewerage": (
        "water supply", "water", "पानी", "नल", "पाइपलाइन", "टैंकर", "जल",
    ),
    "Electricity & Power": (
        "electricity", "power", "transformer", "बिजली", "ट्रांसफार्मर",
        "स्ट्रीट लाइट", "मीटर",
    ),
    "Roads & Transport": (
        "road", "pothole", "bus", "traffic", "transport", "सड़क", "बस",
        "ट्रैफिक", "गड्ढे", "ऑटो",
    ),
    "Public Health": (
        "hospital", "doctor", "medicine", "ambulance", "clinic", "health",
        "अस्पताल", "डॉक्टर", "दवा", "एम्बुलेंस", "स्वास्थ्य", "टीके",
    ),
    "Environment & Pollution": (
        "garbage", "waste", "drain", "mosquito", "sanitation", "कचरा",
        "नाला", "मच्छर", "सफाई", "बदबू", "कूड़ेदान",
    ),
    "Public Distribution System": (
        "ration", "aadhar", "aadhaar", "food grain", "राशन", "आधार",
        "अनाज", "बायोमेट्रिक", "डीलर",
    ),
}


def score(expected: pd.Series, predicted: pd.Series) -> dict:
    return {
        "rows": int(len(expected)),
        "accuracy": float(accuracy_score(expected, predicted)),
        "macro_f1_six_class": float(
            f1_score(expected, predicted, labels=CLASSES, average="macro")
        ),
        "classification_report": classification_report(
            expected,
            predicted,
            labels=CLASSES,
            target_names=CLASSES,
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(
            expected, predicted, labels=CLASSES
        ).tolist(),
    }


def hybrid_predict(texts: pd.Series, auxiliary_predictions: pd.Series) -> list[str]:
    predictions = []
    for text, fallback in zip(texts, auxiliary_predictions):
        normalized = normalize_grievance_text(text).casefold()
        matches = [
            label
            for label, terms in RULES.items()
            if any(term.casefold() in normalized for term in terms)
        ]
        predictions.append(matches[0] if len(matches) == 1 else fallback)
    return predictions


def main() -> None:
    frame = pd.read_csv(TEST)
    expected = frame["expected_department"]
    current_pipeline = joblib.load(CURRENT / "pipeline.joblib")
    current_encoder = joblib.load(CURRENT / "label_encoder.joblib")
    auxiliary_pipeline = joblib.load(AUXILIARY / "pipeline.joblib")
    auxiliary_encoder = joblib.load(AUXILIARY / "label_encoder.joblib")
    texts = frame["grievance_text"].map(normalize_grievance_text)
    current = pd.Series(
        current_encoder.inverse_transform(current_pipeline.predict(texts)),
        index=frame.index,
    )
    auxiliary = pd.Series(
        auxiliary_encoder.inverse_transform(auxiliary_pipeline.predict(texts)),
        index=frame.index,
    )
    hybrid = pd.Series(hybrid_predict(texts, auxiliary), index=frame.index)
    systems = {}
    for name, predictions in (
        ("current_14_class_restricted", current),
        ("six_class_auxiliary", auxiliary),
        ("hybrid_rules_plus_auxiliary", hybrid),
    ):
        systems[name] = {
            "overall": score(expected, predictions),
            "by_language": {
                language: score(
                    group["mapped_department"],
                    predictions.loc[group.index],
                )
                for language, group in frame.groupby("language", sort=True)
            },
        }
    result = {
        "dataset": "GCD assistant-approved evaluation split",
        "rows": int(len(frame)),
        "classes": CLASSES,
        "ambiguous_rows_excluded": 12,
        "split_grouping": "source_group; no source/template group crosses train and test",
        "systems": systems,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
