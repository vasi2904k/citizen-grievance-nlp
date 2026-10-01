"""Train the India-oriented department routing model on curated examples.

The current NYC 311 artifacts remain available as a separate variant. This
model is intentionally trained on authored India-specific examples until an
independently labelled Indian public-grievance dataset is available.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import LabelEncoder

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "evaluation" / "india_department_examples.csv"
MODEL_DIR = ROOT / "models" / "india_departments"
METRICS = ROOT / "evaluation" / "india_department_metrics.json"
HOLDOUT = ROOT / "evaluation" / "india_department_holdout.csv"


def main() -> None:
    df = pd.read_csv(DATA).dropna(subset=["grievance_text", "expected_department"])
    df["grievance_text"] = df["grievance_text"].astype(str).str.strip()
    df["expected_department"] = df["expected_department"].astype(str).str.strip()
    if df["expected_department"].nunique() < 10:
        raise ValueError("The India taxonomy must contain at least 10 departments.")

    train, test = train_test_split(
        df,
        test_size=0.25,
        random_state=42,
        stratify=df["expected_department"],
    )
    encoder = LabelEncoder()
    y_train = encoder.fit_transform(train["expected_department"])
    y_test = encoder.transform(test["expected_department"])
    pipeline = Pipeline(
        [
            (
                "features",
                FeatureUnion(
                    [
                        (
                            "word",
                            TfidfVectorizer(
                                ngram_range=(1, 2),
                                min_df=1,
                                sublinear_tf=True,
                                strip_accents="unicode",
                                max_features=30000,
                            ),
                        ),
                        (
                            "character",
                            TfidfVectorizer(
                                analyzer="char_wb",
                                ngram_range=(3, 5),
                                min_df=1,
                                sublinear_tf=True,
                                max_features=30000,
                            ),
                        ),
                    ]
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=3000,
                    class_weight="balanced",
                    C=2.0,
                    solver="liblinear",
                    random_state=42,
                ),
            ),
        ]
    )
    pipeline.fit(train["grievance_text"], y_train)
    train_predictions = pipeline.predict(train["grievance_text"])
    predictions = pipeline.predict(test["grievance_text"])
    metrics = {
        "model": "Word and character TF-IDF + Logistic Regression",
        "dataset_type": "curated India-oriented examples",
        "split": "stratified 75/25 holdout",
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "train_accuracy": float(accuracy_score(y_train, train_predictions)),
        "test_accuracy": float(accuracy_score(y_test, predictions)),
        "train_macro_f1": float(f1_score(y_train, train_predictions, average="macro")),
        "test_macro_f1": float(f1_score(y_test, predictions, average="macro")),
        "classes": encoder.classes_.tolist(),
        "classification_report": classification_report(
            y_test,
            predictions,
            labels=list(range(len(encoder.classes_))),
            target_names=encoder.classes_,
            output_dict=True,
            zero_division=0,
        ),
    }
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_DIR / "pipeline.joblib")
    joblib.dump(encoder, MODEL_DIR / "label_encoder.joblib")
    test.assign(
        predicted_department=encoder.inverse_transform(predictions),
        confidence=pipeline.predict_proba(test["grievance_text"]).max(axis=1),
    ).to_csv(HOLDOUT, index=False)
    METRICS.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
