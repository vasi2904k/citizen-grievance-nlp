"""Train a separate six-class multilingual GCD routing model."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import LabelEncoder

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from text_normalization import normalize_grievance_text

REVIEWED = ROOT / "data" / "evaluation" / "india_gcd_reviewed.csv"
AUTHORED = ROOT / "data" / "evaluation" / "india_gcd_hindi_examples.csv"
MODEL_DIR = ROOT / "models" / "india_gcd_auxiliary"
SPLIT = ROOT / "evaluation" / "india_gcd_auxiliary_split.csv"


def make_pipeline() -> Pipeline:
    return Pipeline(
        [
            (
                "features",
                FeatureUnion(
                    [
                        (
                            "word",
                            TfidfVectorizer(
                                ngram_range=(1, 2),
                                strip_accents=None,
                                sublinear_tf=True,
                                max_features=30000,
                            ),
                        ),
                        (
                            "character",
                            TfidfVectorizer(
                                analyzer="char",
                                ngram_range=(2, 5),
                                min_df=1,
                                sublinear_tf=True,
                                max_features=50000,
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


def main() -> None:
    reviewed = pd.read_csv(REVIEWED)
    reviewed = reviewed[
        reviewed["review_status"] == "assistant_reviewed_approved"
    ].copy()
    authored = pd.read_csv(AUTHORED)
    source = reviewed[
        ["grievance_text", "language", "mapped_department", "source_file"]
    ].copy()
    source["grievance_text"] = source["grievance_text"].map(normalize_grievance_text)
    source["is_evaluation_source"] = True
    source["source_group"] = source["source_file"]
    authored = authored.rename(columns={"source_category": "source_file"})
    authored["source_file"] = "authored_hindi/" + authored["source_file"]
    authored["is_evaluation_source"] = False
    authored = authored[
        ["grievance_text", "language", "mapped_department", "source_file",
         "is_evaluation_source"]
    ]
    split = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
    train_indices, test_indices = next(
        split.split(source, groups=source["source_group"])
    )
    gcd_train = source.iloc[train_indices].copy()
    gcd_test = source.iloc[test_indices].copy()
    training = pd.concat([gcd_train, authored], ignore_index=True)
    encoder = LabelEncoder()
    labels = encoder.fit_transform(training["mapped_department"])
    pipeline = make_pipeline()
    pipeline.fit(training["grievance_text"], labels)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_DIR / "pipeline.joblib")
    joblib.dump(encoder, MODEL_DIR / "label_encoder.joblib")
    gcd_test.assign(
        expected_department=gcd_test["mapped_department"],
        predicted_department=encoder.inverse_transform(
            pipeline.predict(gcd_test["grievance_text"])
        ),
    ).to_csv(SPLIT, index=False)
    metadata = {
        "model": "word and character TF-IDF + Logistic Regression",
        "classes": encoder.classes_.tolist(),
        "gcd_train_rows": int(len(gcd_train)),
        "gcd_test_rows": int(len(gcd_test)),
        "authored_hindi_training_rows": int(len(authored)),
        "normalization": "Unicode NFKC, quote cleanup, whitespace normalization",
        "evaluation_source_excluded_from_training": True,
    }
    (MODEL_DIR / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
