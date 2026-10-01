"""Prepare the downloaded GCD complaint source for manual review.

This source is TTS-generated and must remain separate from the human-reviewed
benchmark. The output is a deduplicated review queue, not training data.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "raw" / "india_government_complaints_gcd" / "metadata.csv"
OUTPUT = ROOT / "data" / "evaluation" / "india_gcd_review_queue.csv"

CATEGORY_MAP = {
    "water_supply": "Water Supply & Sewerage",
    "electricity": "Electricity & Power",
    "roads_transport": "Roads & Transport",
    "healthcare": "Public Health",
    "sanitation": "Environment & Pollution",
    "ration_aadhar": "Public Distribution System",
}


def clean_transcript(value: str) -> str:
    text = re.sub(r"^\s*\d+\.\s*", "", str(value)).strip()
    return re.sub(r"\s+", " ", text)


def main() -> None:
    if not SOURCE.is_file():
        raise FileNotFoundError(f"Source metadata not found: {SOURCE}")

    source = pd.read_csv(SOURCE).dropna(
        subset=["filename", "category", "language", "transcript"]
    )
    unknown = sorted(set(source["category"]) - set(CATEGORY_MAP))
    if unknown:
        raise ValueError(f"Unmapped source categories: {unknown}")

    derived = pd.DataFrame(
        {
            "grievance_text": source["transcript"].map(clean_transcript),
            "source_category": source["category"].astype(str).str.strip(),
            "language": source["language"].astype(str).str.strip().str.lower(),
            "source_file": source["filename"].astype(str).str.strip(),
        }
    )
    derived["mapped_department"] = derived["source_category"].map(CATEGORY_MAP)
    derived["review_status"] = "synthetic_tts_pending_manual_review"
    derived = derived[
        [
            "grievance_text",
            "source_category",
            "language",
            "source_file",
            "review_status",
            "mapped_department",
        ]
    ]

    before = len(derived)
    derived["_dedupe_key"] = (
        derived["grievance_text"].str.casefold().str.replace(r"\s+", " ", regex=True)
    )
    derived = derived.drop_duplicates("_dedupe_key").drop(columns="_dedupe_key")
    derived = derived[derived["grievance_text"].str.len() > 0]
    if derived["grievance_text"].duplicated().any():
        raise ValueError("Duplicate grievance_text values remain after deduplication")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    derived.to_csv(OUTPUT, index=False)
    print(
        f"Wrote {len(derived)} rows to {OUTPUT} "
        f"(removed {before - len(derived)} exact duplicates)."
    )
    print(derived.groupby(["language", "source_category"]).size().to_string())


if __name__ == "__main__":
    main()
