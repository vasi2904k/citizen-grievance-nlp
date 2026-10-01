"""Apply the assistant's explicit review decisions to the GCD review queue."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data" / "evaluation" / "india_gcd_review_queue.csv"
OUTPUT = ROOT / "data" / "evaluation" / "india_gcd_reviewed.csv"

# These rows are category-labelled by the source directory but too generic to
# approve as standalone routing examples without a second reviewer.
NEEDS_SECOND_REVIEW = {
    "english/audio/electricity/electricity_8.mp3",
    "english/audio/healthcare/healthcare_2.mp3",
    "english/audio/healthcare/healthcare_4.mp3",
    "english/audio/ration_aadhar/ration_aadhar_6.mp3",
    "english/audio/ration_aadhar/ration_aadhar_9.mp3",
    "english/audio/water_supply/water_supply_8.mp3",
    "hindi/audio/electricity/electricity_8.mp3",
    "hindi/audio/healthcare/healthcare_3.mp3",
    "hindi/audio/healthcare/healthcare_6.mp3",
    "hindi/audio/ration_aadhar/ration_aadhar_6.mp3",
    "hindi/audio/ration_aadhar/ration_aadhar_9.mp3",
    "hindi/audio/water_supply/water_supply_8.mp3",
}


def main() -> None:
    frame = pd.read_csv(INPUT)
    frame["reviewer_type"] = "assistant"
    frame["review_status"] = frame["source_file"].map(
        lambda value: (
            "assistant_reviewed_needs_second_review"
            if value in NEEDS_SECOND_REVIEW
            else "assistant_reviewed_approved"
        )
    )
    frame["review_notes"] = frame["source_file"].map(
        lambda value: (
            "Source category is plausible, but transcript lacks a specific "
            "service signal; require second review before training."
            if value in NEEDS_SECOND_REVIEW
            else "Transcript is readable and aligned with the source category "
            "and provisional department mapping."
        )
    )
    frame.to_csv(OUTPUT, index=False)
    print(
        frame["review_status"].value_counts().to_string(),
        f"\nWrote {len(frame)} reviewed rows to {OUTPUT}",
    )


if __name__ == "__main__":
    main()
