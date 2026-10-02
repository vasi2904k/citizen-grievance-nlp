"""Create a provisional assistant review without claiming independent validation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data" / "evaluation" / "india_hindi_benchmark_candidates.csv"
OUTPUT = ROOT / "data" / "evaluation" / "india_hindi_benchmark_assistant_review.csv"
CLASSES = {
    "Water Supply & Sewerage",
    "Electricity & Power",
    "Roads & Transport",
    "Public Health",
    "Environment & Pollution",
    "Public Distribution System",
}


def main() -> None:
    frame = pd.read_csv(INPUT)
    frame["assistant_review_status"] = "assistant_provisionally_approved"
    frame["assistant_reviewer"] = "assistant"
    frame["assistant_review_notes"] = (
        "Provisional assistant review; independent validation still required."
    )
    frame["assistant_final_department"] = frame["mapped_department"]
    frame["escalation_flag"] = frame["escalation_flag"].fillna(False).astype(bool)
    frame["escalation_intent"] = frame["escalation_intent"].fillna("")
    frame["escalation_reason"] = frame["escalation_reason"].fillna("")

    source_review_hold = frame["review_status"].eq(
        "assistant_reviewed_needs_second_review"
    )
    invalid_mapping = ~frame["mapped_department"].isin(CLASSES)
    held = source_review_hold | invalid_mapping
    frame.loc[held, "assistant_review_status"] = (
        "assistant_provisionally_needs_second_review"
    )
    frame.loc[held, "assistant_final_department"] = ""
    frame.loc[held, "assistant_review_notes"] = (
        "Held because the source row was generic or had no validated six-class "
        "service-specific evidence."
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUTPUT, index=False)
    print(
        f"Assistant provisional approvals: "
        f"{(~held).sum()}; held for second review: {held.sum()}"
    )
    print("No independent_review_status fields were changed.")


if __name__ == "__main__":
    main()
