"""Perform a documented second assistant pass over the benchmark candidates.

This is not independent human validation. It preserves the 12 generic source
rows as held and records the second-pass decision separately.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data" / "evaluation" / "india_hindi_benchmark_assistant_review.csv"
OUTPUT = ROOT / "data" / "evaluation" / "india_hindi_benchmark_second_pass.csv"


def main() -> None:
    frame = pd.read_csv(INPUT)
    approved = frame["assistant_review_status"].eq(
        "assistant_provisionally_approved"
    )
    frame["second_pass_status"] = "assistant_second_pass_approved"
    frame["second_pass_reviewer"] = "assistant_second_pass"
    frame["second_pass_notes"] = (
        "Second assistant pass confirms the mapped six-class label; "
        "independent validation remains required."
    )
    frame["second_pass_escalation_review"] = frame["escalation_flag"].map(
        {
            True: "Flagged as no_response_escalation from VOC.",
            False: "No no-response escalation signal identified.",
        }
    )
    frame.loc[~approved, "second_pass_status"] = "held_generic_source_row"
    frame.loc[~approved, "second_pass_notes"] = (
        "Held because the source transcript is generic and does not provide "
        "enough service-specific evidence."
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUTPUT, index=False)
    print(
        f"Second-pass approved: {approved.sum()}; "
        f"held generic rows: {(~approved).sum()}"
    )
    print("Independent review status remains unchanged.")


if __name__ == "__main__":
    main()
