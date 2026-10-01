"""Language-safe normalization for multilingual grievance text."""

from __future__ import annotations

import re
import unicodedata


def normalize_grievance_text(value: str) -> str:
    text = unicodedata.normalize("NFKC", str(value))
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = re.sub(r"^\s*['\"]|['\"]\s*$", "", text.strip())
    return re.sub(r"\s+", " ", text).strip()
