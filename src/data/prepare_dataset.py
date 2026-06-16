from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = PROJECT_ROOT / "Turismy" / "reviews.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "Turismy" / "reviews_enriched.csv"

HTML_TAG_RE = re.compile(r"<[^>]+>")
SPACE_RE = re.compile(r"\s+")

POSITIVE_TERMS = {
    "amazing",
    "beautiful",
    "clean",
    "comfortable",
    "excellent",
    "fantastic",
    "friendly",
    "great",
    "lovely",
    "perfect",
    "quiet",
    "recommend",
    "spotless",
    "wonderful",
}

NEGATIVE_TERMS = {
    "bad",
    "dirty",
    "disappointed",
    "issue",
    "loud",
    "noisy",
    "poor",
    "problem",
    "smell",
    "uncomfortable",
}

LABEL_COLUMNS = ["cleanliness", "location", "luxury", "family_friendly"]


def clean_text(text: object) -> str:
    """Remove HTML markup and normalize whitespace in a review comment."""
    if text is None or pd.isna(text):
        return ""

    without_html = HTML_TAG_RE.sub(" ", str(text))
    normalized = SPACE_RE.sub(" ", without_html)
    return normalized.strip()


def _stable_noise(comment: str, index: int) -> float:
    payload = f"{index}:{comment}".encode("utf-8", errors="ignore")
    digest = hashlib.sha256(payload).hexdigest()
    raw = int(digest[:8], 16) / 0xFFFFFFFF
    return (raw - 0.5) * 0.4


def _text_signal(comment: str) -> float:
    words = set(re.findall(r"\b[a-zA-Z']+\b", comment.lower()))
    positive_hits = len(words & POSITIVE_TERMS)
    negative_hits = len(words & NEGATIVE_TERMS)
    return min(0.5, positive_hits * 0.08) - min(0.5, negative_hits * 0.12)


def generate_visitor_rating(
    comment: str,
    cleanliness: int,
    location: int,
    luxury: int,
    family_friendly: int,
    index: int,
) -> float:
    """Generate a deterministic synthetic decimal visitor rating from 1.0 to 5.0."""
    label_score = (
        int(cleanliness) * 0.45
        + int(location) * 0.35
        + int(luxury) * 0.30
        + int(family_friendly) * 0.20
    )
    rating = 3.2 + label_score + _text_signal(comment) + _stable_noise(comment, index)
    bounded = max(1.0, min(5.0, rating))
    return round(bounded, 1)


def prepare_dataset(input_path: str | Path = DEFAULT_INPUT, output_path: str | Path = DEFAULT_OUTPUT) -> pd.DataFrame:
    """Load, clean, deduplicate, enrich, and save the reviews dataset."""
    input_path = Path(input_path)
    output_path = Path(output_path)

    df = pd.read_csv(input_path)
    missing_columns = {"comments", *LABEL_COLUMNS} - set(df.columns)
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing_columns)}")

    enriched = df.copy()
    enriched["clean_comments"] = enriched["comments"].apply(clean_text)
    enriched = enriched[enriched["clean_comments"].str.len() > 0]
    enriched = enriched.drop_duplicates(subset=["clean_comments"]).reset_index(drop=True)

    for label in LABEL_COLUMNS:
        enriched[label] = enriched[label].astype(int)

    enriched["visitor_rating"] = [
        generate_visitor_rating(
            row.clean_comments,
            row.cleanliness,
            row.location,
            row.luxury,
            row.family_friendly,
            index,
        )
        for index, row in enumerate(enriched.itertuples(index=False))
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    enriched.to_csv(output_path, index=False)
    return enriched


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare and enrich the Turismy reviews dataset.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="Input CSV path.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Output CSV path.")
    args = parser.parse_args()

    df = prepare_dataset(args.input, args.output)
    print(f"Wrote {len(df)} rows to {args.output}")


if __name__ == "__main__":
    main()
