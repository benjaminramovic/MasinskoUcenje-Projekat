from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.data.prepare_dataset import DEFAULT_OUTPUT as DEFAULT_DATASET
from src.data.prepare_dataset import prepare_dataset
from src.models.evaluate import LABEL_COLUMNS


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FIGURES_DIR = PROJECT_ROOT / "artifacts" / "figures"
DEFAULT_METRICS_DIR = PROJECT_ROOT / "artifacts" / "metrics"


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def load_dataset(path: str | Path = DEFAULT_DATASET) -> pd.DataFrame:
    dataset_path = Path(path)
    if not dataset_path.exists():
        prepare_dataset(output_path=dataset_path)
    return pd.read_csv(dataset_path)


def generate_figures(
    df: pd.DataFrame,
    output_dir: str | Path = DEFAULT_FIGURES_DIR,
    metrics_dir: str | Path = DEFAULT_METRICS_DIR,
) -> list[Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    metrics_path = Path(metrics_dir)
    written: list[Path] = []

    label_counts = df[list(LABEL_COLUMNS)].sum().sort_values(ascending=False)
    ax = label_counts.plot(kind="bar", figsize=(8, 4), title="Label distribution")
    ax.set_ylabel("Positive examples")
    plt.tight_layout()
    path = output_path / "label-distribution.png"
    plt.savefig(path, dpi=150)
    plt.close()
    written.append(path)

    word_lengths = df["clean_comments"].str.split().str.len()
    ax = word_lengths.plot(kind="hist", bins=40, figsize=(8, 4), title="Review length distribution")
    ax.set_xlabel("Words per review")
    plt.tight_layout()
    path = output_path / "review-length-distribution.png"
    plt.savefig(path, dpi=150)
    plt.close()
    written.append(path)

    ax = df["visitor_rating"].plot(kind="hist", bins=20, figsize=(8, 4), title="Synthetic visitor rating distribution")
    ax.set_xlabel("visitor_rating")
    plt.tight_layout()
    path = output_path / "visitor-rating-distribution.png"
    plt.savefig(path, dpi=150)
    plt.close()
    written.append(path)

    plt.figure(figsize=(6, 5))
    sns.heatmap(df[list(LABEL_COLUMNS)].corr(), annot=True, cmap="Blues", vmin=0, vmax=1)
    plt.title("Label correlation")
    plt.tight_layout()
    path = output_path / "label-correlation.png"
    plt.savefig(path, dpi=150)
    plt.close()
    written.append(path)

    comparison_rows = []
    classification = _load_json(metrics_path / "classification_metrics.json")
    for name, metrics in classification.get("models", {}).items():
        comparison_rows.append({"model": f"classic:{name}", "micro_f1": metrics.get("micro_f1")})
    for filename, name in (
        ("encoder_metrics.json", "encoder"),
        ("prompt_metrics.json", "prompt"),
        ("finetuning_metrics.json", "fine-tuning"),
    ):
        payload = _load_json(metrics_path / filename)
        if "micro_f1" in payload:
            comparison_rows.append({"model": name, "micro_f1": payload["micro_f1"]})

    if comparison_rows:
        comparison = pd.DataFrame(comparison_rows).dropna()
        ax = comparison.plot(kind="bar", x="model", y="micro_f1", legend=False, figsize=(9, 4), title="Model comparison")
        ax.set_ylabel("Micro F1")
        ax.set_ylim(0, 1)
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        path = output_path / "model-comparison.png"
        plt.savefig(path, dpi=150)
        plt.close()
        written.append(path)

    return written


def main() -> None:
    df = load_dataset()
    paths = generate_figures(df)
    print(f"Wrote {len(paths)} figures to {DEFAULT_FIGURES_DIR}")


if __name__ == "__main__":
    main()
