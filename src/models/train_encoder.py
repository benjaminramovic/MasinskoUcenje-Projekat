from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier

from src.data.prepare_dataset import DEFAULT_OUTPUT as DEFAULT_DATASET
from src.data.prepare_dataset import prepare_dataset
from src.models.evaluate import LABEL_COLUMNS, classification_metrics, write_json


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "artifacts" / "models" / "encoder_classifier.joblib"
DEFAULT_METRICS_PATH = PROJECT_ROOT / "artifacts" / "metrics" / "encoder_metrics.json"


class TextEmbedder(Protocol):
    def encode(self, texts: list[str], show_progress_bar: bool = False) -> Any:
        ...


@dataclass(frozen=True)
class EncoderTrainingResult:
    name: str
    classifier: OneVsRestClassifier
    metrics: dict[str, Any]
    sample_size: int


def _load_embedder(model_name: str) -> TextEmbedder:
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


def load_dataset(path: str | Path = DEFAULT_DATASET) -> pd.DataFrame:
    dataset_path = Path(path)
    if not dataset_path.exists():
        prepare_dataset(output_path=dataset_path)
    return pd.read_csv(dataset_path)


def train_encoder_classifier(
    df: pd.DataFrame,
    text_column: str = "clean_comments",
    label_columns: tuple[str, ...] = LABEL_COLUMNS,
    model_name: str = DEFAULT_MODEL_NAME,
    embedder: TextEmbedder | None = None,
    sample_size: int | None = 2_000,
    random_state: int = 42,
    test_size: float = 0.2,
) -> EncoderTrainingResult:
    working = df.copy()
    if sample_size is not None and len(working) > sample_size:
        working = working.sample(n=sample_size, random_state=random_state)

    X = working[text_column].fillna("").astype(str)
    y = working[list(label_columns)].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
    )

    encoder = embedder or _load_embedder(model_name)
    X_train_embeddings = encoder.encode(X_train.tolist(), show_progress_bar=True)
    X_test_embeddings = encoder.encode(X_test.tolist(), show_progress_bar=True)

    classifier = OneVsRestClassifier(
        LogisticRegression(max_iter=1_000, class_weight="balanced", random_state=random_state)
    )
    classifier.fit(X_train_embeddings, y_train)
    predictions = classifier.predict(X_test_embeddings)

    metrics = classification_metrics(y_test, predictions, label_columns)
    metrics.update({"model": model_name, "sample_size": int(len(working))})
    return EncoderTrainingResult(
        name=f"{model_name} + Logistic Regression",
        classifier=classifier,
        metrics=metrics,
        sample_size=int(len(working)),
    )


def save_encoder_result(
    result: EncoderTrainingResult,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    metrics_path: str | Path = DEFAULT_METRICS_PATH,
) -> None:
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(result.classifier, model_path)
    write_json(metrics_path, result.metrics)


def main() -> None:
    df = load_dataset()
    result = train_encoder_classifier(df)
    save_encoder_result(result)
    print(f"Encoder classifier: micro_f1={result.metrics['micro_f1']:.4f}")


if __name__ == "__main__":
    main()
