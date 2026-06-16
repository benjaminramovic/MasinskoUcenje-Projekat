from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from src.data.prepare_dataset import DEFAULT_OUTPUT as DEFAULT_DATASET
from src.data.prepare_dataset import prepare_dataset
from src.models.evaluate import LABEL_COLUMNS, classification_metrics, write_json


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "artifacts" / "models" / "classifier.joblib"
DEFAULT_MODEL_INFO_PATH = PROJECT_ROOT / "artifacts" / "models" / "classifier_info.json"
DEFAULT_METRICS_PATH = PROJECT_ROOT / "artifacts" / "metrics" / "classification_metrics.json"


@dataclass(frozen=True)
class ClassificationTrainingResult:
    name: str
    model: Pipeline
    metrics: dict[str, Any]
    all_metrics: dict[str, dict[str, Any]]
    label_columns: tuple[str, ...]


def _tfidf(max_features: int = 10_000) -> TfidfVectorizer:
    return TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        stop_words="english",
        ngram_range=(1, 2),
        min_df=1,
        max_features=max_features,
    )


def build_classification_models(random_state: int = 42, max_features: int = 10_000) -> dict[str, Pipeline]:
    return {
        "logistic_regression": Pipeline(
            steps=[
                ("tfidf", _tfidf(max_features=max_features)),
                (
                    "classifier",
                    OneVsRestClassifier(
                        LogisticRegression(
                            class_weight="balanced",
                            max_iter=1_000,
                            solver="liblinear",
                            random_state=random_state,
                        )
                    ),
                ),
            ]
        ),
        "linear_svm": Pipeline(
            steps=[
                ("tfidf", _tfidf(max_features=max_features)),
                (
                    "classifier",
                    OneVsRestClassifier(
                        LinearSVC(class_weight="balanced", max_iter=5_000, random_state=random_state)
                    ),
                ),
            ]
        ),
        "naive_bayes": Pipeline(
            steps=[
                ("tfidf", _tfidf(max_features=max_features)),
                ("classifier", OneVsRestClassifier(MultinomialNB())),
            ]
        ),
    }


def train_classification_models(
    df: pd.DataFrame,
    text_column: str = "clean_comments",
    label_columns: tuple[str, ...] = LABEL_COLUMNS,
    random_state: int = 42,
    test_size: float = 0.2,
    models_to_run: tuple[str, ...] | None = None,
) -> ClassificationTrainingResult:
    X = df[text_column].fillna("").astype(str)
    y = df[list(label_columns)].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
    )

    available_models = build_classification_models(random_state=random_state)
    selected_model_names = models_to_run or tuple(available_models.keys())
    all_metrics: dict[str, dict[str, Any]] = {}
    fitted_models: dict[str, Pipeline] = {}

    for name in selected_model_names:
        model = available_models[name]
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        all_metrics[name] = classification_metrics(y_test, predictions, label_columns)
        fitted_models[name] = model

    best_name = max(all_metrics, key=lambda model_name: all_metrics[model_name]["micro_f1"])
    return ClassificationTrainingResult(
        name=best_name,
        model=fitted_models[best_name],
        metrics=all_metrics[best_name],
        all_metrics=all_metrics,
        label_columns=label_columns,
    )


def load_dataset(path: str | Path = DEFAULT_DATASET) -> pd.DataFrame:
    dataset_path = Path(path)
    if not dataset_path.exists():
        prepare_dataset(output_path=dataset_path)
    return pd.read_csv(dataset_path)


def save_classification_result(
    result: ClassificationTrainingResult,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    metrics_path: str | Path = DEFAULT_METRICS_PATH,
    model_info_path: str | Path = DEFAULT_MODEL_INFO_PATH,
) -> None:
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(result.model, model_path)
    write_json(
        metrics_path,
        {
            "best_model": result.name,
            "models": result.all_metrics,
            "labels": result.label_columns,
        },
    )
    write_json(
        model_info_path,
        {
            "name": result.name,
            "display_name": {
                "logistic_regression": "TF-IDF + Logistic Regression",
                "linear_svm": "TF-IDF + Linear SVM",
                "naive_bayes": "TF-IDF + Naive Bayes",
            }[result.name],
            "label_columns": result.label_columns,
        },
    )


def main() -> None:
    df = load_dataset()
    result = train_classification_models(df)
    save_classification_result(result)
    print(f"Best classifier: {result.name} micro_f1={result.metrics['micro_f1']:.4f}")


if __name__ == "__main__":
    main()
