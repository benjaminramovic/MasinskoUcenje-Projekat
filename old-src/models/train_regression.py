from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.data.prepare_dataset import DEFAULT_OUTPUT as DEFAULT_DATASET
from src.data.prepare_dataset import prepare_dataset
from src.models.evaluate import regression_metrics, write_json


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "artifacts" / "models" / "regressor.joblib"
DEFAULT_MODEL_INFO_PATH = PROJECT_ROOT / "artifacts" / "models" / "regressor_info.json"
DEFAULT_METRICS_PATH = PROJECT_ROOT / "artifacts" / "metrics" / "regression_metrics.json"


@dataclass(frozen=True)
class RegressionTrainingResult:
    name: str
    model: Pipeline
    metrics: dict[str, Any]
    all_metrics: dict[str, dict[str, Any]]


def _tfidf(max_features: int = 8_000) -> TfidfVectorizer:
    return TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        stop_words="english",
        ngram_range=(1, 2),
        min_df=1,
        max_features=max_features,
    )


def build_regression_models(random_state: int = 42, max_features: int = 8_000) -> dict[str, Pipeline]:
    return {
        "ridge": Pipeline(steps=[("tfidf", _tfidf(max_features=max_features)), ("regressor", Ridge(alpha=1.0))]),
        "linear_regression": Pipeline(
            steps=[("tfidf", _tfidf(max_features=max_features)), ("regressor", LinearRegression())]
        ),
        "random_forest": Pipeline(
            steps=[
                ("tfidf", _tfidf(max_features=max_features)),
                (
                    "regressor",
                    RandomForestRegressor(
                        n_estimators=60,
                        max_depth=18,
                        min_samples_leaf=3,
                        random_state=random_state,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }


def train_regression_models(
    df: pd.DataFrame,
    text_column: str = "clean_comments",
    target_column: str = "visitor_rating",
    random_state: int = 42,
    test_size: float = 0.2,
    models_to_run: tuple[str, ...] | None = None,
) -> RegressionTrainingResult:
    X = df[text_column].fillna("").astype(str)
    y = df[target_column].astype(float)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
    )

    available_models = build_regression_models(random_state=random_state)
    selected_model_names = models_to_run or tuple(available_models.keys())
    all_metrics: dict[str, dict[str, Any]] = {}
    fitted_models: dict[str, Pipeline] = {}

    for name in selected_model_names:
        model = available_models[name]
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        all_metrics[name] = regression_metrics(y_test, predictions)
        fitted_models[name] = model

    best_name = min(all_metrics, key=lambda model_name: all_metrics[model_name]["mae"])
    return RegressionTrainingResult(
        name=best_name,
        model=fitted_models[best_name],
        metrics=all_metrics[best_name],
        all_metrics=all_metrics,
    )


def load_dataset(path: str | Path = DEFAULT_DATASET) -> pd.DataFrame:
    dataset_path = Path(path)
    if not dataset_path.exists():
        prepare_dataset(output_path=dataset_path)
    return pd.read_csv(dataset_path)


def save_regression_result(
    result: RegressionTrainingResult,
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
        },
    )
    write_json(
        model_info_path,
        {
            "name": result.name,
            "display_name": {
                "ridge": "TF-IDF + Ridge Regression",
                "linear_regression": "TF-IDF + Linear Regression",
                "random_forest": "TF-IDF + Random Forest Regressor",
            }[result.name],
        },
    )


def main() -> None:
    df = load_dataset()
    result = train_regression_models(df)
    save_regression_result(result)
    print(f"Best regressor: {result.name} mae={result.metrics['mae']:.4f}")


if __name__ == "__main__":
    main()
