import pandas as pd

from src.models.train_classic import LABEL_COLUMNS, train_classification_models
from src.models.train_regression import train_regression_models


def _tiny_dataset() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "clean_comments": [
                "spotless clean apartment near the center",
                "dirty room far from everything",
                "luxury canal suite with beautiful design",
                "family friendly home close to museums",
                "clean quiet place with great location",
                "small noisy room with poor comfort",
                "comfortable luxury apartment for a couple",
                "perfect family stay with clean kitchen",
                "central location and helpful host",
                "basic room without luxury features",
                "clean modern studio close to transit",
                "large family apartment with toys",
            ],
            "cleanliness": [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
            "location": [1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0],
            "luxury": [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            "family_friendly": [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
            "visitor_rating": [4.4, 2.4, 4.5, 4.2, 4.6, 2.2, 4.3, 4.7, 4.0, 3.0, 4.3, 4.1],
        }
    )


def test_train_classification_models_returns_best_model_and_metrics():
    result = train_classification_models(_tiny_dataset(), models_to_run=("logistic_regression",), test_size=0.25)

    assert result.name == "logistic_regression"
    assert hasattr(result.model, "predict")
    assert set(result.label_columns) == set(LABEL_COLUMNS)
    assert "micro_f1" in result.metrics
    assert 0.0 <= result.metrics["micro_f1"] <= 1.0


def test_train_regression_models_returns_best_model_and_metrics():
    result = train_regression_models(_tiny_dataset(), models_to_run=("ridge",), test_size=0.25)

    assert result.name == "ridge"
    assert hasattr(result.model, "predict")
    assert "mae" in result.metrics
    assert result.metrics["mae"] >= 0.0
