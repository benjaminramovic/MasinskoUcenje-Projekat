from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from src.models.evaluate import LABEL_COLUMNS


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CLASSIFIER_PATH = PROJECT_ROOT / "artifacts" / "models" / "classifier.joblib"
DEFAULT_REGRESSOR_PATH = PROJECT_ROOT / "artifacts" / "models" / "regressor.joblib"
DEFAULT_CLASSIFIER_INFO_PATH = PROJECT_ROOT / "artifacts" / "models" / "classifier_info.json"
DEFAULT_REGRESSOR_INFO_PATH = PROJECT_ROOT / "artifacts" / "models" / "regressor_info.json"

DEFAULT_CLASSIFIER_DISPLAY = "TF-IDF + Linear SVM"
DEFAULT_REGRESSOR_DISPLAY = "TF-IDF + Random Forest Regressor"


class ModelArtifactsMissingError(RuntimeError):
    """Raised when inference is requested before models are trained."""


@dataclass(frozen=True)
class PredictionService:
    classifier: Any
    regressor: Any
    label_columns: tuple[str, ...]
    classifier_name: str
    regressor_name: str

    @classmethod
    def from_paths(
        cls,
        classifier_path: str | Path,
        regressor_path: str | Path,
        classifier_info_path: str | Path = DEFAULT_CLASSIFIER_INFO_PATH,
        regressor_info_path: str | Path = DEFAULT_REGRESSOR_INFO_PATH,
    ) -> "PredictionService":
        classifier_path = Path(classifier_path)
        regressor_path = Path(regressor_path)

        missing = [str(path) for path in (classifier_path, regressor_path) if not path.exists()]
        if missing:
            raise ModelArtifactsMissingError(f"Model artifacts are not available: {', '.join(missing)}")

        classifier_info = _load_json(classifier_info_path)
        regressor_info = _load_json(regressor_info_path)

        return cls(
            classifier=joblib.load(classifier_path),
            regressor=joblib.load(regressor_path),
            label_columns=tuple(classifier_info.get("label_columns", LABEL_COLUMNS)),
            classifier_name=classifier_info.get("display_name", DEFAULT_CLASSIFIER_DISPLAY),
            regressor_name=regressor_info.get("display_name", DEFAULT_REGRESSOR_DISPLAY),
        )

    @classmethod
    def from_environment(cls) -> "PredictionService":
        return cls.from_paths(
            classifier_path=os.getenv("TURISMY_CLASSIFIER_PATH", str(DEFAULT_CLASSIFIER_PATH)),
            regressor_path=os.getenv("TURISMY_REGRESSOR_PATH", str(DEFAULT_REGRESSOR_PATH)),
            classifier_info_path=os.getenv("TURISMY_CLASSIFIER_INFO_PATH", str(DEFAULT_CLASSIFIER_INFO_PATH)),
            regressor_info_path=os.getenv("TURISMY_REGRESSOR_INFO_PATH", str(DEFAULT_REGRESSOR_INFO_PATH)),
        )

    def predict_comment(self, comment: str) -> dict[str, Any]:
        text = [comment]
        predictions = np.asarray(self.classifier.predict(text))[0]
        probabilities = self._classification_probabilities(text)[0]
        rating = float(np.asarray(self.regressor.predict(text))[0])
        rating = round(max(1.0, min(5.0, rating)), 1)

        labels = {
            label: {
                "prediction": bool(predictions[index]),
                "probability": round(float(probabilities[index]), 4),
            }
            for index, label in enumerate(self.label_columns)
        }

        return {
            "labels": labels,
            "visitor_rating": rating,
            "model_info": {
                "classifier": self.classifier_name,
                "regressor": self.regressor_name,
            },
        }

    def model_info(self) -> dict[str, Any]:
        return {
            "classifier": self.classifier_name,
            "regressor": self.regressor_name,
            "labels": self.label_columns,
        }

    def _classification_probabilities(self, text: list[str]) -> np.ndarray:
        if hasattr(self.classifier, "predict_proba"):
            raw_probabilities = self.classifier.predict_proba(text)
            return _normalize_probability_output(raw_probabilities)

        if hasattr(self.classifier, "decision_function"):
            scores = np.asarray(self.classifier.decision_function(text), dtype=float)
            scores = np.atleast_2d(scores)
            return 1.0 / (1.0 + np.exp(-scores))

        predictions = np.asarray(self.classifier.predict(text), dtype=float)
        return np.atleast_2d(predictions)


def _normalize_probability_output(raw_probabilities: Any) -> np.ndarray:
    if isinstance(raw_probabilities, list):
        columns = []
        for item in raw_probabilities:
            item_array = np.asarray(item, dtype=float)
            if item_array.ndim == 2 and item_array.shape[1] > 1:
                columns.append(item_array[:, 1])
            else:
                columns.append(np.ravel(item_array))
        return np.column_stack(columns)

    probabilities = np.asarray(raw_probabilities, dtype=float)
    probabilities = np.atleast_2d(probabilities)
    return probabilities


def _load_json(path: str | Path) -> dict[str, Any]:
    json_path = Path(path)
    if not json_path.exists():
        return {}
    return json.loads(json_path.read_text(encoding="utf-8"))


def predict_comment(comment: str) -> dict[str, Any]:
    service = PredictionService.from_environment()
    return service.predict_comment(comment)
