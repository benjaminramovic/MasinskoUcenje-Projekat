from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.data.prepare_dataset import DEFAULT_OUTPUT as DEFAULT_DATASET
from src.data.prepare_dataset import prepare_dataset
from src.models.evaluate import LABEL_COLUMNS, write_json
from src.models.train_classic import DEFAULT_MODEL_PATH as DEFAULT_CLASSIFIER_PATH
from src.models.train_regression import DEFAULT_MODEL_PATH as DEFAULT_REGRESSOR_PATH


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "artifacts" / "metrics" / "error_analysis.json"


def _round_float(value: float) -> float:
    return round(float(value), 4)


def classification_error_examples(
    df: pd.DataFrame,
    y_true: Any,
    y_pred: Any,
    label_columns: tuple[str, ...] = LABEL_COLUMNS,
    text_column: str = "clean_comments",
    max_examples: int = 5,
) -> dict[str, Any]:
    y_true_array = np.asarray(y_true, dtype=int)
    y_pred_array = np.asarray(y_pred, dtype=int)
    output: dict[str, Any] = {}

    for label_index, label in enumerate(label_columns):
        false_positive_indices = np.where((y_true_array[:, label_index] == 0) & (y_pred_array[:, label_index] == 1))[0]
        false_negative_indices = np.where((y_true_array[:, label_index] == 1) & (y_pred_array[:, label_index] == 0))[0]

        output[label] = {
            "false_positives": [
                {
                    "row": int(index),
                    "comment": str(df.iloc[index][text_column]),
                    "true": 0,
                    "predicted": 1,
                }
                for index in false_positive_indices[:max_examples]
            ],
            "false_negatives": [
                {
                    "row": int(index),
                    "comment": str(df.iloc[index][text_column]),
                    "true": 1,
                    "predicted": 0,
                }
                for index in false_negative_indices[:max_examples]
            ],
        }
    return output


def regression_error_examples(
    df: pd.DataFrame,
    y_true: Any,
    y_pred: Any,
    text_column: str = "clean_comments",
    max_examples: int = 10,
) -> dict[str, Any]:
    y_true_array = np.asarray(y_true, dtype=float)
    y_pred_array = np.asarray(y_pred, dtype=float)
    errors = y_pred_array - y_true_array
    absolute_errors = np.abs(errors)

    def row_payload(index: int) -> dict[str, Any]:
        return {
            "row": int(index),
            "comment": str(df.iloc[index][text_column]),
            "true": _round_float(y_true_array[index]),
            "predicted": _round_float(y_pred_array[index]),
            "error": _round_float(errors[index]),
            "absolute_error": _round_float(absolute_errors[index]),
        }

    largest = np.argsort(-absolute_errors)[:max_examples]
    over = np.where(errors > 0)[0]
    over = over[np.argsort(-errors[over])][:max_examples]
    under = np.where(errors < 0)[0]
    under = under[np.argsort(errors[under])][:max_examples]

    return {
        "largest_absolute_errors": [row_payload(int(index)) for index in largest],
        "overestimates": [row_payload(int(index)) for index in over],
        "underestimates": [row_payload(int(index)) for index in under],
    }


def load_dataset(path: str | Path = DEFAULT_DATASET) -> pd.DataFrame:
    dataset_path = Path(path)
    if not dataset_path.exists():
        prepare_dataset(output_path=dataset_path)
    return pd.read_csv(dataset_path)


def run_error_analysis(
    df: pd.DataFrame,
    classifier_path: str | Path = DEFAULT_CLASSIFIER_PATH,
    regressor_path: str | Path = DEFAULT_REGRESSOR_PATH,
    text_column: str = "clean_comments",
    label_columns: tuple[str, ...] = LABEL_COLUMNS,
    target_column: str = "visitor_rating",
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict[str, Any]:
    classifier = joblib.load(classifier_path)
    regressor = joblib.load(regressor_path)

    X = df[text_column].fillna("").astype(str)
    y_class = df[list(label_columns)].astype(int)
    _, X_test_class, _, y_test_class = train_test_split(
        X,
        y_class,
        test_size=test_size,
        random_state=random_state,
    )
    class_predictions = classifier.predict(X_test_class)
    class_df = df.loc[X_test_class.index].reset_index(drop=True)

    y_reg = df[target_column].astype(float)
    _, X_test_reg, _, y_test_reg = train_test_split(
        X,
        y_reg,
        test_size=test_size,
        random_state=random_state,
    )
    regression_predictions = regressor.predict(X_test_reg)
    reg_df = df.loc[X_test_reg.index].reset_index(drop=True)

    return {
        "classification": classification_error_examples(
            class_df,
            y_test_class.reset_index(drop=True),
            class_predictions,
            label_columns=label_columns,
            text_column=text_column,
        ),
        "regression": regression_error_examples(
            reg_df,
            y_test_reg.reset_index(drop=True),
            regression_predictions,
            text_column=text_column,
        ),
    }


def save_error_analysis(result: dict[str, Any], output_path: str | Path = DEFAULT_OUTPUT_PATH) -> None:
    write_json(output_path, result)


def main() -> None:
    df = load_dataset()
    result = run_error_analysis(df)
    save_error_analysis(result)
    print(f"Wrote error analysis to {DEFAULT_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
