import numpy as np
import pandas as pd

from src.models.error_analysis import classification_error_examples, regression_error_examples


def test_classification_error_examples_returns_false_positives_and_negatives():
    df = pd.DataFrame({"clean_comments": ["clean room", "dirty room", "central flat"]})
    y_true = np.asarray([[1, 0], [0, 1], [0, 0]])
    y_pred = np.asarray([[0, 0], [1, 1], [0, 1]])

    result = classification_error_examples(
        df,
        y_true,
        y_pred,
        label_columns=("cleanliness", "location"),
        text_column="clean_comments",
        max_examples=2,
    )

    assert result["cleanliness"]["false_negatives"][0]["comment"] == "clean room"
    assert result["cleanliness"]["false_positives"][0]["comment"] == "dirty room"
    assert result["location"]["false_positives"][0]["comment"] == "central flat"


def test_regression_error_examples_returns_largest_absolute_errors():
    df = pd.DataFrame({"clean_comments": ["great stay", "bad stay", "average stay"]})
    result = regression_error_examples(
        df,
        y_true=np.asarray([4.8, 2.0, 3.0]),
        y_pred=np.asarray([4.5, 4.2, 2.9]),
        text_column="clean_comments",
        max_examples=2,
    )

    assert result["largest_absolute_errors"][0]["comment"] == "bad stay"
    assert result["largest_absolute_errors"][0]["absolute_error"] == 2.2
    assert result["overestimates"][0]["comment"] == "bad stay"
