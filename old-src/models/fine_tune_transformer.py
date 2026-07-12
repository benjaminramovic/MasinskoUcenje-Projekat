from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from src.data.prepare_dataset import DEFAULT_OUTPUT as DEFAULT_DATASET
from src.data.prepare_dataset import prepare_dataset
from src.models.evaluate import LABEL_COLUMNS, classification_metrics, write_json


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_NAME = "prajjwal1/bert-tiny"
DEFAULT_MODEL_DIR = PROJECT_ROOT / "artifacts" / "models" / "bert-tiny-finetuned"
DEFAULT_METRICS_PATH = PROJECT_ROOT / "artifacts" / "metrics" / "finetuning_metrics.json"


class MultiLabelTextDataset(Dataset):
    def __init__(self, encodings: dict[str, Any], labels: np.ndarray) -> None:
        self.encodings = encodings
        self.labels = labels.astype("float32")

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        item = {key: torch.tensor(value[index]) for key, value in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[index], dtype=torch.float32)
        return item


@dataclass(frozen=True)
class FineTuningResult:
    model_name: str
    metrics: dict[str, Any]
    model_dir: Path


def load_dataset(path: str | Path = DEFAULT_DATASET) -> pd.DataFrame:
    dataset_path = Path(path)
    if not dataset_path.exists():
        prepare_dataset(output_path=dataset_path)
    return pd.read_csv(dataset_path)


def _predict(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    threshold: float,
) -> np.ndarray:
    model.eval()
    predictions: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            labels = batch.pop("labels")
            encoded = {key: value.to(device) for key, value in batch.items()}
            logits = model(**encoded).logits
            probabilities = torch.sigmoid(logits).cpu().numpy()
            predictions.append((probabilities >= threshold).astype(int))
            batch["labels"] = labels
    return np.vstack(predictions)


def train_finetuned_transformer(
    df: pd.DataFrame,
    text_column: str = "clean_comments",
    label_columns: tuple[str, ...] = LABEL_COLUMNS,
    model_name: str = DEFAULT_MODEL_NAME,
    sample_size: int = 500,
    random_state: int = 42,
    test_size: float = 0.25,
    epochs: int = 1,
    batch_size: int = 8,
    max_length: int = 96,
    learning_rate: float = 5e-5,
    threshold: float = 0.5,
    model_dir: str | Path = DEFAULT_MODEL_DIR,
) -> FineTuningResult:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    torch.manual_seed(random_state)
    working = df.copy()
    if len(working) > sample_size:
        working = working.sample(n=sample_size, random_state=random_state)

    X = working[text_column].fillna("").astype(str)
    y = working[list(label_columns)].astype(int).to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    train_encodings = tokenizer(X_train.tolist(), truncation=True, padding=True, max_length=max_length)
    test_encodings = tokenizer(X_test.tolist(), truncation=True, padding=True, max_length=max_length)

    train_dataset = MultiLabelTextDataset(train_encodings, np.asarray(y_train))
    test_dataset = MultiLabelTextDataset(test_encodings, np.asarray(y_test))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(label_columns),
        problem_type="multi_label_classification",
    )
    device = torch.device("cpu")
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    model.train()
    for _ in range(epochs):
        for batch in train_loader:
            encoded = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad()
            loss = model(**encoded).loss
            loss.backward()
            optimizer.step()

    predictions = _predict(model, test_loader, device=device, threshold=threshold)
    metrics = classification_metrics(y_test, predictions, label_columns)
    metrics.update(
        {
            "model": model_name,
            "sample_size": int(len(working)),
            "epochs": int(epochs),
            "batch_size": int(batch_size),
        }
    )

    output_dir = Path(model_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    return FineTuningResult(model_name=model_name, metrics=metrics, model_dir=output_dir)


def save_finetuning_result(
    result: FineTuningResult,
    metrics_path: str | Path = DEFAULT_METRICS_PATH,
) -> None:
    write_json(metrics_path, result.metrics)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune a tiny BERT model for multi-label classification.")
    parser.add_argument("--sample-size", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    args = parser.parse_args()

    df = load_dataset()
    result = train_finetuned_transformer(df, sample_size=args.sample_size, epochs=args.epochs, model_name=args.model_name)
    save_finetuning_result(result)
    print(f"Fine-tuned transformer: micro_f1={result.metrics['micro_f1']:.4f}")


if __name__ == "__main__":
    main()
