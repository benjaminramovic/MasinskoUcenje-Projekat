import json
from pathlib import Path

from docx import Document

from src.reporting.generate_report import generate_report


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_generate_report_creates_non_empty_docx(tmp_path):
    metrics_dir = tmp_path / "metrics"
    figures_dir = tmp_path / "figures"
    report_path = tmp_path / "Seminarski rad.docx"

    _write_json(
        metrics_dir / "classification_metrics.json",
        {"best_model": "linear_svm", "models": {"linear_svm": {"micro_f1": 0.74, "macro_f1": 0.63}}},
    )
    _write_json(
        metrics_dir / "regression_metrics.json",
        {"best_model": "random_forest", "models": {"random_forest": {"mae": 0.41, "rmse": 0.55, "r2": 0.39}}},
    )
    _write_json(metrics_dir / "encoder_metrics.json", {"model": "all-MiniLM-L6-v2", "micro_f1": 0.7})
    _write_json(metrics_dir / "prompt_metrics.json", {"model": "Qwen/Qwen3-4B-Instruct-2507", "micro_f1": 0.4})
    _write_json(metrics_dir / "finetuning_metrics.json", {"model": "prajjwal1/bert-tiny", "micro_f1": 0.5})
    _write_json(metrics_dir / "error_analysis.json", {"classification": {}, "regression": {}})

    generate_report(metrics_dir=metrics_dir, figures_dir=figures_dir, output_path=report_path)

    assert report_path.exists()
    document = Document(report_path)
    text = "\n".join(paragraph.text for paragraph in document.paragraphs)
    assert "Implementacija encoder i generativnih modela" in text
    assert "Qwen/Qwen3-4B-Instruct-2507" in text
    assert "prajjwal1/bert-tiny" in text
