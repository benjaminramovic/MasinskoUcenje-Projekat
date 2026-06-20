import json
import subprocess
import sys
from pathlib import Path

import pytest
from nbformat import reads, validate


NOTEBOOKS = [
    "01_data_preparation_and_eda.ipynb",
    "02_classic_models.ipynb",
    "03_generative_classification.ipynb",
]


def first_code_cell(notebook_path: Path) -> str:
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            return "".join(cell["source"])
    raise AssertionError(f"{notebook_path} does not contain a code cell")


def notebook_text(notebook_path: Path) -> str:
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    return "\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] in {"markdown", "code"}
    )


def code_cell_sources(notebook_path: Path) -> list[str]:
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    return ["".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"]


@pytest.mark.parametrize("notebook_name", NOTEBOOKS)
def test_notebook_setup_cell_imports_project_modules_from_parent_cwd(notebook_name: str) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    notebook_path = repo_root / "notebooks" / notebook_name
    code = "import matplotlib; matplotlib.use('Agg')\n" + first_code_cell(notebook_path)

    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=repo_root.parent,
        text=True,
        capture_output=True,
        timeout=60,
    )

    assert result.returncode == 0, result.stderr


def test_generative_classification_notebook_focuses_on_prompt_model_vs_classic_classifier() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    notebook_path = repo_root / "notebooks" / "03_generative_classification.ipynb"

    text = notebook_text(notebook_path).lower()

    assert "generativna klasifikacija" in text
    assert "prompt_metrics.json" in text
    assert "classification_metrics.json" in text
    assert "encoder_metrics.json" not in text
    assert "finetuning_metrics.json" not in text
    assert "fine-tuning" not in text
    assert "regression_metrics.json" not in text


def test_generative_classification_notebook_outputs_generative_metrics_before_comparison() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    notebook_path = repo_root / "notebooks" / "03_generative_classification.ipynb"

    text = notebook_text(notebook_path)
    code_cells = code_cell_sources(notebook_path)
    generative_cells = [source for source in code_cells if "generative_metrics_output" in source]
    comparison_cells = [source for source in code_cells if "comparison_metrics_output" in source]

    assert len(generative_cells) == 1
    assert len(comparison_cells) == 1
    assert code_cells.index(generative_cells[0]) < code_cells.index(comparison_cells[0])
    assert "display(generative_metrics_output)" in generative_cells[0]
    assert "display(comparison_metrics_output)" in comparison_cells[0]
    assert generative_cells[0].count("display(") == 1
    assert comparison_cells[0].count("display(") == 1
    assert "metric_scope" in generative_cells[0]
    assert "overall" in generative_cells[0]
    assert "per_label" in generative_cells[0]
    assert "Generativni Qwen prompt" in generative_cells[0]
    assert "Klasicni TF-IDF" not in generative_cells[0]
    assert "metric_scope" in comparison_cells[0]
    assert "overall" in comparison_cells[0]
    assert "per_label" in comparison_cells[0]
    assert "Generativni Qwen prompt" in comparison_cells[0]
    assert "Klasicni TF-IDF" in comparison_cells[0]
    assert "classification_metrics_output" not in text
    assert "display(summary_display)" not in text
    assert "display(per_label.round(4))" not in text
    assert "display(delta_frame.round(4))" not in text
    assert "RUN_GENERATIVE_EXPERIMENT = False" in text


@pytest.mark.parametrize("notebook_name", NOTEBOOKS)
def test_notebook_json_matches_nbformat_schema(notebook_name: str) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    notebook_path = repo_root / "notebooks" / notebook_name
    notebook = reads(notebook_path.read_text(encoding="utf-8"), as_version=4)

    validate(notebook)
