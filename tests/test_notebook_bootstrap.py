import json
import subprocess
import sys
from pathlib import Path

import pytest


NOTEBOOKS = [
    "01_data_preparation_and_eda.ipynb",
    "02_classic_models.ipynb",
    "03_encoder_and_finetuning.ipynb",
]


def first_code_cell(notebook_path: Path) -> str:
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            return "".join(cell["source"])
    raise AssertionError(f"{notebook_path} does not contain a code cell")


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
