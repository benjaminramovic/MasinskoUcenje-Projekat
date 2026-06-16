# Turismy ML Project Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the local CPU-friendly ML workflow, FastAPI inference backend, and React + shadcn/ui frontend for the Turismy reviews project.

**Architecture:** Keep the original CSV immutable, generate an enriched dataset, train models into `artifacts/models`, and expose only inference through FastAPI. The React frontend calls the API and renders label probabilities plus the synthetic visitor rating.

**Tech Stack:** Python, pandas, scikit-learn, matplotlib, seaborn, FastAPI, pytest, Vite, React, TypeScript, Tailwind CSS, shadcn/ui.

---

### Task 1: Project Setup

**Files:**
- Create: `requirements.txt`
- Create: `.gitignore`
- Create directories: `src/data/`, `src/models/`, `src/inference/`, `tests/`, `api/`, `artifacts/models/`, `artifacts/metrics/`, `artifacts/figures/`

- [ ] **Step 1: Add Python requirements**

Create `requirements.txt` with:

```text
pandas>=2.2,<3
numpy>=1.26,<3
scikit-learn>=1.4,<2
joblib>=1.3,<2
matplotlib>=3.8,<4
seaborn>=0.13,<1
jupyter>=1.0,<2
fastapi>=0.111,<1
uvicorn[standard]>=0.30,<1
pydantic>=2.7,<3
pytest>=8,<9
httpx>=0.27,<1
sentence-transformers>=3,<4
transformers>=4.41,<5
torch>=2.2,<3
```

- [ ] **Step 2: Add generated-file ignores**

Create `.gitignore` with:

```text
__pycache__/
.pytest_cache/
.venv/
node_modules/
dist/
artifacts/models/*.joblib
artifacts/models/*.json
artifacts/metrics/*.json
artifacts/figures/*.png
Turismy/reviews_enriched.csv
```

- [ ] **Step 3: Install and verify Python dependencies**

Run:

```powershell
python -m pip install -r requirements.txt
python -c "import pandas, sklearn, fastapi; print('python deps ok')"
```

Expected: prints `python deps ok`.

- [ ] **Step 4: Commit setup**

Run:

```powershell
git add requirements.txt .gitignore src tests api artifacts
git commit -m "chore: add project setup"
```

### Task 2: Data Preparation Pipeline

**Files:**
- Create: `src/data/prepare_dataset.py`
- Create: `tests/test_prepare_dataset.py`

- [ ] **Step 1: Write tests for text cleaning and rating generation**

Create `tests/test_prepare_dataset.py` with tests that assert:

```python
from src.data.prepare_dataset import clean_text, generate_visitor_rating


def test_clean_text_removes_html_and_normalizes_spaces():
    assert clean_text("Nice<br/>   clean\\nplace") == "Nice clean place"


def test_generate_visitor_rating_is_bounded_and_decimal():
    rating = generate_visitor_rating(
        "Very clean place near the center, perfect for family.",
        cleanliness=1,
        location=1,
        luxury=0,
        family_friendly=1,
        index=7,
    )
    assert 1.0 <= rating <= 5.0
    assert round(rating, 1) == rating
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```powershell
pytest tests/test_prepare_dataset.py -v
```

Expected: import failure or missing functions.

- [ ] **Step 3: Implement data preparation**

`src/data/prepare_dataset.py` must provide:

- `clean_text(text: object) -> str`
- `generate_visitor_rating(comment, cleanliness, location, luxury, family_friendly, index) -> float`
- `prepare_dataset(input_path, output_path) -> pandas.DataFrame`
- CLI defaults from `Turismy/reviews.csv` to `Turismy/reviews_enriched.csv`

The rating formula must be deterministic, bounded to `1.0-5.0`, and rounded to one decimal.

- [ ] **Step 4: Run tests and generate enriched dataset**

Run:

```powershell
pytest tests/test_prepare_dataset.py -v
python -m src.data.prepare_dataset
```

Expected: tests pass and `Turismy/reviews_enriched.csv` is created.

- [ ] **Step 5: Commit data pipeline**

Run:

```powershell
git add src/data/prepare_dataset.py tests/test_prepare_dataset.py
git commit -m "feat: add review dataset preparation"
```

### Task 3: Classic Model Training and Metrics

**Files:**
- Create: `src/models/train_classic.py`
- Create: `src/models/train_regression.py`
- Create: `src/models/evaluate.py`
- Create: `tests/test_training_smoke.py`

- [ ] **Step 1: Add smoke tests for training outputs**

Create `tests/test_training_smoke.py` with a tiny in-memory dataset and assertions that:

- classification training returns a fitted estimator and metrics with `micro_f1`
- regression training returns a fitted estimator and metrics with `mae`

- [ ] **Step 2: Run tests and verify they fail**

Run:

```powershell
pytest tests/test_training_smoke.py -v
```

Expected: import failure or missing functions.

- [ ] **Step 3: Implement classic classification**

`src/models/train_classic.py` must:

- load `reviews_enriched.csv`
- split data with `random_state=42`
- train `TF-IDF + OneVsRestClassifier(LogisticRegression(class_weight="balanced"))`
- train `TF-IDF + OneVsRestClassifier(LinearSVC(class_weight="balanced"))`
- evaluate precision/recall/F1 per label plus micro/macro F1
- save the best model as `artifacts/models/classifier.joblib`
- save metrics as `artifacts/metrics/classification_metrics.json`

- [ ] **Step 4: Implement regression**

`src/models/train_regression.py` must:

- load `reviews_enriched.csv`
- split data with `random_state=42`
- train `TF-IDF + Ridge`
- train `TF-IDF + LinearRegression`
- train `TF-IDF + RandomForestRegressor`
- evaluate MAE, RMSE, and R2
- save the best model as `artifacts/models/regressor.joblib`
- save metrics as `artifacts/metrics/regression_metrics.json`

- [ ] **Step 5: Run smoke tests and training scripts**

Run:

```powershell
pytest tests/test_training_smoke.py -v
python -m src.models.train_classic
python -m src.models.train_regression
```

Expected: tests pass, models and metrics are written.

- [ ] **Step 6: Commit model training**

Run:

```powershell
git add src/models tests/test_training_smoke.py
git commit -m "feat: add classic model training"
```

### Task 4: Inference and FastAPI Backend

**Files:**
- Create: `src/inference/predict.py`
- Create: `api/main.py`
- Create: `tests/test_api.py`

- [ ] **Step 1: Add API tests**

Create `tests/test_api.py` with `TestClient` assertions for:

- `GET /health` returns `{ "status": "ok" }`
- `POST /predict` rejects blank comments with 400 or 422
- model loading failure returns a clear 503 response when artifacts are missing

- [ ] **Step 2: Run tests and verify they fail**

Run:

```powershell
pytest tests/test_api.py -v
```

Expected: import failure or missing endpoints.

- [ ] **Step 3: Implement inference helpers**

`src/inference/predict.py` must:

- load `artifacts/models/classifier.joblib`
- load `artifacts/models/regressor.joblib`
- expose `predict_comment(comment: str) -> dict`
- support models with `predict_proba`; fall back to decision scores for SVM-style estimators if needed

- [ ] **Step 4: Implement FastAPI app**

`api/main.py` must expose:

- `GET /health`
- `GET /model-info`
- `POST /predict`

The API must validate non-empty comments and return JSON matching the design spec.

- [ ] **Step 5: Run API tests and manual request**

Run:

```powershell
pytest tests/test_api.py -v
uvicorn api.main:app --reload
```

Then test:

```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/predict -ContentType 'application/json' -Body '{"comment":"The apartment was clean and close to the center."}'
```

Expected: valid label probabilities and `visitor_rating`.

- [ ] **Step 6: Commit inference API**

Run:

```powershell
git add src/inference api tests/test_api.py
git commit -m "feat: add prediction API"
```

### Task 5: React + shadcn/ui Frontend

**Files:**
- Create: `web/`
- Create React components under `web/src/`

- [ ] **Step 1: Scaffold Vite React TypeScript app**

Run:

```powershell
npm create vite@latest web -- --template react-ts
cd web
npm install
```

- [ ] **Step 2: Install Tailwind and shadcn/ui**

Run:

```powershell
cd web
npm install -D tailwindcss postcss autoprefixer
npx tailwindcss init -p
npx shadcn@latest init
npx shadcn@latest add button textarea card badge alert progress separator
```

- [ ] **Step 3: Implement the single-page UI**

`web/src/App.tsx` must render:

- project title
- `Textarea` for review input
- `Button` to call `POST /predict`
- loading state
- error alert
- four `Card` blocks for labels
- `Badge` for positive/negative state
- probability progress display
- visitor rating card
- model info card

- [ ] **Step 4: Add API client**

`web/src/lib/api.ts` must call `http://127.0.0.1:8000/predict` by default and support `VITE_API_URL`.

- [ ] **Step 5: Run frontend checks**

Run:

```powershell
cd web
npm run build
```

Expected: Vite production build succeeds.

- [ ] **Step 6: Commit frontend**

Run:

```powershell
git add web
git commit -m "feat: add React prediction frontend"
```

### Task 6: Notebooks and Final Verification

**Files:**
- Create: `notebooks/01_data_preparation_and_eda.ipynb`
- Create: `notebooks/02_classic_models.ipynb`
- Create: `notebooks/03_encoder_and_finetuning.ipynb`
- Modify: `projektni-zadatak.txt`

- [ ] **Step 1: Add notebook skeletons**

Each notebook must contain runnable cells and markdown explanations for its phase. The classic notebook can call the training scripts instead of duplicating all training code.

- [ ] **Step 2: Update project task notes**

Mark completed tasks in `projektni-zadatak.txt` without deleting the original checklist meaning.

- [ ] **Step 3: Run full verification**

Run:

```powershell
pytest -v
python -m src.data.prepare_dataset
python -m src.models.train_classic
python -m src.models.train_regression
cd web
npm run build
```

Expected: all checks pass.

- [ ] **Step 4: Commit notebooks and final notes**

Run:

```powershell
git add notebooks projektni-zadatak.txt
git commit -m "docs: add notebooks and project notes"
```

