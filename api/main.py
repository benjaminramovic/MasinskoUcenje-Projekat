from __future__ import annotations

from functools import lru_cache

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

from src.inference.predict import ModelArtifactsMissingError, PredictionService


app = FastAPI(title="Turismy Review ML API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictRequest(BaseModel):
    comment: str = Field(..., min_length=1)

    @field_validator("comment")
    @classmethod
    def comment_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Comment must not be blank.")
        return value.strip()


@lru_cache(maxsize=1)
def get_predictor() -> PredictionService:
    return PredictionService.from_environment()


def reset_predictor_cache() -> None:
    get_predictor.cache_clear()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/model-info")
def model_info() -> dict[str, object]:
    try:
        return get_predictor().model_info()
    except ModelArtifactsMissingError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/predict")
def predict(request: PredictRequest) -> dict[str, object]:
    try:
        return get_predictor().predict_comment(request.comment)
    except ModelArtifactsMissingError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
