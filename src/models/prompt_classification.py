from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from src.data.prepare_dataset import DEFAULT_OUTPUT as DEFAULT_DATASET
from src.data.prepare_dataset import prepare_dataset
from src.models.evaluate import LABEL_COLUMNS, classification_metrics, write_json


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"
DEFAULT_METRICS_PATH = PROJECT_ROOT / "artifacts" / "metrics" / "prompt_metrics.json"

PromptGenerator = Callable[[str], str]
GENERATION_BACKEND_TEXT2TEXT = "text2text-generation"
GENERATION_BACKEND_CHAT = "chat-generation"


@dataclass(frozen=True)
class PromptClassificationResult:
    model: str
    metrics: dict[str, Any]
    examples: list[dict[str, Any]]


def _truthy(value: Any) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(value >= 0.5)
    normalized = str(value).strip().lower()
    return int(normalized in {"1", "true", "yes", "y", "positive", "present"})


def parse_prompt_labels(text: str, label_columns: tuple[str, ...] = LABEL_COLUMNS) -> dict[str, int]:
    json_match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if json_match:
        try:
            payload = json.loads(json_match.group(0))
        except json.JSONDecodeError:
            payload = None
        if isinstance(payload, dict):
            return {label: _truthy(payload.get(label, 0)) for label in label_columns}

    parsed: dict[str, int] = {}
    for label in label_columns:
        label_pattern = re.escape(label).replace("_", r"[_\s-]?")
        match = re.search(
            rf"{label_pattern}\s*[:=]\s*(true|false|yes|no|1|0)",
            text,
            flags=re.IGNORECASE,
        )
        parsed[label] = _truthy(match.group(1)) if match else 0
    return parsed


def build_prompt(comment: str) -> str:
    return (
        "Classify this tourism review with 0 or 1 for cleanliness, location, luxury, "
        "and family_friendly. Return JSON only with keys cleanliness, location, luxury, "
        f"family_friendly. Review: {comment}"
    )


LABEL_PROMPT_DESCRIPTIONS = {
    "cleanliness": "cleanliness or hygiene, including clean, spotless, tidy, dirty, smell, bathroom, or kitchen",
    "location": (
        "the place location, including perfect location, central, close to attractions, walking distance, "
        "neighborhood, transport, far away, or nearby places"
    ),
    "luxury": "luxury, premium comfort, elegant or beautiful design, unique/special stay, view, privacy, or high-end feel",
    "family_friendly": "family, children, kids, parents, baby, suitable for families, toys, stroller, or child-friendly stay",
}


def build_label_prompt(comment: str, label: str) -> str:
    description = LABEL_PROMPT_DESCRIPTIONS.get(label, label.replace("_", " "))
    return (
        f"Review: {comment}\n"
        f"Question: Does the review mention {description}?\n"
        "Answer with exactly one word: yes or no."
    )


def parse_binary_answer(text: str) -> int:
    normalized = text.strip().lower()
    if re.search(r"\byes\b|\btrue\b|\b1\b", normalized):
        return 1
    return 0


def generation_backend_for_model(model_name: str) -> str:
    normalized = model_name.lower()
    if "t5" in normalized or "flan" in normalized:
        return GENERATION_BACKEND_TEXT2TEXT
    return GENERATION_BACKEND_CHAT


def _load_text2text_generator(model_name: str, max_new_tokens: int) -> PromptGenerator:
    from transformers import pipeline

    pipe = pipeline(GENERATION_BACKEND_TEXT2TEXT, model=model_name)

    def generate(prompt: str) -> str:
        return pipe(prompt, max_new_tokens=max_new_tokens)[0]["generated_text"]

    return generate


def _load_chat_generator(model_name: str, max_new_tokens: int) -> PromptGenerator:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = torch.float16 if device == "mps" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype)
    model.to(device)
    model.eval()

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    system_prompt = (
        "You are a strict tourism-review classifier. "
        "Answer the user's binary question with exactly one word: yes or no."
    )

    def generate(prompt: str) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
            encoded = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
            )
        else:
            rendered = f"{system_prompt}\n\n{prompt}\nAnswer:"
            encoded = tokenizer(rendered, return_tensors="pt").input_ids

        encoded = encoded.to(device)
        attention_mask = torch.ones_like(encoded, device=device)
        with torch.inference_mode():
            output_ids = model.generate(
                encoded,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        generated_ids = output_ids[0, encoded.shape[-1] :]
        return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    return generate


def _load_pipeline_generator(model_name: str, max_new_tokens: int) -> PromptGenerator:
    backend = generation_backend_for_model(model_name)
    if backend == GENERATION_BACKEND_TEXT2TEXT:
        return _load_text2text_generator(model_name, max_new_tokens=max_new_tokens)
    return _load_chat_generator(model_name, max_new_tokens=max_new_tokens)


def load_dataset(path: str | Path = DEFAULT_DATASET) -> pd.DataFrame:
    dataset_path = Path(path)
    if not dataset_path.exists():
        prepare_dataset(output_path=dataset_path)
    return pd.read_csv(dataset_path)


def run_prompt_classification(
    df: pd.DataFrame,
    text_column: str = "clean_comments",
    label_columns: tuple[str, ...] = LABEL_COLUMNS,
    model_name: str = DEFAULT_MODEL_NAME,
    generator: PromptGenerator | None = None,
    sample_size: int = 500,
    random_state: int = 42,
    max_new_tokens: int = 12,
) -> PromptClassificationResult:
    sample = df.sample(n=min(sample_size, len(df)), random_state=random_state).reset_index(drop=True)
    generate = generator or _load_pipeline_generator(model_name, max_new_tokens=max_new_tokens)

    predictions: list[list[int]] = []
    examples: list[dict[str, Any]] = []
    for row in sample.itertuples(index=False):
        comment = str(getattr(row, text_column))
        raw_outputs: dict[str, str] = {}
        parsed: dict[str, int] = {}
        for label in label_columns:
            raw_output = generate(build_label_prompt(comment, label))
            raw_outputs[label] = raw_output
            parsed[label] = parse_binary_answer(raw_output)
        predictions.append([parsed[label] for label in label_columns])
        examples.append({"comment": comment, "raw_outputs": raw_outputs, "parsed": parsed})

    y_true = sample[list(label_columns)].astype(int).to_numpy()
    y_pred = np.asarray(predictions, dtype=int)
    metrics = classification_metrics(y_true, y_pred, label_columns)
    metrics.update({"model": model_name, "sample_size": int(len(sample)), "examples": examples})
    return PromptClassificationResult(model=model_name, metrics=metrics, examples=examples)


def save_prompt_result(
    result: PromptClassificationResult,
    metrics_path: str | Path = DEFAULT_METRICS_PATH,
) -> None:
    write_json(metrics_path, result.metrics)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run local prompt-based generative classification.")
    parser.add_argument("--sample-size", type=int, default=500)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--max-new-tokens", type=int, default=12)
    args = parser.parse_args()

    df = load_dataset()
    result = run_prompt_classification(
        df,
        sample_size=args.sample_size,
        model_name=args.model_name,
        max_new_tokens=args.max_new_tokens,
    )
    save_prompt_result(result)
    print(f"Prompt classifier: micro_f1={result.metrics['micro_f1']:.4f}")


if __name__ == "__main__":
    main()
