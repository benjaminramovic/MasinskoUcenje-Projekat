from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import os

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


# def parse_prompt_labels(text: str, label_columns: tuple[str, ...] = LABEL_COLUMNS) -> dict[str, int]:
#     # json_match = re.search(r"\{.*\}", text, flags=re.DOTALL)
#     json_match = re.search(
#         r"\{[\s\S]*",
#         text
#     )
#     if json_match:
#         json_text = json_match.group(0)

    
#         if json_text.count("{") > json_text.count("}"):
#             json_text += "}"

#         try:
#             payload = json.loads(json_text)
#         except json.JSONDecodeError:
#             payload = None
#         if isinstance(payload, dict):
#             return {label: _truthy(payload.get(label, 0)) for label in label_columns}

#     parsed: dict[str, int] = {}
#     for label in label_columns:
#         label_pattern = re.escape(label).replace("_", r"[_\s-]?")
#         match = re.search(
#             rf"{label_pattern}\s*[:=]\s*(true|false|yes|no|1|0)",
#             text,
#             flags=re.IGNORECASE,
#         )
#         parsed[label] = _truthy(match.group(1)) if match else 0
#     return parsed
def parse_prompt_labels(
    text: str,
    label_columns: tuple[str, ...] = LABEL_COLUMNS
) -> dict[str, int]:

    # uzmi deo od prve { do kraja
    json_text = text[text.find("{"):].strip()

    # ako nedostaje zatvarajuća zagrada
    if json_text.count("{") > json_text.count("}"):
        json_text += "}"

    try:
        payload = json.loads(json_text)

        if isinstance(payload, dict):
            return {
                label: _truthy(payload.get(label, 0))
                for label in label_columns
            }

    except json.JSONDecodeError:
        pass


    # fallback ako JSON potpuno ne uspe
    parsed = {}

    for label in label_columns:
        match = re.search(
            rf"{label}\s*[:=]\s*(0|1|true|false)",
            text,
            flags=re.IGNORECASE
        )

        parsed[label] = (
            _truthy(match.group(1))
            if match
            else 0
        )

    return parsed


def build_prompt(comment: str) -> str:
    return f"""
You are a tourism review classifier.

Classify the following review into four binary labels.

Return ONLY valid JSON in the following format:

{{
  "cleanliness": 0,
  "location": 0,
  "luxury": 0,
  "family_friendly": 0
}}

Definitions:
- cleanliness: mentions cleanliness, hygiene, dirty, clean rooms, bathroom, smell, etc.
- location: mentions location, distance, neighborhood, transport, city center, attractions, etc.
- luxury: mentions luxury, premium comfort, elegant design, beautiful view, high-end experience, privacy.
- family_friendly: mentions families, children, kids, babies, toys, playground, family stay.

Review:
{comment}
"""


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
    """
    Ucitava chat model samo jednom i vraca funkciju generate(prompt).

    Na Apple Silicon racunarima koristi MPS (Apple GPU), a ako MPS nije
    dostupan koristi CPU.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # 1. Izbor uredjaja na kome ce model raditi.
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        dtype = torch.float16
        print("Model se ucitava na Apple GPU-u (MPS).")
    else:
        device = torch.device("cpu")
        dtype = torch.float32
        print("MPS nije dostupan. Model se ucitava na CPU-u.")

    # 2. Tokenizer pretvara tekst u brojeve koje model razume.
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
    )

    # 3. Model se ucitava SAMO JEDNOM.
    # Ne koristimo device_map="auto" na MPS-u, vec ga rucno prebacujemo
    # na izabrani uredjaj.
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )

    model.to(device)
    model.eval()

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    def generate(prompt: str) -> str:
        messages = [
            {"role": "user", "content": prompt},
        ]

        # Qwen je chat model, pa tokenizer pravi ispravan chat format.
        if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
            encoded = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
        else:
            encoded = tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
            )

        # I tekstualni ulaz mora biti na istom uredjaju kao model.
        encoded = {key: value.to(device) for key, value in encoded.items()}

        input_ids = encoded["input_ids"]
        attention_mask = encoded.get(
            "attention_mask",
            torch.ones_like(input_ids),
        )

        # inference_mode govori PyTorchu da ne racuna gradijente,
        # jer ovde samo koristimo model, a ne treniramo ga.
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        # Model vraca ulaz + novi odgovor. Uzimamo samo novogenerisani deo.
        generated_ids = output_ids[0, input_ids.shape[-1]:]

        return tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
        ).strip()

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
    max_new_tokens: int = 64,
) -> PromptClassificationResult:
    sample = df.sample(n=min(sample_size, len(df)), random_state=random_state).reset_index(drop=True)
    generate = generator or _load_pipeline_generator(model_name, max_new_tokens=max_new_tokens)

    predictions: list[list[int]] = []
    examples: list[dict[str, Any]] = []
    
    for i, row in enumerate(sample.itertuples(index=False), start=1):
        comment = str(getattr(row, text_column))

        raw_output = generate(build_prompt(comment))
        parsed = parse_prompt_labels(raw_output, label_columns)

        predictions.append([
            parsed[label]
            for label in label_columns
        ])

        examples.append({
            "comment": comment,
            "raw_output": raw_output,
            "parsed": parsed,
        })

        if i == 1 or i % 25 == 0 or i == len(sample):
            print(f"Obrađeno: {i}/{len(sample)} recenzija")

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
    parser.add_argument("--max-new-tokens", type=int, default=64)
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