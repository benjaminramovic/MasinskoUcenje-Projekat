import numpy as np
import pandas as pd

from inspect import signature

from src.models import prompt_classification
from src.models.fine_tune_transformer import train_finetuned_transformer
from src.models.prompt_classification import build_label_prompt, parse_prompt_labels, run_prompt_classification
from src.models.train_encoder import train_encoder_classifier
from src.pipeline.run_all import run_all


LABEL_COLUMNS = ("cleanliness", "location", "luxury", "family_friendly")


class FakeEmbedder:
    def encode(self, texts, show_progress_bar=False):
        rows = []
        for text in texts:
            lowered = text.lower()
            rows.append(
                [
                    float("clean" in lowered or "spotless" in lowered),
                    float("center" in lowered or "central" in lowered),
                    float("luxury" in lowered),
                    float("family" in lowered),
                    len(text) / 100.0,
                ]
            )
        return np.asarray(rows, dtype=float)


def _tiny_dataset() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "clean_comments": [
                "spotless clean apartment near the center",
                "dirty room far from everything",
                "luxury canal suite with beautiful design",
                "family friendly home close to museums",
                "clean quiet place with great location",
                "small noisy room with poor comfort",
                "comfortable luxury apartment for a couple",
                "perfect family stay with clean kitchen",
                "central location and helpful host",
                "basic room without luxury features",
                "clean modern studio close to transit",
                "large family apartment with toys",
            ],
            "cleanliness": [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
            "location": [1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0],
            "luxury": [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            "family_friendly": [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
        }
    )


def test_train_encoder_classifier_returns_metrics_with_fake_embedder():
    result = train_encoder_classifier(
        _tiny_dataset(),
        embedder=FakeEmbedder(),
        label_columns=LABEL_COLUMNS,
        sample_size=None,
        test_size=0.25,
    )

    assert result.name == "sentence-transformers/all-MiniLM-L6-v2 + Logistic Regression"
    assert "micro_f1" in result.metrics
    assert set(result.metrics["per_label"]) == set(LABEL_COLUMNS)
    assert hasattr(result.classifier, "predict")


def test_parse_prompt_labels_accepts_json_and_boolean_words():
    parsed = parse_prompt_labels(
        """
        {
          "cleanliness": true,
          "location": 1,
          "luxury": false,
          "family_friendly": "yes"
        }
        """
    )

    assert parsed == {
        "cleanliness": 1,
        "location": 1,
        "luxury": 0,
        "family_friendly": 1,
    }


def test_parse_prompt_labels_uses_regex_fallback():
    parsed = parse_prompt_labels("cleanliness: 1, location: no, luxury: yes, family_friendly: 0")

    assert parsed == {
        "cleanliness": 1,
        "location": 0,
        "luxury": 1,
        "family_friendly": 0,
    }


def test_default_prompt_model_uses_qwen3_instruct():
    assert prompt_classification.DEFAULT_MODEL_NAME == "Qwen/Qwen3-4B-Instruct-2507"


def test_generation_backend_distinguishes_seq2seq_and_chat_models():
    generation_backend_for_model = getattr(prompt_classification, "generation_backend_for_model", None)

    assert callable(generation_backend_for_model)
    assert generation_backend_for_model("google/flan-t5-base") == "text2text-generation"
    assert generation_backend_for_model("Qwen/Qwen3-4B-Instruct-2507") == "chat-generation"


def test_location_prompt_mentions_common_location_signals():
    prompt = build_label_prompt("The houseboat has a perfect location.", "location").lower()

    assert "perfect location" in prompt
    assert "walking distance" in prompt


def test_advanced_experiment_defaults_use_500_samples():
    run_all_signature = signature(run_all)

    assert run_all_signature.parameters["prompt_sample_size"].default == 500
    assert run_all_signature.parameters["finetune_sample_size"].default == 500
    assert signature(run_prompt_classification).parameters["sample_size"].default == 500
    assert signature(train_finetuned_transformer).parameters["sample_size"].default == 500


def test_run_prompt_classification_queries_each_label_with_fake_generator():
    prompts = []

    def fake_generator(prompt: str) -> str:
        prompts.append(prompt)
        if "clean, spotless" in prompt:
            return "yes"
        if "central, close" in prompt:
            return "yes"
        return "no"

    result = run_prompt_classification(
        _tiny_dataset().head(1),
        generator=fake_generator,
        label_columns=LABEL_COLUMNS,
        sample_size=1,
    )

    assert len(prompts) == 4
    assert result.examples[0]["parsed"] == {
        "cleanliness": 1,
        "location": 1,
        "luxury": 0,
        "family_friendly": 0,
    }
