from __future__ import annotations

import argparse
from pathlib import Path

from src.data.prepare_dataset import prepare_dataset
from src.models.error_analysis import run_error_analysis, save_error_analysis
from src.models.fine_tune_transformer import save_finetuning_result, train_finetuned_transformer
from src.models.prompt_classification import run_prompt_classification, save_prompt_result
from src.models.train_classic import save_classification_result, train_classification_models
from src.models.train_encoder import save_encoder_result, train_encoder_classifier
from src.models.train_regression import save_regression_result, train_regression_models
from src.reporting.generate_figures import generate_figures
from src.reporting.generate_report import generate_report


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def run_all(
    encoder_sample_size: int = 800,
    prompt_sample_size: int = 500,
    finetune_sample_size: int = 500,
    skip_prompt: bool = False,
    skip_finetuning: bool = False,
) -> None:
    df = prepare_dataset()

    classification_result = train_classification_models(df)
    save_classification_result(classification_result)

    regression_result = train_regression_models(df)
    save_regression_result(regression_result)

    encoder_result = train_encoder_classifier(df, sample_size=encoder_sample_size)
    save_encoder_result(encoder_result)

    if not skip_prompt:
        prompt_result = run_prompt_classification(df, sample_size=prompt_sample_size)
        save_prompt_result(prompt_result)

    if not skip_finetuning:
        finetuning_result = train_finetuned_transformer(df, sample_size=finetune_sample_size)
        save_finetuning_result(finetuning_result)

    error_analysis = run_error_analysis(df)
    save_error_analysis(error_analysis)

    generate_figures(df)
    generate_report()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the complete Turismy project pipeline.")
    parser.add_argument("--encoder-sample-size", type=int, default=800)
    parser.add_argument("--prompt-sample-size", type=int, default=500)
    parser.add_argument("--finetune-sample-size", type=int, default=500)
    parser.add_argument("--skip-prompt", action="store_true")
    parser.add_argument("--skip-finetuning", action="store_true")
    args = parser.parse_args()

    run_all(
        encoder_sample_size=args.encoder_sample_size,
        prompt_sample_size=args.prompt_sample_size,
        finetune_sample_size=args.finetune_sample_size,
        skip_prompt=args.skip_prompt,
        skip_finetuning=args.skip_finetuning,
    )


if __name__ == "__main__":
    main()
