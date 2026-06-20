"""Evaluate a trained CRNN handwriting recognition checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm

try:
    from .config import DATASET_DIR, OUTPUTS_DIR, SAVED_MODELS_DIR, ModelConfig, TrainConfig
    from .dataset_loader import build_dataloaders
    from .metrics import character_error_rate, exact_match_accuracy, word_error_rate
    from .model import build_model
except ImportError:
    from config import DATASET_DIR, OUTPUTS_DIR, SAVED_MODELS_DIR, ModelConfig, TrainConfig
    from dataset_loader import build_dataloaders
    from metrics import character_error_rate, exact_match_accuracy, word_error_rate
    from model import build_model


@torch.no_grad()
def evaluate_checkpoint(args: argparse.Namespace) -> Dict[str, float]:
    """Load checkpoint and evaluate on the held-out test split."""

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    model_config = ModelConfig(**checkpoint["model_config"])
    train_config = TrainConfig(**checkpoint.get("train_config", {}))
    train_config.batch_size = args.batch_size
    train_config.num_workers = args.num_workers

    _, _, test_loader, encoder = build_dataloaders(
        dataset_dir=args.dataset_dir,
        model_config=model_config,
        train_config=train_config,
        level=args.level,
    )

    model = build_model(model_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    references: List[str] = []
    hypotheses: List[str] = []
    rows: List[Dict[str, object]] = []
    total_loss = 0.0

    for batch in tqdm(test_loader, desc="Testing"):
        images = batch["images"].to(device)
        labels = batch["labels"].to(device)
        label_lengths = batch["label_lengths"].to(device)
        logits = model(images)
        log_probs = logits.log_softmax(dim=-1)
        input_lengths = model.output_lengths(images.size(0), device=device)
        loss = criterion(log_probs, labels, input_lengths, label_lengths)
        total_loss += float(loss.item())

        predictions = encoder.decode_batch(logits)
        references.extend(batch["texts"])
        hypotheses.extend(predictions)
        for sample_id, path, target, prediction in zip(
            batch["sample_ids"], batch["image_paths"], batch["texts"], predictions
        ):
            rows.append(
                {
                    "sample_id": sample_id,
                    "image_path": path,
                    "target": target,
                    "prediction": prediction,
                    "cer": character_error_rate([target], [prediction]),
                    "wer": word_error_rate([target], [prediction]),
                    "exact_match": target == prediction,
                }
            )

    metrics = {
        "test_loss": total_loss / max(1, len(test_loader)),
        "test_cer": character_error_rate(references, hypotheses),
        "test_wer": word_error_rate(references, hypotheses),
        "test_exact_match": exact_match_accuracy(references, hypotheses),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "evaluation_metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )
    pd.DataFrame(rows).to_csv(args.output_dir / "test_predictions.csv", index=False)
    print(json.dumps(metrics, indent=2))
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate CRNN handwriting checkpoint.")
    parser.add_argument("--dataset-dir", type=Path, default=DATASET_DIR / "huggingface_iam")
    parser.add_argument("--level", choices=["words", "lines"], default="words")
    parser.add_argument("--checkpoint-path", type=Path, default=SAVED_MODELS_DIR / "best_crnn_ctc.pt")
    parser.add_argument("--output-dir", type=Path, default=OUTPUTS_DIR)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    evaluate_checkpoint(parse_args())

