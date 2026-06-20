"""Train CRNN + CTC from scratch on the merged handwriting dataset."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm

try:
    from .config import DATASET_DIR, OUTPUTS_DIR, SAVED_MODELS_DIR, ModelConfig, TrainConfig
    from .dataset_loader import DEFAULT_HF_DATASETS, TextEncoder, build_dataloaders
    from .model import build_model, count_parameters
except ImportError:
    from config import DATASET_DIR, OUTPUTS_DIR, SAVED_MODELS_DIR, ModelConfig, TrainConfig
    from dataset_loader import DEFAULT_HF_DATASETS, TextEncoder, build_dataloaders
    from model import build_model, count_parameters


SequenceLike = str | List[str]


def set_seed(seed: int) -> None:
    """Make training as reproducible as practical."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def levenshtein_distance(reference: SequenceLike, hypothesis: SequenceLike) -> int:
    """Levenshtein distance for strings or token lists."""

    if reference == hypothesis:
        return 0
    if not reference:
        return len(hypothesis)
    if not hypothesis:
        return len(reference)

    previous = list(range(len(hypothesis) + 1))
    for i, ref_item in enumerate(reference, start=1):
        current = [i]
        for j, hyp_item in enumerate(hypothesis, start=1):
            insert_cost = current[j - 1] + 1
            delete_cost = previous[j] + 1
            replace_cost = previous[j - 1] + (ref_item != hyp_item)
            current.append(min(insert_cost, delete_cost, replace_cost))
        previous = current
    return previous[-1]


def character_error_rate(references: Iterable[str], hypotheses: Iterable[str]) -> float:
    total_distance = 0
    total_chars = 0
    for reference, hypothesis in zip(references, hypotheses):
        total_distance += levenshtein_distance(reference, hypothesis)
        total_chars += max(1, len(reference))
    return total_distance / max(1, total_chars)


def word_error_rate(references: Iterable[str], hypotheses: Iterable[str]) -> float:
    total_distance = 0
    total_words = 0
    for reference, hypothesis in zip(references, hypotheses):
        ref_words = reference.split()
        hyp_words = hypothesis.split()
        total_distance += levenshtein_distance(ref_words, hyp_words)
        total_words += max(1, len(ref_words))
    return total_distance / max(1, total_words)


def exact_match_accuracy(references: Iterable[str], hypotheses: Iterable[str]) -> float:
    pairs = list(zip(references, hypotheses))
    if not pairs:
        return 0.0
    return sum(reference == hypothesis for reference, hypothesis in pairs) / len(pairs)


def train_one_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.CTCLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    max_grad_norm: float,
) -> float:
    """Run one training epoch."""

    model.train()
    running_loss = 0.0
    progress = tqdm(dataloader, desc="Training", leave=False)
    for batch in progress:
        images = batch["images"].to(device)
        labels = batch["labels"].to(device)
        label_lengths = batch["label_lengths"].to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        log_probs = logits.log_softmax(dim=-1)
        input_lengths = model.output_lengths(images.size(0), device=device)
        loss = criterion(log_probs, labels, input_lengths, label_lengths)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        running_loss += float(loss.item())
        progress.set_postfix(loss=f"{loss.item():.4f}")
    return running_loss / max(1, len(dataloader))


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.CTCLoss,
    encoder: TextEncoder,
    device: torch.device,
) -> Dict[str, float]:
    """Evaluate model and compute CER/WER/exact match."""

    model.eval()
    running_loss = 0.0
    references: List[str] = []
    hypotheses: List[str] = []

    for batch in tqdm(dataloader, desc="Validating", leave=False):
        images = batch["images"].to(device)
        labels = batch["labels"].to(device)
        label_lengths = batch["label_lengths"].to(device)
        logits = model(images)
        log_probs = logits.log_softmax(dim=-1)
        input_lengths = model.output_lengths(images.size(0), device=device)
        loss = criterion(log_probs, labels, input_lengths, label_lengths)
        running_loss += float(loss.item())

        predictions = encoder.decode_batch(logits)
        references.extend(batch["texts"])
        hypotheses.extend(predictions)

    return {
        "loss": running_loss / max(1, len(dataloader)),
        "cer": character_error_rate(references, hypotheses),
        "wer": word_error_rate(references, hypotheses),
        "exact_match": exact_match_accuracy(references, hypotheses),
    }


def save_checkpoint(
    path: Path,
    model: nn.Module,
    model_config: ModelConfig,
    train_config: TrainConfig,
    encoder: TextEncoder,
    epoch: int,
    metrics: Dict[str, float],
) -> None:
    """Save a checkpoint compatible with prediction.py."""

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": model_config.__dict__,
            "train_config": train_config.__dict__,
            "alphabet": encoder.alphabet,
            "epoch": epoch,
            "metrics": metrics,
        },
        path,
    )


def run_training(args: argparse.Namespace) -> Path:
    """Train from scratch and save the best checkpoint by validation CER."""

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model_config = ModelConfig(
        image_width=args.image_width,
        image_height=args.image_height,
        rnn_hidden_size=args.hidden_size,
        dropout=args.dropout,
    )
    train_config = TrainConfig(
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        patience=args.patience,
        num_workers=0,
        seed=args.seed,
    )

    print("Training runtime configuration")
    print("------------------------------")
    print(f"batch_size: {train_config.batch_size}")
    print(f"num_workers: {train_config.num_workers}")
    print(f"device: {device}")
    print("------------------------------")

    train_loader, validation_loader, _test_loader, encoder = build_dataloaders(
        dataset_dir=args.dataset_dir,
        model_config=model_config,
        train_config=train_config,
        hf_dataset_names=args.hf_datasets,
        use_hf_datasets=not args.no_hf_datasets,
    )

    model = build_model(model_config).to(device)
    total_params, trainable_params = count_parameters(model)
    print(f"Training from scratch on device: {device}")
    print(f"Vocabulary size including blank: {model_config.num_classes}")
    print(f"Alphabet: {encoder.alphabet}")
    print(f"Model parameters: {trainable_params:,} trainable / {total_params:,} total")

    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = AdamW(
        model.parameters(),
        lr=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=3,
    )

    best_cer = float("inf")
    patience_counter = 0
    history: List[Dict[str, float]] = []
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, train_config.epochs + 1):
        print(f"\nEpoch {epoch}/{train_config.epochs}")
        train_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            getattr(train_config, "max_grad_norm", 5.0),
        )
        val_metrics = validate(model, validation_loader, criterion, encoder, device)
        scheduler.step(val_metrics["cer"])

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_cer": val_metrics["cer"],
            "val_wer": val_metrics["wer"],
            "val_exact_match": val_metrics["exact_match"],
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(row)
        (args.output_dir / "training_history.json").write_text(
            json.dumps(history, indent=2),
            encoding="utf-8",
        )
        print(json.dumps(row, indent=2))

        if val_metrics["cer"] < best_cer:
            best_cer = val_metrics["cer"]
            patience_counter = 0
            save_checkpoint(
                args.checkpoint_path,
                model,
                model_config,
                train_config,
                encoder,
                epoch,
                val_metrics,
            )
            print(f"Saved best checkpoint to {args.checkpoint_path} using validation CER={best_cer:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= train_config.patience:
                print("Early stopping triggered.")
                break

    return args.checkpoint_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CRNN on merged IAM Hugging Face datasets.")
    parser.add_argument("--dataset-dir", type=Path, default=DATASET_DIR / "huggingface_iam")
    parser.add_argument("--checkpoint-path", type=Path, default=SAVED_MODELS_DIR / "best_crnn_ctc.pt")
    parser.add_argument("--output-dir", type=Path, default=OUTPUTS_DIR)
    parser.add_argument("--hf-datasets", nargs="*", default=list(DEFAULT_HF_DATASETS))
    parser.add_argument("--no-hf-datasets", action="store_true", help="Use only the existing local dataset.")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--image-width", type=int, default=512)
    parser.add_argument("--image-height", type=int, default=64)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--num-workers", type=int, default=0, help="Ignored; DataLoader workers are forced to 0.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run_training(parse_args())
