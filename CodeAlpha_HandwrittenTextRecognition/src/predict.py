"""Prediction utilities and CLI for handwritten text images."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch

try:
    from .config import SAVED_MODELS_DIR, ModelConfig
    from .dataset_loader import TextEncoder
    from .model import build_model
    from .preprocessing import ImagePreprocessor
except ImportError:
    from config import SAVED_MODELS_DIR, ModelConfig
    from dataset_loader import TextEncoder
    from model import build_model
    from preprocessing import ImagePreprocessor


def load_checkpoint(
    checkpoint_path: Path,
    device: torch.device | None = None,
) -> Tuple[torch.nn.Module, TextEncoder, ModelConfig, Dict[str, object]]:
    """Load a saved CRNN checkpoint for inference."""

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_config = ModelConfig(**checkpoint["model_config"])
    alphabet = checkpoint.get("alphabet", model_config.alphabet)
    model_config.alphabet = alphabet

    model = build_model(model_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    encoder = TextEncoder(alphabet)
    return model, encoder, model_config, checkpoint


@torch.no_grad()
def predict_image(
    image: str | Path | np.ndarray,
    checkpoint_path: Path = SAVED_MODELS_DIR / "best_crnn_ctc.pt",
    device: torch.device | None = None,
    use_advanced_preprocessing: bool = True,
) -> Dict[str, object]:
    """Predict handwritten text from one image."""

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, encoder, model_config, checkpoint = load_checkpoint(checkpoint_path, device)
    processor = ImagePreprocessor(
        image_width=model_config.image_width,
        image_height=model_config.image_height,
        advanced=use_advanced_preprocessing,
    )
    tensor, stages = processor.preprocess_with_intermediates(image)
    tensor = tensor.unsqueeze(0).to(device)

    logits = model(tensor)
    log_probs = logits.log_softmax(dim=-1)
    probabilities = log_probs.exp()
    best_probs, _ = probabilities.max(dim=-1)
    confidence = float(best_probs.mean().item())
    text = encoder.decode_batch(logits)[0]

    return {
        "text": text,
        "confidence": confidence,
        "preprocessed_image": stages["final"],
        "preprocessing_stages": stages,
        "checkpoint_metrics": checkpoint.get("metrics", {}),
        "model_config": model_config.__dict__,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict text from a handwritten image.")
    parser.add_argument("image_path", type=Path)
    parser.add_argument("--checkpoint-path", type=Path, default=SAVED_MODELS_DIR / "best_crnn_ctc.pt")
    parser.add_argument("--simple-preprocessing", action="store_true", help="Use IAM-style preprocessing only.")
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cpu") if args.cpu else None
    result = predict_image(
        args.image_path,
        checkpoint_path=args.checkpoint_path,
        device=device,
        use_advanced_preprocessing=not args.simple_preprocessing,
    )
    print(result["text"])
    print(f"Confidence: {result['confidence']:.3f}")


if __name__ == "__main__":
    main()
