"""Central configuration for the handwritten text recognition project."""

from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "dataset"
SAVED_MODELS_DIR = PROJECT_ROOT / "saved_models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"


DEFAULT_ALPHABET = (
    " !\"#&'()*+,-./0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "abcdefghijklmnopqrstuvwxyz"
)


@dataclass
class ModelConfig:
    """Configuration shared by training, evaluation, and prediction."""

    image_width: int = 1024
    image_height: int = 64
    cnn_output_channels: int = 256
    rnn_hidden_size: int = 256
    rnn_layers: int = 2
    dropout: float = 0.25
    alphabet: str = DEFAULT_ALPHABET

    @property
    def num_classes(self) -> int:
        """Total classes including CTC blank at index 0."""

        return len(self.alphabet) + 1


@dataclass
class TrainConfig:
    """Training hyperparameters."""

    batch_size: int = 32
    epochs: int = 50
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 8
    num_workers: int = 2
    seed: int = 42
    validation_split: float = 0.1
    test_split: float = 0.1
    max_grad_norm: float = 5.0

