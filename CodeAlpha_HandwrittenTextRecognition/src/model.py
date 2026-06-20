"""CRNN model: CNN feature extractor + BiLSTM sequence model + CTC logits.

Important CTC note:
    `output_lengths()` must not run a dummy tensor through the CNN during
    training. The CNN uses BatchNorm, so repeated dummy forwards in train mode
    corrupt BatchNorm running statistics and make validation/inference fail even
    when training loss looks good.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn

try:
    from .config import ModelConfig
except ImportError:
    from config import ModelConfig


class CNNFeatureExtractor(nn.Module):
    """Convolutional feature extractor for handwriting images."""

    def __init__(self, output_channels: int = 256, dropout: float = 0.25) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(256, output_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
            nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)


class CRNN(nn.Module):
    """Convolutional Recurrent Neural Network for CTC handwriting recognition."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.cnn = CNNFeatureExtractor(
            output_channels=config.cnn_output_channels,
            dropout=config.dropout,
        )

        channels, reduced_height, sequence_width = self._infer_cnn_output_shape()
        self.sequence_width = sequence_width
        rnn_input_size = channels * reduced_height

        self.sequence_model = nn.LSTM(
            input_size=rnn_input_size,
            hidden_size=config.rnn_hidden_size,
            num_layers=config.rnn_layers,
            dropout=config.dropout if config.rnn_layers > 1 else 0.0,
            bidirectional=True,
            batch_first=False,
        )
        self.classifier = nn.Linear(config.rnn_hidden_size * 2, config.num_classes)

    def _infer_cnn_output_shape(self) -> Tuple[int, int, int]:
        """Infer CNN output shape once without changing BatchNorm statistics."""

        was_training = self.cnn.training
        self.cnn.eval()
        with torch.no_grad():
            dummy = torch.zeros(1, 1, self.config.image_height, self.config.image_width)
            output = self.cnn(dummy)
        self.cnn.train(was_training)

        _, channels, height, width = output.shape
        return int(channels), int(height), int(width)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Return unnormalized CTC logits with shape [T, B, C]."""

        features = self.cnn(images)
        batch_size, channels, height, width = features.shape
        sequence = features.permute(3, 0, 1, 2).contiguous()
        sequence = sequence.view(width, batch_size, channels * height)
        recurrent, _ = self.sequence_model(sequence)
        logits = self.classifier(recurrent)
        return logits

    def output_lengths(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Return cached CTC input lengths without forwarding dummy images."""

        return torch.full(
            (batch_size,),
            self.sequence_width,
            dtype=torch.long,
            device=device,
        )

    def infer_sequence_width(self) -> int:
        """Return the cached CTC time dimension for compatibility."""

        return self.sequence_width


def build_model(config: ModelConfig) -> CRNN:
    """Factory function used by training and inference scripts."""

    return CRNN(config)


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """Return total and trainable parameter counts."""

    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return total, trainable
