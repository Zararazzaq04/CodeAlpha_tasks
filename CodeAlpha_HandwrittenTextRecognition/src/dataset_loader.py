"""Merged handwriting dataset loading for CRNN + CTC training.

This module combines:
    1. Existing local parquet dataset, when available
    2. Bibek130/IAM-line from Hugging Face
    3. alpayariyak/IAM_Sentences from Hugging Face

It merges only training splits into the training loader, keeps validation/test
splits separate, removes duplicate IAM samples when possible, rebuilds the
character vocabulary, and keeps validation/inference deterministic.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

try:
    from datasets import load_dataset
except ImportError:  # pragma: no cover - handled with a clear runtime error.
    load_dataset = None

try:
    from .config import DEFAULT_ALPHABET, ModelConfig, TrainConfig
    from .preprocessing import ImagePreprocessor
except ImportError:
    from config import DEFAULT_ALPHABET, ModelConfig, TrainConfig
    from preprocessing import ImagePreprocessor


DEFAULT_HF_DATASETS = ("Bibek130/IAM-line", "alpayariyak/IAM_Sentences")
TEXT_COLUMNS = ("text", "sentence", "transcription", "label", "target", "gt", "ground_truth")
IMAGE_COLUMNS = ("image", "img", "line_image", "word_image")
ID_COLUMNS = ("id", "sample_id", "line_id", "image_id", "path", "file_name", "filename")


@dataclass(frozen=True)
class HandwritingSample:
    """One handwriting sample from a local or Hugging Face dataset."""

    sample_id: str
    source: str
    split: str
    image: Any
    text: str
    image_path: str
    dedupe_key: str


class TextEncoder:
    """Character-level encoder for CTC training."""

    def __init__(self, alphabet: str = DEFAULT_ALPHABET) -> None:
        self.alphabet = alphabet
        self.blank_index = 0
        self.char_to_idx: Dict[str, int] = {
            char: idx + 1 for idx, char in enumerate(alphabet)
        }
        self.idx_to_char: Dict[int, str] = {
            idx + 1: char for idx, char in enumerate(alphabet)
        }

    def encode(self, text: str) -> List[int]:
        """Convert text into integer labels, skipping unsupported characters."""

        return [self.char_to_idx[char] for char in text if char in self.char_to_idx]

    def decode(self, indices: Sequence[int], collapse_repeats: bool = True) -> str:
        """Decode CTC indices into text."""

        chars: List[str] = []
        previous = None
        for idx in indices:
            idx = int(idx)
            if idx == self.blank_index:
                previous = idx
                continue
            if collapse_repeats and idx == previous:
                previous = idx
                continue
            chars.append(self.idx_to_char.get(idx, ""))
            previous = idx
        return "".join(chars)

    def decode_batch(self, predictions: torch.Tensor) -> List[str]:
        """Greedy CTC decode from logits shaped [T, B, C]."""

        if predictions.dim() != 3:
            raise ValueError("Expected predictions with 3 dimensions")

        best = predictions.argmax(dim=-1).detach().cpu()
        return [self.decode(best[:, batch_idx].tolist()) for batch_idx in range(best.shape[1])]


class HandwritingAugmenter:
    """Mild training-only augmentation for IAM-style handwriting images."""

    def __init__(
        self,
        brightness_range: Tuple[float, float] = (0.70, 1.30),
        contrast_range: Tuple[float, float] = (0.70, 1.30),
        blur_probability: float = 0.35,
        noise_probability: float = 0.50,
        rotation_degrees: float = 3.0,
        scale_range: Tuple[float, float] = (0.90, 1.10),
        translation_fraction: float = 0.035,
        perspective_probability: float = 0.35,
        perspective_fraction: float = 0.025,
    ) -> None:
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.blur_probability = blur_probability
        self.noise_probability = noise_probability
        self.rotation_degrees = rotation_degrees
        self.scale_range = scale_range
        self.translation_fraction = translation_fraction
        self.perspective_probability = perspective_probability
        self.perspective_fraction = perspective_fraction

    def __call__(self, image: np.ndarray) -> np.ndarray:
        image = image.astype(np.uint8)
        image = self.random_brightness_contrast(image)
        image = self.random_affine(image)
        image = self.random_perspective(image)
        image = self.random_gaussian_blur(image)
        image = self.random_gaussian_noise(image)
        return np.clip(image, 0, 255).astype(np.uint8)

    def random_brightness_contrast(self, image: np.ndarray) -> np.ndarray:
        brightness = np.random.uniform(*self.brightness_range)
        contrast = np.random.uniform(*self.contrast_range)
        image_float = image.astype(np.float32)
        mean = float(image_float.mean())
        adjusted = (image_float - mean) * contrast + mean
        return np.clip(adjusted * brightness, 0, 255).astype(np.uint8)

    def random_affine(self, image: np.ndarray) -> np.ndarray:
        height, width = image.shape[:2]
        angle = np.random.uniform(-self.rotation_degrees, self.rotation_degrees)
        scale = np.random.uniform(*self.scale_range)
        tx = np.random.uniform(-self.translation_fraction, self.translation_fraction) * width
        ty = np.random.uniform(-self.translation_fraction, self.translation_fraction) * height
        matrix = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), angle, scale)
        matrix[0, 2] += tx
        matrix[1, 2] += ty
        return cv2.warpAffine(
            image,
            matrix,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=255,
        )

    def random_perspective(self, image: np.ndarray) -> np.ndarray:
        if np.random.rand() > self.perspective_probability:
            return image

        height, width = image.shape[:2]
        max_dx = self.perspective_fraction * width
        max_dy = self.perspective_fraction * height
        source = np.float32(
            [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]]
        )
        destination = source + np.float32(
            [
                [np.random.uniform(-max_dx, max_dx), np.random.uniform(-max_dy, max_dy)],
                [np.random.uniform(-max_dx, max_dx), np.random.uniform(-max_dy, max_dy)],
                [np.random.uniform(-max_dx, max_dx), np.random.uniform(-max_dy, max_dy)],
                [np.random.uniform(-max_dx, max_dx), np.random.uniform(-max_dy, max_dy)],
            ]
        )
        matrix = cv2.getPerspectiveTransform(source, destination)
        return cv2.warpPerspective(
            image,
            matrix,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=255,
        )

    def random_gaussian_blur(self, image: np.ndarray) -> np.ndarray:
        if np.random.rand() > self.blur_probability:
            return image
        kernel_size = int(np.random.choice([3, 5]))
        sigma = float(np.random.uniform(0.1, 1.0))
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigmaX=sigma)

    def random_gaussian_noise(self, image: np.ndarray) -> np.ndarray:
        if np.random.rand() > self.noise_probability:
            return image
        noise = np.random.normal(0.0, np.random.uniform(3.0, 12.0), image.shape)
        return np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)


class CombinedHandwritingDataset(Dataset):
    """PyTorch dataset over merged handwriting samples."""

    def __init__(
        self,
        samples: Sequence[HandwritingSample],
        processor: ImagePreprocessor,
        encoder: TextEncoder,
        augmenter: HandwritingAugmenter | None = None,
    ) -> None:
        self.samples = list(samples)
        self.processor = processor
        self.encoder = encoder
        self.augmenter = augmenter

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, object]:
        sample = self.samples[index]
        image = image_record_to_grayscale(sample.image, sample.image_path)
        if self.augmenter is not None:
            image = self.augmenter(image)

        labels = torch.tensor(self.encoder.encode(sample.text), dtype=torch.long)
        return {
            "image": self.processor(image),
            "labels": labels,
            "text": sample.text,
            "image_path": sample.image_path,
            "sample_id": sample.sample_id,
            "source": sample.source,
        }


def build_dataloaders(
    dataset_dir: Path,
    model_config: ModelConfig,
    train_config: TrainConfig,
    level: str = "words",
    hf_dataset_names: Sequence[str] = DEFAULT_HF_DATASETS,
    use_hf_datasets: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader, TextEncoder]:
    """Build dataloaders from local data plus Hugging Face IAM datasets."""

    _ = level
    train_samples, validation_samples, test_samples = load_combined_samples(
        dataset_dir=Path(dataset_dir),
        hf_dataset_names=hf_dataset_names,
        use_hf_datasets=use_hf_datasets,
    )

    alphabet = build_alphabet(train_samples + validation_samples)
    model_config.alphabet = alphabet
    encoder = TextEncoder(alphabet)
    max_label_length = ctc_max_label_length(model_config)

    train_samples = filter_by_ctc_length(train_samples, encoder, max_label_length)
    validation_samples = filter_by_ctc_length(validation_samples, encoder, max_label_length)
    test_samples = filter_by_ctc_length(test_samples, encoder, max_label_length)

    if not train_samples:
        raise RuntimeError("No training samples remain after CTC length filtering.")
    if not validation_samples:
        raise RuntimeError("No validation samples found. Keep validation splits separate and available.")

    print_dataset_statistics(
        train_samples=train_samples,
        validation_samples=validation_samples,
        test_samples=test_samples,
        encoder=encoder,
        alphabet=alphabet,
        ctc_max_length=max_label_length,
    )

    processor = ImagePreprocessor(
        image_width=model_config.image_width,
        image_height=model_config.image_height,
    )
    train_dataset = CombinedHandwritingDataset(
        train_samples,
        processor=processor,
        encoder=encoder,
        augmenter=HandwritingAugmenter(),
    )
    validation_dataset = CombinedHandwritingDataset(
        validation_samples,
        processor=processor,
        encoder=encoder,
        augmenter=None,
    )
    test_dataset = CombinedHandwritingDataset(
        test_samples or validation_samples,
        processor=processor,
        encoder=encoder,
        augmenter=None,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config.batch_size,
        shuffle=True,
        num_workers=train_config.num_workers,
        collate_fn=collate_ctc_batch,
        pin_memory=torch.cuda.is_available(),
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=train_config.batch_size,
        shuffle=False,
        num_workers=train_config.num_workers,
        collate_fn=collate_ctc_batch,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=train_config.batch_size,
        shuffle=False,
        num_workers=train_config.num_workers,
        collate_fn=collate_ctc_batch,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, validation_loader, test_loader, encoder


def print_dataset_statistics(
    train_samples: Sequence[HandwritingSample],
    validation_samples: Sequence[HandwritingSample],
    test_samples: Sequence[HandwritingSample],
    encoder: TextEncoder,
    alphabet: str,
    ctc_max_length: int,
) -> None:
    """Print merged dataset statistics before training starts."""

    all_samples = list(train_samples) + list(validation_samples) + list(test_samples)
    source_counts = Counter(sample.source for sample in all_samples)
    max_observed_label_length = max(
        (len(encoder.encode(sample.text)) for sample in all_samples),
        default=0,
    )

    print("\nDataset statistics")
    print("------------------")
    print(f"Train samples: {len(train_samples):,}")
    print(f"Validation samples: {len(validation_samples):,}")
    print(f"Test samples: {len(test_samples):,}")
    print(f"Samples from local dataset: {source_counts.get('existing_local', 0):,}")
    print(f"Samples from IAM-Line: {source_counts.get('Bibek130/IAM-line', 0):,}")
    print(f"Samples from IAM-Sentences: {source_counts.get('alpayariyak/IAM_Sentences', 0):,}")
    print(f"Alphabet size: {len(alphabet):,}")
    print(f"Maximum label length: {max_observed_label_length:,}")
    print(f"CTC allowed maximum label length: {ctc_max_length:,}")
    print("------------------\n")


def load_combined_samples(
    dataset_dir: Path,
    hf_dataset_names: Sequence[str],
    use_hf_datasets: bool,
) -> Tuple[List[HandwritingSample], List[HandwritingSample], List[HandwritingSample]]:
    """Load, merge, and deduplicate local/HF train-validation-test splits."""

    train_candidates: List[HandwritingSample] = []
    validation_candidates: List[HandwritingSample] = []
    test_candidates: List[HandwritingSample] = []

    local_dir = resolve_existing_dataset_dir(dataset_dir)
    if local_dir is not None:
        local_train, local_validation, local_test = load_local_parquet_samples(local_dir)
        train_candidates.extend(local_train)
        validation_candidates.extend(local_validation)
        test_candidates.extend(local_test)

    if use_hf_datasets:
        if load_dataset is None:
            raise ImportError("Install the `datasets` package to download Hugging Face datasets.")
        for dataset_name in hf_dataset_names:
            hf_train, hf_validation, hf_test = load_huggingface_samples(dataset_name)
            train_candidates.extend(hf_train)
            validation_candidates.extend(hf_validation)
            test_candidates.extend(hf_test)

    seen: set[str] = set()
    train_samples = dedupe_samples(train_candidates, seen)
    validation_samples = dedupe_samples(validation_candidates, seen)
    test_samples = dedupe_samples(test_candidates, seen)
    return train_samples, validation_samples, test_samples


def load_local_parquet_samples(
    dataset_dir: Path,
) -> Tuple[List[HandwritingSample], List[HandwritingSample], List[HandwritingSample]]:
    """Load existing local parquet split files when they are present."""

    return (
        samples_from_parquet(dataset_dir / "train.parquet", "existing_local", "train"),
        samples_from_parquet(dataset_dir / "validation.parquet", "existing_local", "validation"),
        samples_from_parquet(dataset_dir / "test.parquet", "existing_local", "test"),
    )


def samples_from_parquet(path: Path, source: str, split: str) -> List[HandwritingSample]:
    if not path.exists():
        return []
    dataframe = pd.read_parquet(path)
    return samples_from_records(dataframe.to_dict("records"), list(dataframe.columns), source, split)


def load_huggingface_samples(
    dataset_name: str,
) -> Tuple[List[HandwritingSample], List[HandwritingSample], List[HandwritingSample]]:
    """Download one Hugging Face dataset and extract available splits."""

    dataset_dict = load_dataset(dataset_name)
    available_splits = set(dataset_dict.keys())

    train = []
    validation = []
    test = []
    if "train" in available_splits:
        train = samples_from_hf_split(dataset_dict["train"], dataset_name, "train")
    if "validation" in available_splits:
        validation = samples_from_hf_split(dataset_dict["validation"], dataset_name, "validation")
    if "test" in available_splits:
        test = samples_from_hf_split(dataset_dict["test"], dataset_name, "test")

    return train, validation, test


def samples_from_hf_split(dataset: Any, source: str, split: str) -> List[HandwritingSample]:
    return samples_from_records(dataset, list(dataset.column_names), source, split)


def samples_from_records(
    records: Iterable[Dict[str, Any]],
    columns: Sequence[str],
    source: str,
    split: str,
) -> List[HandwritingSample]:
    """Convert generic records into normalized handwriting samples."""

    text_column = find_first_column(columns, TEXT_COLUMNS)
    image_column = find_first_column(columns, IMAGE_COLUMNS)
    id_column = find_first_column(columns, ID_COLUMNS, required=False)

    if text_column is None or image_column is None:
        raise ValueError(f"{source}/{split} must contain text and image columns. Found: {columns}")

    samples: List[HandwritingSample] = []
    for index, record in enumerate(records):
        text = normalize_text(record.get(text_column))
        if not text:
            continue

        image_record = record.get(image_column)
        image_path = image_record_path(image_record)
        raw_id = record.get(id_column) if id_column else None
        sample_id = str(raw_id) if raw_id else f"{source}_{split}_{index}"
        dedupe_key = make_dedupe_key(image_record, image_path, text)
        samples.append(
            HandwritingSample(
                sample_id=sample_id,
                source=source,
                split=split,
                image=image_record,
                text=text,
                image_path=image_path,
                dedupe_key=dedupe_key,
            )
        )
    return samples


def find_first_column(
    columns: Sequence[str],
    candidates: Sequence[str],
    required: bool = True,
) -> str | None:
    lower_to_original = {column.lower(): column for column in columns}
    for candidate in candidates:
        if candidate.lower() in lower_to_original:
            return lower_to_original[candidate.lower()]
    if required:
        return None
    return None


def normalize_text(text: Any) -> str:
    """Normalize whitespace while keeping characters for vocabulary rebuild."""

    if text is None:
        return ""
    return " ".join(str(text).replace("\n", " ").replace("\r", " ").split())


def build_alphabet(samples: Sequence[HandwritingSample]) -> str:
    """Build a stable alphabet from merged training/validation text."""

    observed = set()
    for sample in samples:
        observed.update(sample.text)

    ordered_known = [char for char in DEFAULT_ALPHABET if char in observed]
    extra_chars = sorted(char for char in observed if char not in set(DEFAULT_ALPHABET))
    alphabet = "".join(ordered_known + extra_chars)
    if not alphabet:
        raise RuntimeError("Cannot build an alphabet from empty text data.")
    return alphabet


def filter_by_ctc_length(
    samples: Sequence[HandwritingSample],
    encoder: TextEncoder,
    max_label_length: int,
) -> List[HandwritingSample]:
    """Remove samples whose labels cannot fit the CRNN CTC time dimension."""

    filtered = []
    for sample in samples:
        length = len(encoder.encode(sample.text))
        if 0 < length <= max_label_length:
            filtered.append(sample)
    return filtered


def ctc_max_label_length(model_config: ModelConfig) -> int:
    """The current CRNN reduces image width by a factor of 4."""

    return max(1, model_config.image_width // 4)


def dedupe_samples(
    samples: Sequence[HandwritingSample],
    seen: set[str],
) -> List[HandwritingSample]:
    """Deduplicate samples across local and Hugging Face IAM sources."""

    unique = []
    for sample in samples:
        if sample.dedupe_key in seen:
            continue
        seen.add(sample.dedupe_key)
        unique.append(sample)
    return unique


def make_dedupe_key(image_record: Any, image_path: str, text: str) -> str:
    """Prefer image fingerprint dedupe; fall back to normalized text."""

    image_hash = image_fingerprint(image_record)
    if image_hash:
        return f"image:{image_hash}"
    if image_path:
        return f"path:{Path(image_path).stem.lower()}"
    return "text:" + hashlib.md5(text.lower().encode("utf-8")).hexdigest()


def image_fingerprint(image_record: Any) -> str:
    """Return an md5 fingerprint for image bytes when practical."""

    data = extract_image_bytes(image_record)
    if data:
        return hashlib.md5(data).hexdigest()

    if isinstance(image_record, Image.Image):
        image = image_record.convert("L")
        payload = image.tobytes() + str(image.size).encode("utf-8")
        return hashlib.md5(payload).hexdigest()

    if isinstance(image_record, np.ndarray):
        payload = image_record.tobytes() + str(image_record.shape).encode("utf-8")
        return hashlib.md5(payload).hexdigest()

    return ""


def image_record_to_grayscale(image_record: Any, image_path: str = "") -> np.ndarray:
    """Convert supported HF/local image records to grayscale uint8."""

    if isinstance(image_record, Image.Image):
        return np.asarray(image_record.convert("L"), dtype=np.uint8)

    if isinstance(image_record, np.ndarray):
        if image_record.ndim == 2:
            return image_record.astype(np.uint8)
        return cv2.cvtColor(image_record.astype(np.uint8), cv2.COLOR_RGB2GRAY)

    data = extract_image_bytes(image_record)
    if data:
        return np.asarray(Image.open(BytesIO(data)).convert("L"), dtype=np.uint8)

    path = image_path or image_record_path(image_record)
    if path:
        image = Image.open(path).convert("L")
        return np.asarray(image, dtype=np.uint8)

    raise ValueError("Could not decode image record.")


def extract_image_bytes(image_record: Any) -> bytes | None:
    if isinstance(image_record, dict):
        data = image_record.get("bytes")
    else:
        data = getattr(image_record, "bytes", None)

    if data is None:
        return None
    if isinstance(data, bytes):
        return data
    if isinstance(data, bytearray):
        return bytes(data)
    if isinstance(data, memoryview):
        return data.tobytes()
    if isinstance(data, np.ndarray):
        return data.tobytes()
    if hasattr(data, "as_py"):
        value = data.as_py()
        return value if isinstance(value, bytes) else None
    if hasattr(data, "tobytes"):
        return data.tobytes()
    return None


def image_record_path(image_record: Any) -> str:
    if isinstance(image_record, dict):
        value = image_record.get("path")
    else:
        value = getattr(image_record, "path", None)
    return str(value) if value else ""


def resolve_existing_dataset_dir(dataset_dir: Path) -> Path | None:
    """Find a local existing parquet dataset, if available."""

    dataset_dir = Path(dataset_dir)
    if (dataset_dir / "train.parquet").exists():
        return dataset_dir

    fallback = dataset_dir.parent / "huggingface_iam"
    if (fallback / "train.parquet").exists():
        return fallback

    return None


def collate_ctc_batch(batch: Sequence[Dict[str, object]]) -> Dict[str, object]:
    """Collate variable-length text labels for CTC loss."""

    images = torch.stack([item["image"] for item in batch])
    labels_list = [item["labels"] for item in batch]
    label_lengths = torch.tensor([len(labels) for labels in labels_list], dtype=torch.long)
    labels = torch.cat(labels_list) if labels_list else torch.empty(0, dtype=torch.long)

    return {
        "images": images,
        "labels": labels,
        "label_lengths": label_lengths,
        "texts": [item["text"] for item in batch],
        "image_paths": [item["image_path"] for item in batch],
        "sample_ids": [item["sample_id"] for item in batch],
        "sources": [item["source"] for item in batch],
    }


def samples_to_dataframe(samples: Iterable[HandwritingSample]) -> pd.DataFrame:
    """Convert samples to a DataFrame for inspection."""

    return pd.DataFrame(
        {
            "sample_id": sample.sample_id,
            "source": sample.source,
            "split": sample.split,
            "image_path": sample.image_path,
            "text": sample.text,
            "num_chars": len(sample.text),
            "num_words": len(sample.text.split()),
        }
        for sample in samples
    )
