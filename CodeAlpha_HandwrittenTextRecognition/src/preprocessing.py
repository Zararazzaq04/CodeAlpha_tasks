"""Image preprocessing utilities for CRNN handwriting recognition."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, Union

import cv2
import numpy as np
import torch
from PIL import Image


PathLike = Union[str, Path]
ImageInput = Union[PathLike, np.ndarray, Image.Image]


@dataclass
class ImagePreprocessor:
    """Prepare handwriting images for the CRNN model.

    The advanced path keeps all preprocessing inspection stages in memory for
    Streamlit. It does not save preprocessing images to disk.
    """

    image_width: int = 256
    image_height: int = 64
    keep_aspect_ratio: bool = True
    invert: bool = False
    advanced: bool = False
    min_component_area: int = 120

    def load_image(self, image: ImageInput) -> np.ndarray:
        """Load an image path, PIL image, or array into an OpenCV-style array."""

        if isinstance(image, (str, Path)):
            loaded = cv2.imread(str(image), cv2.IMREAD_UNCHANGED)
            if loaded is None:
                raise FileNotFoundError(f"Could not read image: {image}")
            return self._drop_alpha_on_white(loaded)

        if isinstance(image, Image.Image):
            return cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)

        if not isinstance(image, np.ndarray):
            raise TypeError(f"Unsupported image type: {type(image)!r}")
        return self._drop_alpha_on_white(image)

    def to_grayscale(self, image: np.ndarray) -> np.ndarray:
        """Convert image to grayscale uint8."""

        if image.ndim == 2:
            return image.astype(np.uint8)
        if image.ndim == 3 and image.shape[2] == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        raise ValueError(f"Unsupported image shape: {image.shape}")

    def preprocess_simple(self, image: ImageInput) -> Tuple[torch.Tensor, Dict[str, np.ndarray]]:
        """Training-compatible IAM preprocessing."""

        gray = self.to_grayscale(self.load_image(image))
        final = self.resize_with_padding(gray, interpolation=cv2.INTER_AREA)
        normalized = self.normalize(final)
        return self.to_tensor(normalized), {
            "grayscale": gray,
            "contrast_enhanced": gray,
            "thresholded": gray,
            "final": final,
        }

    def preprocess_advanced(self, image: ImageInput) -> Tuple[torch.Tensor, Dict[str, np.ndarray]]:
        """Preprocess notebook photos while preserving sharp handwriting strokes."""

        original = self.load_image(image)
        gray = self.to_grayscale(original)
        shadow_removed = self.remove_shadows(gray)
        contrast = self.enhance_contrast(shadow_removed)
        thresholded = self.adaptive_threshold(contrast)
        thresholded = self.morphological_speckle_cleanup(thresholded)
        thresholded = self.remove_tiny_components(thresholded)
        cropped = self.crop_to_handwriting(thresholded)
        final = self.resize_with_padding(cropped, interpolation=cv2.INTER_NEAREST)
        normalized = self.normalize(final)

        return self.to_tensor(normalized), {
            "grayscale": gray,
            "contrast_enhanced": contrast,
            "thresholded": thresholded,
            "final": final,
        }

    def remove_shadows(self, gray: np.ndarray) -> np.ndarray:
        """Flatten uneven lighting before contrast enhancement and thresholding."""

        kernel_size = self._odd_kernel(max(101, min(301, min(gray.shape[:2]) // 4)))
        background = cv2.medianBlur(gray, kernel_size)
        flattened = cv2.divide(gray, background, scale=255)
        return np.clip(flattened, 0, 255).astype(np.uint8)

    def enhance_contrast(self, gray: np.ndarray) -> np.ndarray:
        """Increase local contrast before thresholding with CLAHE."""

        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(4, 4))
        return clahe.apply(gray)

    def adaptive_threshold(self, gray: np.ndarray) -> np.ndarray:
        """Create a binary black-text-on-white image using local thresholds."""

        block_size = self._odd_kernel(max(51, min(151, min(gray.shape[:2]) // 8)))
        return cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            block_size,
            19,
        )

    def morphological_speckle_cleanup(self, binary: np.ndarray) -> np.ndarray:
        """Remove isolated black dots without eroding preserved handwriting.

        A tiny morphological opening is used only to detect which connected
        components are isolated speckles. Kept components are copied back from
        the original thresholded image, so handwriting edges are not smoothed,
        eroded, or redrawn.
        """

        ink = (binary < 128).astype(np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        opened = cv2.morphologyEx(ink, cv2.MORPH_OPEN, kernel, iterations=1)

        num_labels, labels, stats, _centroids = cv2.connectedComponentsWithStats(
            ink,
            connectivity=8,
        )
        stable_labels = set(np.unique(labels[opened > 0]).tolist())
        cleaned = np.full(binary.shape, 255, dtype=np.uint8)
        dynamic_min_area = self._component_area_threshold(binary)

        for label in range(1, num_labels):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if label in stable_labels or area >= dynamic_min_area:
                cleaned[labels == label] = 0

        return cleaned

    def remove_tiny_components(self, binary: np.ndarray) -> np.ndarray:
        """Remove small black components that are too small to be sentence text."""

        ink = (binary < 128).astype(np.uint8)
        num_labels, labels, stats, _centroids = cv2.connectedComponentsWithStats(
            ink,
            connectivity=8,
        )
        cleaned = np.full(binary.shape, 255, dtype=np.uint8)
        min_area = self._component_area_threshold(binary)

        for label in range(1, num_labels):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area >= min_area:
                cleaned[labels == label] = 0

        return cleaned

    def crop_to_handwriting(self, binary: np.ndarray) -> np.ndarray:
        """Crop around actual handwriting components, ignoring leftover noise."""

        crop_mask = self.handwriting_crop_mask(binary)
        coords = cv2.findNonZero(crop_mask)
        if coords is None:
            return binary

        x, y, w, h = cv2.boundingRect(coords)
        margin_x = max(12, int(w * 0.08))
        margin_y = max(8, int(h * 0.22))
        x0 = max(0, x - margin_x)
        y0 = max(0, y - margin_y)
        x1 = min(binary.shape[1], x + w + margin_x)
        y1 = min(binary.shape[0], y + h + margin_y)
        return binary[y0:y1, x0:x1]

    def handwriting_crop_mask(self, binary: np.ndarray) -> np.ndarray:
        """Build a crop mask from components large enough to be handwriting."""

        ink = (binary < 128).astype(np.uint8)
        num_labels, labels, stats, _centroids = cv2.connectedComponentsWithStats(
            ink,
            connectivity=8,
        )
        crop_mask = np.zeros(binary.shape, dtype=np.uint8)
        min_area = self._component_area_threshold(binary)

        for label in range(1, num_labels):
            area = int(stats[label, cv2.CC_STAT_AREA])
            x = int(stats[label, cv2.CC_STAT_LEFT])
            y = int(stats[label, cv2.CC_STAT_TOP])
            width = int(stats[label, cv2.CC_STAT_WIDTH])
            height = int(stats[label, cv2.CC_STAT_HEIGHT])
            if self._is_handwriting_component(area, width, height, min_area):
                crop_mask[labels == label] = 255

        return crop_mask

    @staticmethod
    def _is_handwriting_component(area: int, width: int, height: int, min_area: int) -> bool:
        """Decide whether a component should influence the handwriting crop."""

        if area < min_area:
            return False
        if width <= 1 or height <= 1:
            return False

        aspect_ratio = width / max(1, height)
        if height <= 2 and aspect_ratio > 12:
            return False
        return True

    def _component_area_threshold(self, image: np.ndarray) -> int:
        """Use a larger threshold for high-resolution sentence photos."""

        image_area = int(image.shape[0] * image.shape[1])
        return max(int(self.min_component_area), int(image_area * 0.00003))

    def resize_with_padding(self, image: np.ndarray, interpolation: int = cv2.INTER_AREA) -> np.ndarray:
        """Resize while preserving aspect ratio and pad to target dimensions."""

        h, w = image.shape[:2]
        if h <= 0 or w <= 0:
            raise ValueError("Image has invalid dimensions")

        if not self.keep_aspect_ratio:
            return cv2.resize(
                image,
                (self.image_width, self.image_height),
                interpolation=interpolation,
            )

        scale = min(self.image_width / w, self.image_height / h)
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        resized = cv2.resize(image, (new_w, new_h), interpolation=interpolation)

        canvas = np.full((self.image_height, self.image_width), 255, dtype=np.uint8)
        top = (self.image_height - new_h) // 2
        left = (self.image_width - new_w) // 2
        canvas[top : top + new_h, left : left + new_w] = resized
        return canvas

    def normalize(self, image: np.ndarray) -> np.ndarray:
        """Normalize image pixels to float32 in the range [0, 1]."""

        if self.invert:
            image = 255 - image
        return image.astype(np.float32) / 255.0

    def to_tensor(self, image: np.ndarray) -> torch.Tensor:
        """Convert a normalized grayscale image to [1, H, W]."""

        if image.ndim != 2:
            raise ValueError("Expected a grayscale image")
        return torch.from_numpy(image).unsqueeze(0)

    def preprocess_array(self, image: np.ndarray) -> Tuple[torch.Tensor, np.ndarray]:
        """Compatibility helper returning tensor and final preview image."""

        tensor, stages = self.preprocess_with_intermediates(image)
        return tensor, stages["final"]

    def preprocess_with_intermediates(
        self,
        image: ImageInput,
    ) -> Tuple[torch.Tensor, Dict[str, np.ndarray]]:
        """Preprocess and return in-memory visual stages."""

        if self.advanced:
            return self.preprocess_advanced(image)
        return self.preprocess_simple(image)

    def __call__(self, image: ImageInput) -> torch.Tensor:
        """Preprocess an image into a model-ready tensor."""

        tensor, _stages = self.preprocess_with_intermediates(image)
        return tensor

    @staticmethod
    def _odd_kernel(value: int) -> int:
        value = max(3, int(value))
        return value if value % 2 == 1 else value + 1

    @staticmethod
    def _drop_alpha_on_white(image: np.ndarray) -> np.ndarray:
        """Composite transparent images over white and return uint8 data."""

        if image.ndim == 3 and image.shape[2] == 4:
            bgr = image[:, :, :3].astype(np.float32)
            alpha = image[:, :, 3:4].astype(np.float32) / 255.0
            white = np.full_like(bgr, 255, dtype=np.float32)
            return (bgr * alpha + white * (1.0 - alpha)).astype(np.uint8)
        return image.astype(np.uint8)


def preprocess_for_display(
    image: ImageInput,
    image_width: int = 256,
    image_height: int = 64,
    advanced: bool = True,
) -> np.ndarray:
    """Return the final padded image for visualization in Streamlit."""

    processor = ImagePreprocessor(
        image_width=image_width,
        image_height=image_height,
        advanced=advanced,
    )
    _tensor, stages = processor.preprocess_with_intermediates(image)
    return stages["final"]
