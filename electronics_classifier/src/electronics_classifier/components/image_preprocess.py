"""Image pre-processing steps (kind ``image_preprocess``) and the helpers that chain them."""

from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from ..interfaces import BuildContext, ImageTransform, PreprocessState
from ..registry import register

EXPECTED_IMAGE = "an RGB uint8 array with shape (H, W, 3)"


class _NoParams(BaseModel):
    model_config = ConfigDict(extra="forbid")


def validate_image(image: object) -> np.ndarray:
    """Strict input check used by the pipeline and by ``ensure_rgb_uint8``."""
    if not isinstance(image, np.ndarray):
        raise ValueError(f"image must be {EXPECTED_IMAGE}; got {type(image).__name__}")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"image must be {EXPECTED_IMAGE}; got shape {image.shape}")
    if image.dtype != np.uint8:
        raise ValueError(f"image must be {EXPECTED_IMAGE}, dtype uint8; got dtype {image.dtype}")
    if image.shape[0] == 0 or image.shape[1] == 0:
        raise ValueError(f"image must not be empty; got shape {image.shape}")
    return image


def _identity_map_back(boxes_xyxy: np.ndarray, state: PreprocessState) -> np.ndarray:
    return np.asarray(boxes_xyxy, dtype=np.float64)


@register("image_preprocess", "ensure_rgb_uint8")
class EnsureRgbUint8:
    """Validate the image is ``(H, W, 3) uint8`` and return a contiguous array."""

    Params = _NoParams

    def __init__(self, params: _NoParams, build: BuildContext) -> None:
        self.params = params

    def apply(self, image: np.ndarray, state: PreprocessState) -> np.ndarray:
        return np.ascontiguousarray(validate_image(image))

    map_boxes_back = staticmethod(_identity_map_back)


@register("image_preprocess", "resize_max_side")
class ResizeMaxSide:
    """Geometric: shrink so the longer side is at most ``max_side``; never upscales."""

    class Params(BaseModel):
        model_config = ConfigDict(extra="forbid")
        max_side: int = Field(default=1280, gt=0, description="Longest allowed image side, px.")

    def __init__(self, params: Params, build: BuildContext) -> None:
        self.params = params
        self._key = f"resize_max_side:{id(self)}"

    def apply(self, image: np.ndarray, state: PreprocessState) -> np.ndarray:
        h, w = image.shape[:2]
        longest = max(h, w)
        if longest <= self.params.max_side:
            state.records[self._key] = (1.0, 1.0, 0.0, 0.0)
            return image
        factor = self.params.max_side / longest
        new_w = max(1, round(w * factor))
        new_h = max(1, round(h * factor))
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        sx, sy = new_w / w, new_h / h
        state.records[self._key] = (sx, sy, 0.0, 0.0)
        state.scale_x *= sx
        state.scale_y *= sy
        state.offset_x *= sx
        state.offset_y *= sy
        return resized

    def map_boxes_back(self, boxes_xyxy: np.ndarray, state: PreprocessState) -> np.ndarray:
        boxes = np.asarray(boxes_xyxy, dtype=np.float64)
        sx, sy, ox, oy = state.records.get(self._key, (1.0, 1.0, 0.0, 0.0))
        return (boxes - np.array([ox, oy, ox, oy])) / np.array([sx, sy, sx, sy])


@register("image_preprocess", "clahe")
class Clahe:
    """Non-geometric: contrast-limited adaptive histogram equalization on the L channel."""

    class Params(BaseModel):
        model_config = ConfigDict(extra="forbid")
        clip_limit: float = Field(default=2.0, gt=0.0, description="CLAHE contrast clip limit.")
        tile_grid: int = Field(default=8, gt=0, description="CLAHE tiles per side.")

    def __init__(self, params: Params, build: BuildContext) -> None:
        self.params = params

    def apply(self, image: np.ndarray, state: PreprocessState) -> np.ndarray:
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        engine = cv2.createCLAHE(
            clipLimit=self.params.clip_limit,
            tileGridSize=(self.params.tile_grid, self.params.tile_grid),
        )
        lab[..., 0] = engine.apply(lab[..., 0])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    map_boxes_back = staticmethod(_identity_map_back)


@register("image_preprocess", "gamma")
class Gamma:
    """Non-geometric: ``out = 255 * (in / 255) ** (1 / gamma)``; gamma > 1 brightens."""

    class Params(BaseModel):
        model_config = ConfigDict(extra="forbid")
        gamma: float = Field(default=1.0, gt=0.0, description="Gamma; above 1 brightens.")

    def __init__(self, params: Params, build: BuildContext) -> None:
        self.params = params
        levels = np.arange(256, dtype=np.float64) / 255.0
        self._lut = np.clip(np.rint(255.0 * levels ** (1.0 / params.gamma)), 0, 255).astype(
            np.uint8
        )

    def apply(self, image: np.ndarray, state: PreprocessState) -> np.ndarray:
        return cv2.LUT(image, self._lut)

    map_boxes_back = staticmethod(_identity_map_back)


def apply_preprocess(
    steps: Sequence[ImageTransform], image: np.ndarray
) -> tuple[np.ndarray, PreprocessState]:
    """Run the steps in order on a fresh per-run state."""
    state = PreprocessState(original_h=image.shape[0], original_w=image.shape[1])
    for step in steps:
        image = step.apply(image, state)
    return image, state


def map_boxes_back(
    steps: Sequence[ImageTransform], boxes_xyxy: np.ndarray, state: PreprocessState
) -> np.ndarray:
    """Undo the steps in reverse order; returns float boxes in original-image pixels."""
    boxes = np.asarray(boxes_xyxy, dtype=np.float64).reshape(-1, 4)
    for step in reversed(steps):
        boxes = step.map_boxes_back(boxes, state)
    return boxes
