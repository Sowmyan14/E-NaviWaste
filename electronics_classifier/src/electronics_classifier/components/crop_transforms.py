"""Crop transforms (kind ``crop_transform``) applied to the crop before it goes to the VLM."""

from __future__ import annotations

import cv2
import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from ..interfaces import BuildContext
from ..registry import register


class _NoParams(BaseModel):
    model_config = ConfigDict(extra="forbid")


@register("crop_transform", "identity")
class Identity:
    """Leave the crop unchanged."""

    Params = _NoParams

    def __init__(self, params: _NoParams, build: BuildContext) -> None:
        self.params = params

    def apply(self, crop_rgb: np.ndarray) -> np.ndarray:
        return crop_rgb


@register("crop_transform", "upscale_min_side")
class UpscaleMinSide:
    """Upscale (bicubic) crops whose shorter side is below ``min_side``; never downscales."""

    class Params(BaseModel):
        model_config = ConfigDict(extra="forbid")
        min_side: int = Field(gt=0, description="Smallest allowed shorter side, px.")

    def __init__(self, params: Params, build: BuildContext) -> None:
        self.params = params

    def apply(self, crop_rgb: np.ndarray) -> np.ndarray:
        h, w = crop_rgb.shape[:2]
        shorter = min(h, w)
        if shorter >= self.params.min_side:
            return crop_rgb
        factor = self.params.min_side / shorter
        size = (max(1, round(w * factor)), max(1, round(h * factor)))
        return cv2.resize(crop_rgb, size, interpolation=cv2.INTER_CUBIC)


@register("crop_transform", "sharpen")
class Sharpen:
    """Unsharp mask: ``crop + amount * (crop - gaussian_blur(crop, sigma=1))``."""

    class Params(BaseModel):
        model_config = ConfigDict(extra="forbid")
        amount: float = Field(ge=0.0, description="Sharpening strength; 0 leaves the crop as is.")

    def __init__(self, params: Params, build: BuildContext) -> None:
        self.params = params

    def apply(self, crop_rgb: np.ndarray) -> np.ndarray:
        blurred = cv2.GaussianBlur(crop_rgb, (0, 0), sigmaX=1.0)
        amount = self.params.amount
        return cv2.addWeighted(crop_rgb, 1.0 + amount, blurred, -amount, 0)


@register("crop_transform", "clahe")
class Clahe:
    """Contrast-limited adaptive histogram equalization on the L channel (8x8 tiles)."""

    class Params(BaseModel):
        model_config = ConfigDict(extra="forbid")
        clip_limit: float = Field(default=2.0, gt=0.0, description="CLAHE contrast clip limit.")

    def __init__(self, params: Params, build: BuildContext) -> None:
        self.params = params

    def apply(self, crop_rgb: np.ndarray) -> np.ndarray:
        lab = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2LAB)
        engine = cv2.createCLAHE(clipLimit=self.params.clip_limit, tileGridSize=(8, 8))
        lab[..., 0] = engine.apply(lab[..., 0])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
