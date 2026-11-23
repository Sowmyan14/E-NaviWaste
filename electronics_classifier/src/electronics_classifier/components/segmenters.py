"""Segmenters (kind ``segmenter``): ``none``, ``sam`` (segment-anything) and ``sam2``.

``sam`` and ``sam2`` are interchangeable: both prompt with a box and return one bool mask per
box in the original image frame. They fail fast (``ConfigError``) when the checkpoint file is
missing, before any heavy import.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from ..config import ConfigError
from ..interfaces import BuildContext
from ..registry import register

log = logging.getLogger(__name__)


class _NoParams(BaseModel):
    model_config = ConfigDict(extra="forbid")


@register("segmenter", "none")
class NoSegmenter:
    """No segmentation: every item gets ``None`` (a plain rectangular crop)."""

    Params = _NoParams

    def __init__(self, params: _NoParams, build: BuildContext) -> None:
        self.params = params

    def segment(self, image_rgb: np.ndarray, boxes_xyxy: np.ndarray) -> list[np.ndarray | None]:
        return [None] * len(boxes_xyxy)


def _require_checkpoint(checkpoint: str) -> Path:
    path = Path(checkpoint).expanduser()
    if not path.is_file():
        raise ConfigError(f"params.checkpoint: checkpoint file not found: {path}")
    return path


class _BoxPromptSegmenter:
    """Shared flow: load a predictor once, then prompt it with each box."""

    def __init__(self, params: BaseModel, build: BuildContext) -> None:
        self.params = params
        self.device = build.device
        self.checkpoint = _require_checkpoint(params.checkpoint)  # type: ignore[attr-defined]
        self._predictor = self._load_predictor()

    def _load_predictor(self) -> Any:
        raise NotImplementedError

    def segment(self, image_rgb: np.ndarray, boxes_xyxy: np.ndarray) -> list[np.ndarray | None]:
        masks: list[np.ndarray | None] = []
        if len(boxes_xyxy) == 0:
            return masks
        try:
            self._predictor.set_image(image_rgb)
        except Exception as exc:  # e.g. out of memory: degrade to "no masks", do not abort
            log.warning("segmentation setup failed, no masks for this image: %s", exc)
            return [None] * len(boxes_xyxy)
        for box in np.asarray(boxes_xyxy, dtype=np.float32):
            try:
                result, _scores, _logits = self._predictor.predict(
                    box=box, multimask_output=False
                )
                masks.append(np.asarray(result[0]).astype(bool))
            except Exception as exc:  # one bad box must not lose the others
                log.warning("segmentation failed for box %s: %s", box.tolist(), exc)
                masks.append(None)
        return masks


@register("segmenter", "sam")
class SamSegmenter(_BoxPromptSegmenter):
    """Segment Anything (v1) prompted with each detection box."""

    class Params(BaseModel):
        model_config = ConfigDict(extra="forbid")
        checkpoint: str = Field(description="Path to the SAM checkpoint file (.pth).")
        model_type: Literal["vit_b", "vit_l", "vit_h"] = Field(
            default="vit_b", description="SAM backbone matching the checkpoint."
        )

    def _load_predictor(self) -> Any:
        try:
            from segment_anything import SamPredictor, sam_model_registry
        except ImportError as exc:
            raise ImportError(
                "the sam segmenter needs segment-anything: pip install electronics-classifier[sam]"
            ) from exc
        model = sam_model_registry[self.params.model_type](checkpoint=str(self.checkpoint))
        return SamPredictor(model.to(self.device))


@register("segmenter", "sam2")
class Sam2Segmenter(_BoxPromptSegmenter):
    """Segment Anything 2, a drop-in alternative to ``sam``."""

    class Params(BaseModel):
        model_config = ConfigDict(extra="forbid")
        checkpoint: str = Field(description="Path to the SAM 2 checkpoint file (.pt).")
        model_cfg: str = Field(
            description="SAM 2 model config name, e.g. configs/sam2.1/sam2.1_hiera_b+.yaml."
        )

    def _load_predictor(self) -> Any:
        try:
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
        except ImportError as exc:
            raise ImportError(
                "the sam2 segmenter needs sam2: pip install electronics-classifier[sam2]"
            ) from exc
        model = build_sam2(self.params.model_cfg, str(self.checkpoint), device=self.device)
        return SAM2ImagePredictor(model)
