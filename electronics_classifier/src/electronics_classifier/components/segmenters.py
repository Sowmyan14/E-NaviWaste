"""Segmenters (kind ``segmenter``): ``none``, ``hf_sam``, ``sam`` (segment-anything), ``sam2``.

``sam`` and ``sam2`` are interchangeable: both prompt with a box and return one bool mask per
box in the original image frame. They fail fast (``ConfigError``) when the checkpoint file is
missing, before any heavy import.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Literal

import cv2
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


def fill_mask_holes(mask: np.ndarray) -> np.ndarray:
    """Fill holes that are completely inside the mask (the outline is kept as it is).

    SAM often leaves a label sticker or a camera window out of a device's mask. The crop would
    then blank exactly the text the VLM and OCR need to read.
    """
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    filled = np.zeros(mask.shape, dtype=np.uint8)
    cv2.drawContours(filled, contours, -1, 1, thickness=cv2.FILLED)
    return filled.astype(bool)


@register("segmenter", "hf_sam")
class HfSamSegmenter:
    """Segment Anything from Hugging Face ``transformers``: no checkpoint file to download by hand.

    The Grounded-SAM mask step. All boxes go to SAM in one call; if that call fails each box is
    retried alone, so one bad box loses only its own mask.
    """

    class Params(BaseModel):
        model_config = ConfigDict(extra="forbid")
        model_id: str = Field(
            default="facebook/sam-vit-base",
            description="Hugging Face SAM id: facebook/sam-vit-base, -large or -huge.",
        )
        mps_fallback: bool = Field(
            default=True,
            description="On Apple GPUs let unsupported ops fall back to the CPU.",
        )
        fill_holes: bool = Field(
            default=True,
            description="Fill holes inside each mask so label stickers stay visible in the crop.",
        )

    def __init__(self, params: Params, build: BuildContext) -> None:
        self.params = params
        self.device = build.device
        if build.device == "mps" and params.mps_fallback:
            os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        self._load_backend()

    def _load_backend(self) -> None:
        try:
            import torch
            from transformers import SamModel, SamProcessor
        except ImportError as exc:
            raise ImportError(
                "the hf_sam segmenter needs torch and transformers: "
                "pip install electronics-classifier[grounded]"
            ) from exc
        self._torch = torch
        self._processor = SamProcessor.from_pretrained(self.params.model_id)
        self._model = SamModel.from_pretrained(self.params.model_id).to(self.device).eval()

    def _masks(self, image_rgb: np.ndarray, boxes: list[list[float]]) -> list[np.ndarray]:
        """One model call: a bool ``(H, W)`` mask per box, in ``image_rgb`` pixels."""
        from PIL import Image

        inputs = self._processor(
            images=Image.fromarray(image_rgb), input_boxes=[boxes], return_tensors="pt"
        )
        for key, value in list(inputs.items()):  # Apple GPUs (mps) have no float64
            if getattr(value, "dtype", None) == self._torch.float64:
                inputs[key] = value.to(self._torch.float32)
        inputs = inputs.to(self.device)
        with self._torch.no_grad():
            outputs = self._model(**inputs, multimask_output=False)
        masks = self._processor.image_processor.post_process_masks(
            outputs.pred_masks.cpu(),
            inputs["original_sizes"].cpu(),
            inputs["reshaped_input_sizes"].cpu(),
        )[0]  # (n_boxes, 1, H, W)
        result = [np.asarray(m[0]).astype(bool) for m in masks]
        return [fill_mask_holes(m) for m in result] if self.params.fill_holes else result

    def segment(self, image_rgb: np.ndarray, boxes_xyxy: np.ndarray) -> list[np.ndarray | None]:
        boxes = np.asarray(boxes_xyxy, dtype=np.float32).reshape(-1, 4).tolist()
        if not boxes:
            return []
        try:
            masks = self._masks(image_rgb, boxes)
            if len(masks) == len(boxes):
                return list(masks)
            log.warning("SAM returned %d masks for %d boxes", len(masks), len(boxes))
        except Exception as exc:  # fall through to the per-box retry
            log.warning("segmentation of all boxes failed, retrying one by one: %s", exc)
        result: list[np.ndarray | None] = []
        for box in boxes:
            try:
                result.append(self._masks(image_rgb, [box])[0])
            except Exception as exc:  # one bad box must not lose the others
                log.warning("segmentation failed for box %s: %s", box, exc)
                result.append(None)
        return result
