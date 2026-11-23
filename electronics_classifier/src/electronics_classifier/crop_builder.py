"""Crop builder (fixed logic, params from config): segmentation orchestration, padding, crops.

All crops, the context image and masks come from the ORIGINAL image; boxes are in original
pixels. Nothing here mutates the input image.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import supervision as sv
from PIL import Image

from .components.segmenters import NoSegmenter
from .config import CropConfig
from .interfaces import CropTransform, SegmentationGate, Segmenter
from .schemas import BBox, PreparedItem, RawDetection, WarningCode

log = logging.getLogger(__name__)

MASK_FILL = 128  # outside-mask pixels in a masked crop are neutral gray


@dataclass(frozen=True)
class SegmentationOutcome:
    """``masks`` has one entry per detection; ``approved`` lists the indices the gate chose."""

    masks: list[np.ndarray | None]
    approved: tuple[int, ...]


def segment_approved(
    segmenter: Segmenter,
    gate: SegmentationGate,
    image_rgb: np.ndarray,
    dets: Sequence[RawDetection],
) -> SegmentationOutcome:
    """Masks where the gate approved and the segmenter produced one (else ``None``).

    The gate is only consulted when a real segmenter is configured.
    """
    masks: list[np.ndarray | None] = [None] * len(dets)
    if isinstance(segmenter, NoSegmenter):
        return SegmentationOutcome(masks, ())
    approved = tuple(i for i in range(len(dets)) if gate.should_segment(i, dets))
    if approved:
        boxes = np.array([dets[i].bbox_xyxy for i in approved], dtype=np.float64)
        for i, mask in zip(approved, segmenter.segment(image_rgb, boxes), strict=False):
            masks[i] = mask
    return SegmentationOutcome(masks, approved)


def segmentation_warnings(
    outcome: SegmentationOutcome, items: Sequence[PreparedItem]
) -> dict[str, list[str]]:
    """``SEGMENTATION_FAILED`` for every approved item that ended up without a usable mask.

    Call it with the items built from ``outcome.masks``; the result is keyed by detection id
    (so ids must already be assigned) and merges into the assembler's ``warnings_by_id``.
    """
    warnings: dict[str, list[str]] = {}
    for i in outcome.approved:
        if items[i].mask is not None:
            continue
        det_id = items[i].detection.detection_id
        if det_id is None:
            raise ValueError("detection ids must be assigned before segmentation_warnings")
        warnings[det_id] = [WarningCode.SEGMENTATION_FAILED.value]
    return warnings


def pad_and_clamp(bbox: BBox, padding_ratio: float, image_w: int, image_h: int) -> BBox:
    """Expand by ``padding_ratio`` of the box width/height per side, clamp, keep >= 1 px."""
    x1, y1, x2, y2 = bbox
    padded = sv.pad_boxes(
        np.array([bbox]),
        px=round((x2 - x1) * padding_ratio),
        py=round((y2 - y1) * padding_ratio),
    )
    nx1, ny1, nx2, ny2 = (int(v) for v in sv.clip_boxes(padded, (image_w, image_h))[0])
    nx1, ny1 = min(nx1, image_w - 1), min(ny1, image_h - 1)  # a crop is never empty
    return (nx1, ny1, max(nx2, nx1 + 1), max(ny2, ny1 + 1))


def downscale_to_max_side(image: np.ndarray, max_side: int) -> np.ndarray:
    """A new array no larger than ``max_side`` on its longer side (never upscales)."""
    thumbnail = Image.fromarray(image)
    thumbnail.thumbnail((max_side, max_side))
    return np.array(thumbnail)


def _usable_mask(mask: object, image_h: int, image_w: int, box: BBox) -> np.ndarray | None:
    """A bool ``(H, W)`` mask with at least one pixel inside ``box``, else ``None``."""
    if mask is None:
        return None
    if not isinstance(mask, np.ndarray) or mask.shape != (image_h, image_w):
        log.warning("ignoring mask with unexpected shape: %s", getattr(mask, "shape", type(mask)))
        return None
    mask = mask.astype(bool, copy=False)
    x1, y1, x2, y2 = box
    if not mask[y1:y2, x1:x2].any():
        log.warning("ignoring empty mask for box %s", box)
        return None
    return mask


def build_prepared_items(
    image_rgb: np.ndarray,
    dets: Sequence[RawDetection],
    masks: Sequence[np.ndarray | None],
    crop_cfg: CropConfig,
    crop_transforms: Sequence[CropTransform] = (),
) -> list[PreparedItem]:
    """Build one ``PreparedItem`` per detection, in the order given."""
    image_h, image_w = image_rgb.shape[:2]
    context = None
    if crop_cfg.include_full_image_context:
        context = downscale_to_max_side(image_rgb, crop_cfg.max_image_side_px)
        context.setflags(write=False)  # shared by every item

    items: list[PreparedItem] = []
    for idx, det in enumerate(dets):
        padded = pad_and_clamp(det.bbox_xyxy, crop_cfg.padding_ratio, image_w, image_h)
        x1, y1, x2, y2 = padded
        crop = image_rgb[y1:y2, x1:x2].copy()

        mask = _usable_mask(masks[idx] if idx < len(masks) else None, image_h, image_w, padded)
        if mask is not None:
            crop[~mask[y1:y2, x1:x2]] = MASK_FILL

        for transform in crop_transforms:
            crop = transform.apply(crop)
        crop = downscale_to_max_side(crop, crop_cfg.max_image_side_px)

        items.append(
            PreparedItem(
                detection=det,
                padded_bbox_xyxy=padded,
                crop_rgb=crop,
                context_rgb=context,
                mask=mask,
            )
        )
    return items
