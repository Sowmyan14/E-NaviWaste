"""Segmentation gates (kind ``segmentation_gate``): which items get a SAM mask."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from ..interfaces import BuildContext
from ..registry import register
from ..schemas import RawDetection


def overlaps_other(idx: int, dets: Sequence[RawDetection], min_iou: float) -> bool:
    """True if detection ``idx`` intersects another one with IoU of at least ``min_iou``."""
    if len(dets) < 2:
        return False
    import supervision as sv

    boxes = np.array([d.bbox_xyxy for d in dets], dtype=np.float64)
    ious = sv.box_iou_batch(boxes[idx : idx + 1], boxes)[0]
    ious[idx] = 0.0  # a box never overlaps itself
    return bool(np.any((ious > 0.0) & (ious >= min_iou)))


class _NoParams(BaseModel):
    model_config = ConfigDict(extra="forbid")


@register("segmentation_gate", "always")
class AlwaysGate:
    """Segment every item."""

    Params = _NoParams

    def __init__(self, params: _NoParams, build: BuildContext) -> None:
        self.params = params

    def should_segment(self, item_idx: int, dets: Sequence[RawDetection]) -> bool:
        return True


@register("segmentation_gate", "overlap")
class OverlapGate:
    """Segment only items whose box overlaps another detection (IoU >= ``iou``)."""

    class Params(BaseModel):
        model_config = ConfigDict(extra="forbid")
        iou: float = Field(
            default=0.15, ge=0.0, le=1.0, description="Segment items overlapping another by IoU."
        )

    def __init__(self, params: Params, build: BuildContext) -> None:
        self.params = params

    def should_segment(self, item_idx: int, dets: Sequence[RawDetection]) -> bool:
        return overlaps_other(item_idx, dets, self.params.iou)
