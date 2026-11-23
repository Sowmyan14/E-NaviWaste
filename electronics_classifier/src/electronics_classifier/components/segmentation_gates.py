"""Segmentation gates (kind ``segmentation_gate``): which items get a SAM mask."""

from __future__ import annotations

from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict, Field

from ..interfaces import BuildContext
from ..registry import register
from ..schemas import BBox, RawDetection


def box_iou(a: BBox, b: BBox) -> float:
    """Intersection over union of two ``x1,y1,x2,y2`` boxes (0.0 for empty boxes)."""
    ix = min(a[2], b[2]) - max(a[0], b[0])
    iy = min(a[3], b[3]) - max(a[1], b[1])
    if ix <= 0 or iy <= 0:
        return 0.0
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def max_overlap(idx: int, dets: Sequence[RawDetection]) -> float:
    """Highest IoU between detection ``idx`` and any other detection."""
    return max(
        (box_iou(dets[idx].bbox_xyxy, d.bbox_xyxy) for j, d in enumerate(dets) if j != idx),
        default=0.0,
    )


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
        return max_overlap(item_idx, dets) >= self.params.iou
