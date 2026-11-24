"""Detection filters (kind ``detection_filter``) plus the fixed post-detector steps.

``sanitize_detections`` always runs right after the detector (clamping lives there, not in a
filter) and ``assign_ids`` runs after the configured filters.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Collection, Iterable

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from ..interfaces import BuildContext, FilterContext
from ..registry import register
from ..schemas import RawDetection

log = logging.getLogger(__name__)


def _rank_key(det: RawDetection) -> tuple:
    """Total order: highest confidence first, ties broken by position and phrase."""
    x1, y1, x2, y2 = det.bbox_xyxy
    return (-det.confidence, y1, x1, y2, x2, det.phrase)


def sanitize_detections(
    dets: Iterable[RawDetection],
    image_w: int,
    image_h: int,
    vocabulary: Collection[str] | None = None,
) -> list[RawDetection]:
    """Clamp boxes to the image and drop anything unusable. Never raises.

    Dropped: non-finite coordinates, non-finite or out-of-[0,1] confidence, boxes with no area
    after clamping, and (when ``vocabulary`` is given) phrases outside the vocabulary.
    """
    kept: list[RawDetection] = []
    for det in dets:
        try:
            coords = [float(v) for v in det.bbox_xyxy]
            confidence = float(det.confidence)
            phrase = str(det.phrase)
        except (TypeError, ValueError):
            log.warning("dropping detection with unreadable fields: %r", det)
            continue
        if len(coords) != 4 or not all(math.isfinite(v) for v in coords):
            log.warning("dropping detection with non-finite box: %r", det)
            continue
        if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            log.warning("dropping detection with invalid confidence %r", det.confidence)
            continue
        if vocabulary is not None and phrase not in vocabulary:
            log.warning("dropping detection with phrase '%s' not in the vocabulary", phrase)
            continue
        x1 = min(max(round(coords[0]), 0), image_w)
        y1 = min(max(round(coords[1]), 0), image_h)
        x2 = min(max(round(coords[2]), 0), image_w)
        y2 = min(max(round(coords[3]), 0), image_h)
        if x2 <= x1 or y2 <= y1:
            log.warning("dropping detection with empty box after clamping: %r", det)
            continue
        kept.append(
            RawDetection(
                detection_id=det.detection_id,
                phrase=phrase,
                confidence=confidence,
                bbox_xyxy=(x1, y1, x2, y2),
            )
        )
    return kept


def assign_ids(dets: Iterable[RawDetection]) -> list[RawDetection]:
    """Order by ``(y1, x1)`` and number them ``obj_1..obj_N`` so runs are deterministic."""

    def order(det: RawDetection) -> tuple:
        x1, y1, x2, y2 = det.bbox_xyxy
        return (y1, x1, y2, x2, det.phrase, -det.confidence)

    ordered = sorted(dets, key=order)
    return [
        det.model_copy(update={"detection_id": f"obj_{i}"}) for i, det in enumerate(ordered, 1)
    ]


def _boxes(dets: list[RawDetection]) -> np.ndarray:
    return np.array([d.bbox_xyxy for d in dets], dtype=np.float64).reshape(-1, 4)


class _Filter:
    """Shared constructor for filters."""

    def __init__(self, params: BaseModel, build: BuildContext) -> None:
        self.params = params


@register("detection_filter", "score_threshold")
class ScoreThreshold(_Filter):
    """Keep detections whose confidence is at least ``min_score``."""

    class Params(BaseModel):
        model_config = ConfigDict(extra="forbid")
        min_score: float = Field(default=0.25, ge=0.0, le=1.0, description="Minimum confidence.")

    def apply(self, dets: list[RawDetection], ctx: FilterContext) -> list[RawDetection]:
        return [d for d in dets if d.confidence >= self.params.min_score]


@register("detection_filter", "min_size")
class MinSize(_Filter):
    """Keep boxes at least ``min_width_px`` by ``min_height_px``."""

    class Params(BaseModel):
        model_config = ConfigDict(extra="forbid")
        min_width_px: int = Field(default=32, ge=0, description="Minimum box width, px.")
        min_height_px: int = Field(default=32, ge=0, description="Minimum box height, px.")

    def apply(self, dets: list[RawDetection], ctx: FilterContext) -> list[RawDetection]:
        kept = []
        for d in dets:
            x1, y1, x2, y2 = d.bbox_xyxy
            if x2 - x1 >= self.params.min_width_px and y2 - y1 >= self.params.min_height_px:
                kept.append(d)
        return kept


@register("detection_filter", "min_area_ratio")
class MinAreaRatio(_Filter):
    """Keep boxes covering at least ``min_ratio`` of the image area."""

    class Params(BaseModel):
        model_config = ConfigDict(extra="forbid")
        min_ratio: float = Field(
            default=0.002, ge=0.0, le=1.0, description="Minimum box area / image area."
        )

    def apply(self, dets: list[RawDetection], ctx: FilterContext) -> list[RawDetection]:
        image_area = max(ctx.image_w * ctx.image_h, 1)
        kept = []
        for d in dets:
            x1, y1, x2, y2 = d.bbox_xyxy
            if (x2 - x1) * (y2 - y1) / image_area >= self.params.min_ratio:
                kept.append(d)
        return kept


@register("detection_filter", "full_frame_guard")
class FullFrameGuard(_Filter):
    """Handle boxes that cover (almost) the whole image.

    A generic phrase ("electronic device", "appliance") can match the entire photo. Such a box
    localizes nothing, and in a later ``containment`` step it would swallow every real box
    inside it. So: if other, smaller boxes exist, the full-frame boxes are dropped; if every
    box is full-frame, only the most confident one is kept (it becomes a whole-image item).
    """

    class Params(BaseModel):
        model_config = ConfigDict(extra="forbid")
        min_ratio: float = Field(
            default=0.90, gt=0.0, le=1.0, description="Box area / image area that counts as full frame."
        )  # fmt: skip

    def apply(self, dets: list[RawDetection], ctx: FilterContext) -> list[RawDetection]:
        image_area = max(ctx.image_w * ctx.image_h, 1)

        def is_full(d: RawDetection) -> bool:
            x1, y1, x2, y2 = d.bbox_xyxy
            return (x2 - x1) * (y2 - y1) / image_area >= self.params.min_ratio

        full = [d for d in dets if is_full(d)]
        if not full:
            return list(dets)
        rest = [d for d in dets if not is_full(d)]
        return rest or [min(full, key=_rank_key)]


@register("detection_filter", "nms")
class Nms(_Filter):
    """Non-maximum suppression (via ``supervision``): drop boxes with IoU > threshold."""

    class Params(BaseModel):
        model_config = ConfigDict(extra="forbid")
        iou_threshold: float = Field(
            default=0.5, ge=0.0, le=1.0, description="Suppress boxes whose IoU exceeds this."
        )
        class_agnostic: bool = Field(
            default=True, description="If true, boxes with different phrases also suppress."
        )

    def __init__(self, params: Params, build: BuildContext) -> None:
        super().__init__(params, build)
        import supervision as sv

        self._sv = sv

    def apply(self, dets: list[RawDetection], ctx: FilterContext) -> list[RawDetection]:
        if len(dets) < 2:
            return list(dets)
        ranked = sorted(dets, key=_rank_key)  # canonical order => input order never matters
        phrases = sorted({d.phrase for d in ranked})
        class_ids = np.array([phrases.index(d.phrase) for d in ranked], dtype=int)
        detections = self._sv.Detections(
            xyxy=_boxes(ranked),
            confidence=np.array([d.confidence for d in ranked], dtype=np.float64),
            class_id=class_ids,
            data={"rank": np.arange(len(ranked))},
        )
        survivors = detections.with_nms(
            threshold=self.params.iou_threshold, class_agnostic=self.params.class_agnostic
        )
        return [ranked[int(i)] for i in survivors.data["rank"]]


@register("detection_filter", "containment")
class Containment(_Filter):
    """Drop a box that lies (almost) entirely inside a strictly larger box.

    NMS compares IoU, which is tiny when a small box sits inside a big one, so parts of a
    device (camera lens, label, button panel) survive it as separate items. This filter uses
    intersection over the *smaller* box instead. The decision does not depend on confidence
    or phrase, because the detector often labels a part with an unrelated phrase. Side effect:
    a real device lying fully inside another (a charger on a power strip's box) is dropped too;
    use ``containers`` to limit which phrases may swallow other boxes.
    """

    class Params(BaseModel):
        model_config = ConfigDict(extra="forbid")
        min_containment: float = Field(
            default=0.9, gt=0.0, le=1.0,
            description="Drop a box when this fraction of its area is inside a larger box.",
        )  # fmt: skip
        containers: list[str] = Field(
            default_factory=list,
            description="Only boxes with these phrases may swallow others; empty = any phrase.",
        )

    def apply(self, dets: list[RawDetection], ctx: FilterContext) -> list[RawDetection]:
        if len(dets) < 2:
            return list(dets)
        boxes = _boxes(dets)
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        allowed = set(self.params.containers)
        kept = []
        for i, det in enumerate(dets):
            swallowed = False
            for j, other in enumerate(dets):
                if i == j or areas[j] <= areas[i] or (allowed and other.phrase not in allowed):
                    continue
                w = min(boxes[i, 2], boxes[j, 2]) - max(boxes[i, 0], boxes[j, 0])
                h = min(boxes[i, 3], boxes[j, 3]) - max(boxes[i, 1], boxes[j, 1])
                if w > 0 and h > 0 and w * h / areas[i] >= self.params.min_containment:
                    swallowed = True
                    break
            if not swallowed:
                kept.append(det)
        return kept


@register("detection_filter", "top_k")
class TopK(_Filter):
    """Keep the ``k`` most confident detections."""

    class Params(BaseModel):
        model_config = ConfigDict(extra="forbid")
        k: int = Field(default=20, ge=0, description="Maximum number of detections to keep.")

    def apply(self, dets: list[RawDetection], ctx: FilterContext) -> list[RawDetection]:
        return sorted(dets, key=_rank_key)[: self.params.k]
