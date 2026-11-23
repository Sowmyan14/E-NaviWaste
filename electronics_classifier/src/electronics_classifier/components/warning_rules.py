"""Warning rules (kind ``warning_rule``): each returns the warning codes it wants to add."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from ..config import FALLBACK_CATEGORY
from ..interfaces import BuildContext, ItemContext
from ..registry import register
from ..schemas import ClassifiedItem, WarningCode
from .attribute_postprocess import UNKNOWN
from .segmentation_gates import overlaps_other


class _NoParams(BaseModel):
    model_config = ConfigDict(extra="forbid")


@register("warning_rule", "unknown_field")
class UnknownField:
    """BRAND_UNKNOWN / MODEL_UNKNOWN / COLOR_UNKNOWN when that field is "unknown"."""

    Params = _NoParams

    _CODES = (
        ("brand", WarningCode.BRAND_UNKNOWN),
        ("model", WarningCode.MODEL_UNKNOWN),
        ("color", WarningCode.COLOR_UNKNOWN),
    )

    def __init__(self, params: _NoParams, build: BuildContext) -> None:
        self.params = params

    def evaluate(self, draft: ClassifiedItem, item_ctx: ItemContext) -> list[str]:
        return [
            code.value for name, code in self._CODES if getattr(draft.attributes, name) == UNKNOWN
        ]


@register("warning_rule", "low_detector_confidence")
class LowDetectorConfidence:
    """LOW_DETECTOR_CONFIDENCE when the detector score is below ``threshold``."""

    class Params(BaseModel):
        model_config = ConfigDict(extra="forbid")
        threshold: float = Field(
            default=0.40, ge=0.0, le=1.0, description="Warn below this detector confidence."
        )

    def __init__(self, params: Params, build: BuildContext) -> None:
        self.params = params

    def evaluate(self, draft: ClassifiedItem, item_ctx: ItemContext) -> list[str]:
        if draft.detector_confidence < self.params.threshold:
            return [WarningCode.LOW_DETECTOR_CONFIDENCE.value]
        return []


@register("warning_rule", "small_crop")
class SmallCrop:
    """SMALL_CROP when the padded crop's shorter side is under ``min_side_px``."""

    class Params(BaseModel):
        model_config = ConfigDict(extra="forbid")
        min_side_px: int = Field(
            default=64, ge=0, description="Warn when the padded crop's shorter side is below this."
        )

    def __init__(self, params: Params, build: BuildContext) -> None:
        self.params = params

    def evaluate(self, draft: ClassifiedItem, item_ctx: ItemContext) -> list[str]:
        x1, y1, x2, y2 = item_ctx.item.padded_bbox_xyxy
        if min(x2 - x1, y2 - y1) < self.params.min_side_px:
            return [WarningCode.SMALL_CROP.value]
        return []


@register("warning_rule", "overlap")
class Overlap:
    """PARTIALLY_OCCLUDED_OR_OVERLAPPING when the box overlaps another by IoU >= ``iou``."""

    class Params(BaseModel):
        model_config = ConfigDict(extra="forbid")
        iou: float = Field(
            default=0.15, ge=0.0, le=1.0, description="Warn when IoU with another box reaches this."
        )

    def __init__(self, params: Params, build: BuildContext) -> None:
        self.params = params

    def evaluate(self, draft: ClassifiedItem, item_ctx: ItemContext) -> list[str]:
        if overlaps_other(item_ctx.index, item_ctx.all_detections, self.params.iou):
            return [WarningCode.PARTIALLY_OCCLUDED_OR_OVERLAPPING.value]
        return []


@register("warning_rule", "category_oov")
class CategoryOov:
    """CATEGORY_OUT_OF_VOCAB when the item's category is not in the vocabulary.

    Not in the default config: `category_vocab` already fixes categories, so this only matters
    when that post-processor is removed.
    """

    Params = _NoParams

    def __init__(self, params: _NoParams, build: BuildContext) -> None:
        self.params = params
        self.categories = frozenset(build.categories)

    def evaluate(self, draft: ClassifiedItem, item_ctx: ItemContext) -> list[str]:
        if draft.attributes.category not in self.categories:
            return [WarningCode.CATEGORY_OUT_OF_VOCAB.value]
        return []


@register("warning_rule", "category_mismatch")
class CategoryMismatch:
    """CATEGORY_MISMATCH when the detector's phrase and the VLM's category are different.

    Only two *specific* categories can disagree: when either side is the generic
    ``electronic device`` (for instance after a VLM fallback) nothing is flagged.
    """

    Params = _NoParams

    def __init__(self, params: _NoParams, build: BuildContext) -> None:
        self.params = params

    def evaluate(self, draft: ClassifiedItem, item_ctx: ItemContext) -> list[str]:
        phrase = " ".join(draft.detector_phrase.lower().split())
        category = " ".join(draft.attributes.category.lower().split())
        if FALLBACK_CATEGORY in (phrase, category) or phrase == category:
            return []
        return [WarningCode.CATEGORY_MISMATCH.value]
