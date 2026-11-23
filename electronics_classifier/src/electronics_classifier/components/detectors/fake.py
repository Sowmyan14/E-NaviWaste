"""Fake detector for offline smoke tests (``EC_USE_FAKES=1``) and config experiments."""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from ...interfaces import BuildContext
from ...registry import register
from ...schemas import RawDetection


class FakeBox(BaseModel):
    model_config = ConfigDict(extra="forbid")
    phrase: str = Field(default="router", description="Phrase reported for this box.")
    confidence: float = Field(default=0.9, ge=0.0, le=1.0, description="Reported confidence.")
    box_rel: tuple[float, float, float, float] = Field(
        default=(0.25, 0.25, 0.75, 0.75),
        description="x1, y1, x2, y2 as fractions of the image width/height.",
    )


@register("detector", "fake")
class FakeDetector:
    """Returns fixed boxes, placed relative to the image size. Not a real detector."""

    class Params(BaseModel):
        model_config = ConfigDict(extra="forbid")
        detections: list[FakeBox] = Field(
            default_factory=lambda: [FakeBox()],
            description="Boxes to return on every call (default: one centred 'router').",
        )

    def __init__(self, params: Params, build: BuildContext) -> None:
        self.params = params

    def predict(self, image_rgb: np.ndarray) -> list[RawDetection]:
        h, w = image_rgb.shape[:2]
        found = []
        for item in self.params.detections:
            fx1, fy1, fx2, fy2 = item.box_rel
            found.append(
                RawDetection(
                    phrase=item.phrase,
                    confidence=item.confidence,
                    bbox_xyxy=(
                        round(fx1 * w),
                        round(fy1 * h),
                        round(fx2 * w),
                        round(fy2 * h),
                    ),
                )
            )
        return found
