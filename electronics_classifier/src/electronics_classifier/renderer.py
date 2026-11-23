"""Renderer (fixed logic, styled by config): boxes, optional masks and a label banner per item.

Boxes and masks are drawn with supervision; banners are placed by ``plan_banners`` (a pure
function) and drawn with OpenCV text primitives. Everything is deterministic.
"""

from __future__ import annotations

import logging
import string
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import cv2
import numpy as np
import supervision as sv

from .config import ConfigError
from .schemas import ClassifiedItem

log = logging.getLogger(__name__)

LABEL_FIELDS = frozenset({"id", "category", "brand", "model", "color", "confidence"})
MAX_LABEL_CHARS = 80
ELLIPSIS = "..."
FONT = cv2.FONT_HERSHEY_SIMPLEX
PALETTE = sv.ColorPalette.DEFAULT  # fixed order: item i always gets colour i


@dataclass(frozen=True)
class RenderStyle:
    label_template: str
    stroke_scale: float = 1.0
    font_scale: float = 1.0
    mask_alpha: float = 0.35

    @classmethod
    def from_config(cls, rendering: Mapping) -> RenderStyle:
        return cls(
            label_template=rendering["label_template"],
            stroke_scale=float(rendering["stroke_scale"]),
            font_scale=float(rendering["font_scale"]),
            mask_alpha=float(rendering["mask_alpha"]),
        )


@dataclass(frozen=True)
class Banner:
    """A laid-out label: ``(x, y)`` is its top-left corner, all inside the image."""

    item_id: str
    text: str
    x: int
    y: int
    w: int
    h: int
    baseline: int


# --- label text ----------------------------------------------------------------------


def validate_label_template(template: str) -> str:
    """Reject unknown placeholders (and positional/attribute access) at build time."""
    try:
        fields = [name for _, name, _, _ in string.Formatter().parse(template) if name is not None]
    except ValueError as exc:
        raise ConfigError(f"rendering.label_template: invalid format string ({exc})") from None
    unknown = sorted({name for name in fields if name not in LABEL_FIELDS})
    if unknown:
        allowed = ", ".join(sorted(LABEL_FIELDS))
        raise ConfigError(
            f"rendering.label_template: unknown placeholder(s) {unknown}; allowed: {allowed}"
        )
    try:
        template.format(id="", category="", brand="", model="", color="", confidence=0.0)
    except (ValueError, KeyError, IndexError) as exc:
        raise ConfigError(f"rendering.label_template: invalid format spec ({exc})") from None
    return template


def _ascii_only(text: str) -> str:
    """OpenCV fonts cannot draw non-ASCII or control characters: show them as ``?``."""
    return "".join(ch if " " <= ch <= "~" else "?" for ch in text)


def format_label(item: ClassifiedItem, template: str) -> str:
    """The one place label text is made. Safe for any attribute text; never raises."""
    try:
        text = template.format(
            id=item.id,
            category=item.attributes.category,
            brand=item.attributes.brand,
            model=item.attributes.model,
            color=item.attributes.color,
            confidence=item.detector_confidence,
        )
    except (ValueError, KeyError, IndexError, AttributeError):
        log.warning("label template failed for %s; using the id", item.id)
        text = item.id
    text = _ascii_only(text)
    if item.warnings:
        text = f"! {text}"
    if len(text) > MAX_LABEL_CHARS:
        text = text[: MAX_LABEL_CHARS - len(ELLIPSIS)] + ELLIPSIS
    return text


# --- layout --------------------------------------------------------------------------


def _scales(image_h: int, image_w: int, style: RenderStyle) -> tuple[int, float]:
    longest = max(image_h, image_w)
    thickness = max(1, round(longest / 400 * style.stroke_scale))
    font_scale = max(0.3, longest / 1500 * style.font_scale)
    return thickness, font_scale


def _fit_text(text: str, max_width: int, font_scale: float, thickness: int) -> str:
    """Shorten ``text`` (with an ellipsis) until it fits ``max_width`` pixels."""
    while len(text) > len(ELLIPSIS):
        if cv2.getTextSize(text, FONT, font_scale, thickness)[0][0] <= max_width:
            return text
        text = text[: -len(ELLIPSIS) - 1].rstrip() + ELLIPSIS
    return text


def _collides(x: int, y: int, w: int, h: int, placed: Sequence[Banner]) -> bool:
    return any(x < b.x + b.w and b.x < x + w and y < b.y + b.h and b.y < y + h for b in placed)


def plan_banners(
    items: Sequence[ClassifiedItem], image_h: int, image_w: int, style: RenderStyle
) -> list[Banner]:
    """Place one banner per item: above its box, else inside the box top, always in frame.

    A banner that collides with an earlier one is staggered vertically, trying downwards
    and upwards in banner-height steps, then falls back to the first position.
    """
    thickness, font_scale = _scales(image_h, image_w, style)
    pad = max(2, thickness + 1)
    banners: list[Banner] = []
    for item in items:
        text = _fit_text(
            format_label(item, style.label_template), image_w - 2 * pad, font_scale, thickness
        )
        (tw, th), baseline = cv2.getTextSize(text, FONT, font_scale, thickness)
        w = min(tw + 2 * pad, image_w)
        h = min(th + baseline + 2 * pad, image_h)
        x1, y1, _, _ = item.bbox_xyxy
        x = min(max(x1, 0), image_w - w)

        def clamp_y(value: int, h: int = h) -> int:
            return min(max(value, 0), image_h - h)

        preferred = y1 - h if y1 - h >= 0 else y1  # above the box, else inside its top edge
        candidates = [clamp_y(preferred)]
        for step in range(1, image_h // max(h, 1) + 1):
            candidates += [clamp_y(preferred + step * h), clamp_y(preferred - step * h)]
        y = next((c for c in candidates if not _collides(x, c, w, h, banners)), candidates[0])
        banners.append(Banner(item.id, text, x, y, w, h, baseline))
    return banners


# --- drawing -------------------------------------------------------------------------


def _array_color(index: int) -> sv.Color:
    """supervision draws BGR; our arrays are RGB, so swap red and blue."""
    c = PALETTE.by_idx(index)
    return sv.Color(r=c.b, g=c.g, b=c.r)


def _text_color(index: int) -> tuple[int, int, int]:
    c = PALETTE.by_idx(index)
    luminance = 0.299 * c.r + 0.587 * c.g + 0.114 * c.b
    return (0, 0, 0) if luminance > 140 else (255, 255, 255)


def render_annotated_image(
    original_image_rgb: np.ndarray,
    classified_items: Sequence[ClassifiedItem],
    masks_by_id: Mapping[str, np.ndarray | None],
    style: RenderStyle,
) -> np.ndarray:
    """A new image with masks, boxes and banners; the input is never modified."""
    canvas = original_image_rgb.copy()
    items = list(classified_items)
    if not items:
        return canvas
    image_h, image_w = canvas.shape[:2]
    thickness, font_scale = _scales(image_h, image_w, style)

    palette = sv.ColorPalette([_array_color(i) for i in range(len(items))])
    detections = sv.Detections(
        xyxy=np.array([item.bbox_xyxy for item in items], dtype=np.float64),
        class_id=np.arange(len(items)),
    )

    masked = [
        (i, mask)
        for i, item in enumerate(items)
        if isinstance(mask := masks_by_id.get(item.id), np.ndarray)
        and mask.shape == (image_h, image_w)
        and mask.any()
    ]
    if masked and style.mask_alpha > 0:
        mask_detections = sv.Detections(
            xyxy=detections.xyxy[[i for i, _ in masked]],
            mask=np.stack([m.astype(bool) for _, m in masked]),
            class_id=np.array([i for i, _ in masked]),
        )
        canvas = sv.MaskAnnotator(
            color=palette, opacity=style.mask_alpha, color_lookup=sv.ColorLookup.CLASS
        ).annotate(canvas, mask_detections)

    canvas = sv.BoxAnnotator(
        color=palette, thickness=thickness, color_lookup=sv.ColorLookup.CLASS
    ).annotate(canvas, detections)

    for index, banner in enumerate(plan_banners(items, image_h, image_w, style)):
        r, g, b = PALETTE.by_idx(index).as_rgb()
        top_left, bottom_right = (banner.x, banner.y), (banner.x + banner.w, banner.y + banner.h)
        cv2.rectangle(canvas, top_left, bottom_right, (r, g, b), thickness=cv2.FILLED)
        pad = max(2, thickness + 1)
        cv2.putText(
            canvas,
            banner.text,
            (banner.x + pad, banner.y + banner.h - pad - banner.baseline),
            FONT,
            font_scale,
            _text_color(index),
            thickness,
            cv2.LINE_AA,
        )
    return canvas
