"""Component protocols and the context objects handed to components.

Every component class defines an inner ``Params`` Pydantic model (``extra="forbid"``) and is
constructed as ``Component(params, build_ctx)``.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Protocol

import numpy as np

from .schemas import ClassifiedItem, PreparedItem, RawDetection

Attributes = dict[str, Any]  # raw VLM attributes: brand, model, category, color


def resolve_device(requested: str) -> str:
    """``auto`` resolves to cuda, then mps, then cpu. Never requires torch."""
    if requested != "auto":
        return requested
    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


@dataclass(frozen=True)
class BuildContext:
    """Shared, read-only build-time facts. The vocabulary lives here and nowhere else."""

    categories: tuple[str, ...]
    device: str
    seed: int
    config: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_config(cls, resolved: Mapping[str, Any]) -> BuildContext:
        return cls(
            categories=tuple(resolved["vocabulary"]["categories"]),
            device=resolve_device(resolved["run"]["device"]),
            seed=int(resolved["run"]["seed"]),
            config=MappingProxyType(copy.deepcopy(dict(resolved))),
        )


@dataclass(frozen=True)
class FilterContext:
    image_w: int
    image_h: int


@dataclass(frozen=True)
class ItemContext:
    """What post-processors and warning rules may inspect for one item."""

    item: PreparedItem
    index: int  # position of this item's detection within ``all_detections``
    all_detections: Sequence[RawDetection]
    image_w: int
    image_h: int


@dataclass
class PreprocessState:
    """Per-run state shared by ``image_preprocess`` steps.

    ``scale_*``/``offset_*`` accumulate the mapping original -> current:
    ``current = original * scale + offset``. Each geometric step also records its own
    ``(scale_x, scale_y, offset_x, offset_y)`` under ``records[key]`` so that
    ``map_boxes_back`` can undo exactly that step. Steps keep no per-run data on ``self``.
    """

    original_h: int
    original_w: int
    scale_x: float = 1.0
    scale_y: float = 1.0
    offset_x: float = 0.0
    offset_y: float = 0.0
    records: dict[str, tuple[float, float, float, float]] = field(default_factory=dict)


class ImageTransform(Protocol):
    def apply(self, image: np.ndarray, state: PreprocessState) -> np.ndarray: ...

    def map_boxes_back(self, boxes_xyxy: np.ndarray, state: PreprocessState) -> np.ndarray:
        """Identity for non-geometric steps."""
        ...


class Detector(Protocol):
    def predict(self, image_rgb: np.ndarray) -> list[RawDetection]:
        """No filtering; only a low floor score."""
        ...


class DetectionFilter(Protocol):
    def apply(self, dets: list[RawDetection], ctx: FilterContext) -> list[RawDetection]: ...


class Segmenter(Protocol):
    def segment(self, image_rgb: np.ndarray, boxes_xyxy: np.ndarray) -> list[np.ndarray | None]:
        """Bool masks in the original frame, or None where no mask was produced."""
        ...


class SegmentationGate(Protocol):
    def should_segment(self, item_idx: int, dets: Sequence[RawDetection]) -> bool: ...


class CropTransform(Protocol):
    def apply(self, crop_rgb: np.ndarray) -> np.ndarray: ...


class VLMClient(Protocol):
    def generate(self, system_prompt: str, images: list[np.ndarray], user_text: str) -> str: ...


class AttributePostprocessor(Protocol):
    def apply(self, attrs: Attributes, item_ctx: ItemContext) -> tuple[Attributes, list[str]]: ...


class WarningRule(Protocol):
    def evaluate(self, draft: ClassifiedItem, item_ctx: ItemContext) -> list[str]: ...
