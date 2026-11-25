"""Pipeline trace: what every stage produced, for debugging.

Only created when ``run(..., return_trace=True)``; otherwise nothing is recorded, so there is
no memory cost. Images are stored by reference, not copied.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .interfaces import Attributes
from .schemas import PreparedItem, RawDetection


@dataclass
class PipelineTrace:
    preprocessed_image: np.ndarray | None = None
    # exactly what the detector returned, in the pre-processed image's pixels
    raw_detections: list[RawDetection] = field(default_factory=list)
    # mapped back to original pixels, clamped and validated (the input of the filters)
    mapped_detections: list[RawDetection] = field(default_factory=list)
    # (filter name, detections after that filter) in the configured order
    filter_steps: list[tuple[str, list[RawDetection]]] = field(default_factory=list)
    detections: list[RawDetection] = field(default_factory=list)  # final, with ids
    prepared_items: list[PreparedItem] = field(default_factory=list)  # crops, masks, context
    # ids of items the VLM judged not electronic and that were dropped (vlm.on_not_electronic)
    dropped_not_electronic: list[str] = field(default_factory=list)
    vlm_replies: dict[str, list[str]] = field(default_factory=dict)  # item id -> raw VLM text
    attributes_before: dict[str, Attributes] = field(default_factory=dict)
    # item id -> attributes after each post-processor, in order
    attributes_after: dict[str, list[Attributes]] = field(default_factory=dict)
