from enum import Enum
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

BBox = tuple[int, int, int, int]  # x1,y1,x2,y2 in ORIGINAL image pixels, x2>x1, y2>y1


class RawDetection(BaseModel):
    # assigned after filtering: "obj_1", "obj_2" ordered by (y1, x1)
    detection_id: str | None = None
    phrase: str
    confidence: float = Field(ge=0.0, le=1.0)
    bbox_xyxy: BBox


class PreparedItem(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    detection: RawDetection
    padded_bbox_xyxy: BBox  # padded and clamped
    crop_rgb: np.ndarray  # (h,w,3) uint8, after crop_transforms
    context_rgb: np.ndarray | None = None
    mask: np.ndarray | None = None  # (H,W) bool, original frame


class SchemaOrgProductAttributes(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    context: str = Field(default="https://schema.org", alias="@context")
    type: str = Field(default="Product", alias="@type")
    brand: str = "unknown"
    model: str = "unknown"
    category: str = "unknown"
    color: str = "unknown"
    # serialize with model_dump(by_alias=True)


class WarningCode(str, Enum):  # built-in codes; custom rules may emit any other str
    BRAND_UNKNOWN = "BRAND_UNKNOWN"
    MODEL_UNKNOWN = "MODEL_UNKNOWN"
    COLOR_UNKNOWN = "COLOR_UNKNOWN"
    LOW_DETECTOR_CONFIDENCE = "LOW_DETECTOR_CONFIDENCE"
    CATEGORY_OUT_OF_VOCAB = "CATEGORY_OUT_OF_VOCAB"
    VLM_PARSE_FAILED = "VLM_PARSE_FAILED"
    SMALL_CROP = "SMALL_CROP"
    PARTIALLY_OCCLUDED_OR_OVERLAPPING = "PARTIALLY_OCCLUDED_OR_OVERLAPPING"


class ClassifiedItem(BaseModel):
    id: str
    bbox_xyxy: list[int]
    detector_phrase: str
    detector_confidence: float
    attributes: SchemaOrgProductAttributes
    warnings: list[str] = Field(default_factory=list)


class RunMeta(BaseModel):
    config_hash: str
    components: dict[str, str | list[str]]  # kind -> component name(s)


class ClassificationResult(BaseModel):
    image_width: int
    image_height: int
    items: list[ClassifiedItem]
    annotated_image_format: Literal["png", "jpeg"] = "png"
    meta: RunMeta | None = None
