"""Assembler (fixed logic): turns prepared items + attributes + warnings into the result."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from itertools import chain
from typing import Any, Literal

from .interfaces import Attributes, ItemContext, WarningRule
from .schemas import (
    ClassificationResult,
    ClassifiedItem,
    PreparedItem,
    RunMeta,
    SchemaOrgProductAttributes,
)

_FIELDS = ("brand", "model", "category", "color")


def _product_attributes(attributes: Mapping[str, Any]) -> SchemaOrgProductAttributes:
    return SchemaOrgProductAttributes(
        **{name: str(attributes.get(name, "unknown")) for name in _FIELDS}
    )


def unique_in_order(*warning_lists: Iterable[str]) -> list[str]:
    """Warnings de-duplicated, keeping the order in which they first appeared."""
    return list(dict.fromkeys(chain.from_iterable(warning_lists)))


def make_draft(item: PreparedItem, attributes: Attributes) -> ClassifiedItem:
    """The item as the warning rules see it (no warnings yet)."""
    det = item.detection
    return ClassifiedItem(
        id=_require_id(item),
        bbox_xyxy=list(det.bbox_xyxy),  # the unpadded detector box, original pixels
        detector_phrase=det.phrase,
        detector_confidence=det.confidence,
        attributes=_product_attributes(attributes),
    )


def evaluate_warning_rules(
    rules: Sequence[WarningRule], draft: ClassifiedItem, item_ctx: ItemContext
) -> list[str]:
    """Run the rules in order; the codes come back de-duplicated in stable order."""
    return unique_in_order(*(rule.evaluate(draft, item_ctx) for rule in rules))


def assemble_result_payload(
    prepared_items: Sequence[PreparedItem],
    attributes_by_id: Mapping[str, Attributes],
    warnings_by_id: Mapping[str, Sequence[str]],
    image_w: int,
    image_h: int,
    meta: RunMeta | None,
    annotated_image_format: Literal["png", "jpeg"] = "png",
) -> ClassificationResult:
    """One ``ClassifiedItem`` per prepared item, in the order given."""
    items = []
    for item in prepared_items:
        item_id = _require_id(item)
        draft = make_draft(item, attributes_by_id.get(item_id, {}))
        warnings = unique_in_order(warnings_by_id.get(item_id, ()))
        items.append(draft.model_copy(update={"warnings": warnings}))
    return ClassificationResult(
        image_width=image_w,
        image_height=image_h,
        items=items,
        annotated_image_format=annotated_image_format,
        meta=meta,
    )


def result_to_json(result: ClassificationResult) -> str:
    """JSON with ``@context``/``@type`` aliases; non-ASCII text stays as is (write as UTF-8)."""
    return result.model_dump_json(by_alias=True, indent=2)


def _require_id(item: PreparedItem) -> str:
    if item.detection.detection_id is None:
        raise ValueError("detection ids must be assigned before assembling the result")
    return item.detection.detection_id
