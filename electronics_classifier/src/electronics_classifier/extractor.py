"""Attribute extraction (fixed logic): prompt, VLM call, JSON parsing, retries, fallback.

One item's VLM trouble never raises: the item gets all-``unknown`` attributes (category
``electronic device``) and a ``VLM_PARSE_FAILED`` warning.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Annotated, Any

from pydantic import BaseModel, BeforeValidator, ConfigDict, ValidationError

from .config import FALLBACK_CATEGORY
from .interfaces import AttributePostprocessor, Attributes, ItemContext, VLMClient
from .schemas import PreparedItem, WarningCode
from .system_prompt import build_vlm_system_prompt

log = logging.getLogger(__name__)

USER_TEXT = (
    "Image 1 is the device crop. Image 2 (if present) is the full scene for context only; "
    "describe only the device in Image 1."
)
PHRASE_HINT = (
    " The detector proposed the label '{phrase}'; treat it as a hint to check, not as a fact."
)
RETRY_REMINDER = (
    "Return ONLY valid JSON: a single object with the keys brand, model, category, color."
)

_FENCE_OPEN = re.compile(r"^```[A-Za-z]*\s*")


def _scalar_to_text(value: Any) -> str:
    """JSON null -> "unknown"; str/int/float/bool -> str; objects and lists are wrong-shaped."""
    if value is None:
        return "unknown"
    if isinstance(value, str | int | float | bool):
        return str(value)
    raise ValueError("expected a plain value, not an object or list")


Scalar = Annotated[str, BeforeValidator(_scalar_to_text)]

_TRUE_WORDS = frozenset({"true", "yes", "y", "1"})
_FALSE_WORDS = frozenset({"false", "no", "n", "0"})


def _flag_or_none(value: Any) -> bool | None:
    """A yes/no answer as a bool. Missing, null or unreadable answers are ``None`` (unknown)."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        word = value.strip().lower()
        if word in _TRUE_WORDS:
            return True
        if word in _FALSE_WORDS:
            return False
    return None


Flag = Annotated[bool | None, BeforeValidator(_flag_or_none)]


class VLMReply(BaseModel):
    """The four attributes plus the optional ``is_electronic`` verdict.

    Extra keys are ignored; missing keys become ``unknown``; a missing verdict is ``None``.
    """

    model_config = ConfigDict(extra="ignore")
    brand: Scalar = "unknown"
    model: Scalar = "unknown"
    category: Scalar = "unknown"
    color: Scalar = "unknown"
    is_electronic: Flag = None


def find_json_value(text: str) -> Any:
    """The first JSON object in ``text`` (prose and code fences around it are skipped).

    A reply that *is* a JSON array is returned as that list, so it is rejected as wrong-shaped
    rather than having an element picked out of it. ``None`` when nothing decodes.
    """
    decoder = json.JSONDecoder()
    stripped = _FENCE_OPEN.sub("", text.strip())
    if stripped.startswith("["):
        try:
            return decoder.raw_decode(stripped)[0]
        except json.JSONDecodeError:
            return None
    start = text.find("{")
    while start != -1:
        try:
            return decoder.raw_decode(text, start)[0]
        except json.JSONDecodeError:
            start = text.find("{", start + 1)
    return None


def parse_reply(text: object) -> Attributes | None:
    """Attributes from a raw VLM reply, or ``None`` if it is unusable."""
    if not isinstance(text, str):
        return None
    value = find_json_value(text)
    if not isinstance(value, dict):
        return None
    try:
        attributes = VLMReply.model_validate(value).model_dump()
    except ValidationError:
        return None
    if attributes["is_electronic"] is None:
        del attributes["is_electronic"]  # no verdict given: keep the four-key shape
    return attributes


def fallback_attributes() -> Attributes:
    return {
        "brand": "unknown",
        "model": "unknown",
        "category": FALLBACK_CATEGORY,
        "color": "unknown",
    }


@dataclass
class ExtractionResult:
    attributes: Attributes
    warnings: list[str] = field(default_factory=list)
    raw_replies: list[str] = field(default_factory=list)  # every reply received, for the trace


class AttributeExtractor:
    """Builds the prompt once, then extracts attributes item by item."""

    def __init__(
        self,
        client: VLMClient,
        categories: Sequence[str],
        template: str | None = None,
        max_retries: int = 1,
    ) -> None:
        self.client = client
        self.max_retries = max_retries
        self.system_prompt = build_vlm_system_prompt(categories, template)

    def extract(self, item: PreparedItem) -> ExtractionResult:
        base_text = USER_TEXT + PHRASE_HINT.format(phrase=item.detection.phrase)
        images = [item.crop_rgb]  # the crop first, then the full-scene context
        if item.context_rgb is not None:
            images.append(item.context_rgb)
        raw_replies: list[str] = []
        for attempt in range(self.max_retries + 1):
            user_text = base_text if attempt == 0 else f"{base_text}\n{RETRY_REMINDER}"
            try:
                reply = self.client.generate(self.system_prompt, images, user_text)
            except Exception as exc:  # timeouts, OOM...: usually repeatable, so no retry
                log.warning("VLM call failed for %s: %s", item.detection.detection_id, exc)
                break
            raw_replies.append(reply if isinstance(reply, str) else repr(reply))
            attributes = parse_reply(reply)
            if attributes is not None:
                return ExtractionResult(attributes, [], raw_replies)
            log.warning("unparseable VLM reply for %s", item.detection.detection_id)
        return ExtractionResult(
            fallback_attributes(), [WarningCode.VLM_PARSE_FAILED.value], raw_replies
        )


def apply_postprocessors(
    processors: Sequence[AttributePostprocessor], attributes: Attributes, item_ctx: ItemContext
) -> tuple[Attributes, list[str], list[Attributes]]:
    """Run the chain in order.

    Returns the final attributes, the warnings the steps emitted, and a snapshot of the
    attributes after each step (for the pipeline trace). The input is never mutated.
    """
    current = dict(attributes)
    warnings: list[str] = []
    snapshots: list[Attributes] = []
    for processor in processors:
        current, extra = processor.apply(dict(current), item_ctx)
        warnings.extend(extra)
        snapshots.append(dict(current))
    return current, warnings, snapshots
