"""Attribute post-processors (kind ``attribute_postprocess``)."""

from __future__ import annotations

import logging
import re
import unicodedata
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

from ..config import FALLBACK_CATEGORY, ConfigError
from ..interfaces import Attributes, BuildContext, ItemContext
from ..registry import register
from ..schemas import WarningCode

log = logging.getLogger(__name__)

UNKNOWN = "unknown"
UNKNOWN_VALUES = frozenset({"", "unknown", "n/a", "na", "none", "null", "-", "?", "unspecified"})
FREE_TEXT_FIELDS = ("brand", "model", "color")


class _NoParams(BaseModel):
    model_config = ConfigDict(extra="forbid")


def clean_text(value: Any, max_len: int) -> str:
    """Cast to text, drop control characters, collapse whitespace, map "no value" to unknown."""
    text = str(value)
    text = "".join(
        (" " if ch.isspace() else "") if unicodedata.category(ch) == "Cc" else ch for ch in text
    )
    text = " ".join(text.split())
    if text.casefold() in UNKNOWN_VALUES:
        return UNKNOWN
    return text[:max_len].rstrip()


@register("attribute_postprocess", "normalize_unknown")
class NormalizeUnknown:
    """Clean brand/model/color: one line, at most ``max_len`` characters, "unknown" if empty."""

    class Params(BaseModel):
        model_config = ConfigDict(extra="forbid")
        max_len: int = Field(default=64, gt=0, description="Longest allowed value, characters.")

    def __init__(self, params: Params, build: BuildContext) -> None:
        self.params = params

    def apply(self, attrs: Attributes, item_ctx: ItemContext) -> tuple[Attributes, list[str]]:
        out = dict(attrs)
        for name in FREE_TEXT_FIELDS:
            out[name] = clean_text(attrs.get(name, UNKNOWN), self.params.max_len)
        return out, []


@register("attribute_postprocess", "category_vocab")
class CategoryVocab:
    """Keep only categories from the vocabulary; anything else becomes "electronic device"."""

    Params = _NoParams

    def __init__(self, params: _NoParams, build: BuildContext) -> None:
        self.params = params
        self.categories = frozenset(build.categories)

    def apply(self, attrs: Attributes, item_ctx: ItemContext) -> tuple[Attributes, list[str]]:
        out = dict(attrs)
        category = " ".join(str(attrs.get("category", UNKNOWN)).lower().split())
        if category in self.categories:
            out["category"] = category
            return out, []
        out["category"] = FALLBACK_CATEGORY
        return out, [WarningCode.CATEGORY_OUT_OF_VOCAB.value]


def alias_key(text: str) -> str:
    """Case-insensitive key that ignores hyphens, underscores and extra spaces."""
    return " ".join(re.sub(r"[-_]", " ", text).lower().split())


@register("attribute_postprocess", "brand_aliases")
class BrandAliases:
    """Map alias spellings of a brand to its canonical name, from a YAML or JSON file."""

    class Params(BaseModel):
        model_config = ConfigDict(extra="forbid")
        mapping_file: str = Field(
            description="YAML/JSON file of `alias: Canonical Name` pairs, e.g. `tp link: TP-Link`."
        )

    def __init__(self, params: Params, build: BuildContext) -> None:
        self.params = params
        path = Path(params.mapping_file).expanduser()
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8"))
        except OSError as exc:
            raise ConfigError(
                f"params.mapping_file: cannot read {path}: {exc.strerror or exc}"
            ) from None
        except yaml.YAMLError as exc:
            raise ConfigError(f"params.mapping_file: {path} is not valid YAML: {exc}") from None
        if not isinstance(data, dict) or not all(
            isinstance(k, str) and isinstance(v, str) for k, v in data.items()
        ):
            raise ConfigError(
                f"params.mapping_file: {path} must be a mapping of alias to canonical name"
            )
        self._aliases = {alias_key(alias): canonical for alias, canonical in data.items()}

    def apply(self, attrs: Attributes, item_ctx: ItemContext) -> tuple[Attributes, list[str]]:
        out = dict(attrs)
        brand = str(attrs.get("brand", UNKNOWN))
        out["brand"] = self._aliases.get(alias_key(brand), brand)
        return out, []


# --- OCR cross-check of the model name ------------------------------------------------------


def normalize_for_match(text: str) -> str:
    """Upper-case letters and digits only, so ``xr-5000`` and ``XR 5000`` compare equal."""
    return "".join(ch for ch in text.upper() if ch.isalnum())


def model_is_visible(model: str, ocr_text: str, min_match: float) -> bool:
    """True if the claimed ``model`` really appears in the text OCR read from the crop.

    The claim is always the pattern being searched for: OCR that saw only a fragment of the
    claim (``XR5`` for ``XR5000``) does not verify it. Short claims must match exactly, since
    a single wrong character changes a short name; longer ones tolerate OCR slips.
    """
    from rapidfuzz import fuzz

    claimed, seen = normalize_for_match(model), normalize_for_match(ocr_text)
    if not claimed or not seen:
        return False
    required = 100.0 if len(claimed) < 5 else min_match
    if len(seen) >= len(claimed):
        score = fuzz.partial_ratio(claimed, seen)
    else:  # OCR text shorter than the claim: compare the whole strings, not a fragment
        score = fuzz.ratio(claimed, seen)
    return score >= required


class RapidOcrReader:
    """Reads the text in an RGB image with RapidOCR (its bundled models, no download)."""

    def __init__(self, min_confidence: float, engine: Any = None) -> None:
        if engine is None:
            import rapidfuzz  # noqa: F401  (fail here, not later, if the extra is incomplete)
            from rapidocr import RapidOCR

            engine = RapidOCR(params={"Global.log_level": "critical"})
        self._engine = engine
        self._min_confidence = min_confidence

    def read(self, image_rgb: Any) -> str:
        import cv2

        result = self._engine(cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))  # OpenCV order
        texts, scores = result.txts or (), result.scores or ()
        lines = zip(texts, scores, strict=False)
        return " ".join(text for text, score in lines if score >= self._min_confidence)


@register("attribute_postprocess", "ocr_verify_model")
class OcrVerifyModel:
    """Accept the VLM's model name only if OCR can find it in the crop; else ``unknown``.

    OCR runs only when it is needed: an item whose model is already ``unknown`` is skipped
    (and the OCR engine is not even loaded). The engine loads on first use.
    """

    class Params(BaseModel):
        model_config = ConfigDict(extra="forbid")
        min_confidence: float = Field(
            default=0.5, ge=0.0, le=1.0, description="Ignore OCR text lines scored below this."
        )
        min_match: float = Field(
            default=85.0,
            ge=0.0,
            le=100.0,
            description="Similarity (0-100) at which the claimed model counts as seen, for names "
            "of 5+ characters. Shorter names must match exactly.",
        )
        if_unavailable: Literal["unverified", "error"] = Field(
            default="unverified",
            description="If RapidOCR is not installed: 'unverified' treats every claimed model as "
            "unverified (with a one-time warning); 'error' fails when the pipeline is built.",
        )

    HINT = "pip install electronics-classifier[ocr]"

    def __init__(self, params: Params, build: BuildContext) -> None:
        self.params = params
        self._reader: Any = None
        self._unavailable = False
        if params.if_unavailable == "error":
            self._get_reader()  # fail fast, at build time

    def _make_reader(self) -> Any:
        return RapidOcrReader(self.params.min_confidence)

    def _get_reader(self) -> Any:
        if self._reader is None and not self._unavailable:
            try:
                self._reader = self._make_reader()
            except ImportError as exc:
                if self.params.if_unavailable == "error":
                    raise ImportError(f"ocr_verify_model needs RapidOCR: {self.HINT}") from exc
                self._unavailable = True
                log.warning(
                    "RapidOCR is not installed, so model names cannot be verified and are "
                    "reported as MODEL_UNVERIFIED (%s)",
                    self.HINT,
                )
        return self._reader

    def apply(self, attrs: Attributes, item_ctx: ItemContext) -> tuple[Attributes, list[str]]:
        out = dict(attrs)
        model = str(attrs.get("model", UNKNOWN))
        if clean_text(model, len(model)) == UNKNOWN:
            return out, []  # no claim, so no OCR needed
        reader = self._get_reader()
        if reader is not None:
            text = reader.read(item_ctx.item.crop_rgb)
            if model_is_visible(model, text, self.params.min_match):
                return out, []
        out["model"] = UNKNOWN
        return out, [WarningCode.MODEL_UNVERIFIED.value]
