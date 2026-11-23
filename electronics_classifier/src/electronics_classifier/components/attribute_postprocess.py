"""Attribute post-processors (kind ``attribute_postprocess``)."""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field

from ..config import FALLBACK_CATEGORY, ConfigError
from ..interfaces import Attributes, BuildContext, ItemContext
from ..registry import register
from ..schemas import WarningCode

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
