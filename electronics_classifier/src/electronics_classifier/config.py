"""Config loading: layer, override, validate, resolve, hash.

Layering order (later wins): built-in ``config/pipeline.yaml`` -> presets -> override files
-> ``--set`` values. Dicts are deep-merged and lists are replaced whole. One exception keeps
component swaps clean: when an override sets a component ``name`` that differs from the base
name, the whole ``{name, params}`` block is replaced, so stale params never carry over.
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

FALLBACK_CATEGORY = "electronic device"


class ConfigError(Exception):
    """Invalid configuration. Messages name the full dotted config path."""


# ---------------------------------------------------------------------------
# Structural models (extra="forbid"). Component params are validated at build time.
# ---------------------------------------------------------------------------


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ComponentSpec(_Strict):
    name: str
    params: dict[str, Any] = Field(default_factory=dict)


class RunConfig(_Strict):
    device: Literal["auto", "cuda", "mps", "cpu"] = "auto"
    seed: int = 0


class VocabularyConfig(_Strict):
    categories: list[str]

    @field_validator("categories")
    @classmethod
    def _normalize(cls, value: list[str]) -> list[str]:
        return normalize_vocabulary(value)


class CropConfig(_Strict):
    padding_ratio: float = Field(default=0.12, ge=0.0, lt=1.0)
    include_full_image_context: bool = True
    max_image_side_px: int = Field(default=1024, gt=0)


class VLMConfig(_Strict):
    client: ComponentSpec
    max_retries: int = Field(default=1, ge=0)
    system_prompt_file: str | None = None
    # what to do with an item the VLM says is not an electronic device
    on_not_electronic: Literal["drop", "warn"] = "drop"


class PipelineConfig(_Strict):
    image_preprocess: list[ComponentSpec]
    detector: ComponentSpec
    detection_filters: list[ComponentSpec]
    segmenter: ComponentSpec
    segmentation_gate: ComponentSpec
    crop: CropConfig
    crop_transforms: list[ComponentSpec]
    vlm: VLMConfig
    attribute_postprocess: list[ComponentSpec]
    warning_rules: list[ComponentSpec]


class RenderingConfig(_Strict):
    output_format: Literal["png", "jpeg"] = "png"
    label_template: str
    stroke_scale: float = Field(default=1.0, gt=0.0)
    font_scale: float = Field(default=1.0, gt=0.0)
    mask_alpha: float = Field(default=0.35, ge=0.0, le=1.0)


class Config(_Strict):
    run: RunConfig
    vocabulary: VocabularyConfig
    pipeline: PipelineConfig
    rendering: RenderingConfig


# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------


def normalize_vocabulary(categories: Iterable[str]) -> list[str]:
    """Strip, lowercase, de-duplicate, drop empties, and make sure the fallback is present.

    Dots are replaced by spaces because "." is the Grounding DINO phrase separator.
    """
    result: list[str] = []
    for raw in categories:
        text = " ".join(raw.replace(".", " ").lower().split())
        if text and text not in result:
            result.append(text)
    if FALLBACK_CATEGORY not in result:
        result.append(FALLBACK_CATEGORY)
    return result


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


def format_validation_error(exc: ValidationError, prefix: str = "") -> str:
    """One line per pydantic error, each naming the full dotted config path."""
    lines = []
    for err in exc.errors():
        parts = ([prefix] if prefix else []) + [str(p) for p in err["loc"]]
        lines.append(f"{'.'.join(parts)}: {err['msg']}")
    return "\n".join(lines)


def validate_config(raw: Mapping[str, Any]) -> Config:
    try:
        return Config.model_validate(raw)
    except ValidationError as exc:
        raise ConfigError("invalid config:\n" + format_validation_error(exc)) from None


# ---------------------------------------------------------------------------
# Loading and layering
# ---------------------------------------------------------------------------


def builtin_config_dir() -> Path:
    """``config/`` next to ``src/`` (works for editable installs and source checkouts)."""
    return Path(__file__).resolve().parents[2] / "config"


def load_yaml(path: Path) -> dict[str, Any]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"cannot read config file {path}: {exc.strerror or exc}") from None
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid YAML in {path}: {exc}") from None
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ConfigError(f"config file {path} must contain a mapping at the top level")
    return data


def resolve_preset(name: str) -> Path:
    """A preset is a path to a YAML file, or a name found in ``config/presets/``."""
    candidate = Path(name)
    if candidate.suffix in (".yaml", ".yml") or candidate.parent != Path("."):
        if candidate.is_file():
            return candidate
        raise ConfigError(f"preset file not found: {name}")
    presets_dir = builtin_config_dir() / "presets"
    path = presets_dir / f"{name}.yaml"
    if path.is_file():
        return path
    available = sorted(p.stem for p in presets_dir.glob("*.yaml")) if presets_dir.is_dir() else []
    raise ConfigError(
        f"unknown preset '{name}'. Available presets: {', '.join(available) or 'none'}"
    )


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Deep-merge dicts; replace lists and scalars whole; replace a component block on rename."""
    merged = copy.deepcopy(dict(base))
    for key, value in override.items():
        current = merged.get(key)
        if isinstance(value, Mapping) and isinstance(current, Mapping):
            if _is_component_rename(current, value):
                merged[key] = copy.deepcopy(dict(value))
            else:
                merged[key] = deep_merge(current, value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _is_component_rename(base: Mapping[str, Any], override: Mapping[str, Any]) -> bool:
    return "name" in base and "name" in override and base["name"] != override["name"]


def apply_set(config: dict[str, Any], assignment: str) -> None:
    """Apply ``dotted.key=value`` (value parsed as YAML) to ``config`` in place."""
    key, sep, raw_value = assignment.partition("=")
    key = key.strip()
    if not sep or not key:
        raise ConfigError(f"--set expects dotted.key=value, got '{assignment}'")
    try:
        value = yaml.safe_load(raw_value)
    except yaml.YAMLError as exc:
        raise ConfigError(f"--set {key}: value is not valid YAML: {exc}") from None

    *parents, last = key.split(".")
    node: Any = config
    walked: list[str] = []
    for part in parents:
        walked.append(part)
        if not isinstance(node, dict) or part not in node:
            raise ConfigError(f"--set {key}: unknown config path '{'.'.join(walked)}'")
        node = node[part]
    if not isinstance(node, dict):
        raise ConfigError(
            f"--set {key}: '{'.'.join(walked)}' is not a mapping; "
            "replace list stages with a YAML flow value or an override file"
        )
    if last == "name" and "name" in node and node["name"] != value:
        node["params"] = {}  # renamed component: drop the old component's params
    node[last] = value


def load_config(
    presets: Iterable[str] = (),
    overrides: Iterable[str | Path] = (),
    sets: Iterable[str] = (),
    base_path: str | Path | None = None,
) -> dict[str, Any]:
    """Layer, validate and resolve the config; returns the fully merged plain dict."""
    base = Path(base_path) if base_path else builtin_config_dir() / "pipeline.yaml"
    merged = load_yaml(base)
    for preset in presets:
        merged = deep_merge(merged, load_yaml(resolve_preset(preset)))
    for override in overrides:
        merged = deep_merge(merged, load_yaml(Path(override)))
    for assignment in sets:
        apply_set(merged, assignment)
    return resolve_config(merged)


def resolve_config(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a merged config and return it with structural defaults filled in."""
    return validate_config(raw).model_dump(mode="json")


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def config_hash(resolved: Mapping[str, Any]) -> str:
    canonical = json.dumps(
        resolved, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]


def dump_yaml(resolved: Mapping[str, Any]) -> str:
    return yaml.safe_dump(dict(resolved), sort_keys=False, allow_unicode=True)
