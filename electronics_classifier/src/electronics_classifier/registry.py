"""Component registry: ``register`` a class, ``build`` it from a ``{name, params}`` spec."""

from __future__ import annotations

import importlib
from collections.abc import Callable, Mapping
from typing import Any

from pydantic import BaseModel, ValidationError

from .config import ComponentSpec, ConfigError, format_validation_error
from .interfaces import BuildContext

KINDS = (
    "image_preprocess",
    "detector",
    "detection_filter",
    "segmenter",
    "segmentation_gate",
    "crop_transform",
    "vlm_client",
    "attribute_postprocess",
    "warning_rule",
)

_REGISTRY: dict[str, dict[str, type]] = {kind: {} for kind in KINDS}


def register(kind: str, name: str) -> Callable[[type], type]:
    """Class decorator, e.g. ``@register("detector", "grounding_dino")``."""
    if kind not in _REGISTRY:
        raise ValueError(f"unknown component kind '{kind}'. Kinds: {', '.join(KINDS)}")
    if ":" in name:
        raise ValueError(f"registered name '{name}' must not contain ':' (reserved for imports)")

    def decorator(cls: type) -> type:
        existing = _REGISTRY[kind].get(name)
        same_class = (
            existing is not None
            and existing.__module__ == cls.__module__
            and existing.__qualname__ == cls.__qualname__
        )
        if existing is not None and not same_class:
            raise ValueError(f"{kind} '{name}' is already registered to {existing!r}")
        _REGISTRY[kind][name] = cls
        return cls

    return decorator


def load_builtin() -> None:
    """Import every built-in component module so it self-registers (idempotent)."""
    from .components import load_builtin as _load

    _load()


def registered_names(kind: str) -> list[str]:
    load_builtin()
    return sorted(_REGISTRY[kind])


def _import_dotted(name: str, path: str) -> type:
    module_name, _, attr = name.partition(":")
    try:
        obj: Any = importlib.import_module(module_name)
    except ImportError as exc:
        raise ConfigError(f"{path}.name: cannot import module '{module_name}' ({exc})") from None
    for part in attr.split("."):
        try:
            obj = getattr(obj, part)
        except AttributeError:
            raise ConfigError(f"{path}.name: '{module_name}' has no attribute '{attr}'") from None
    return obj


def _resolve_class(kind: str, name: str, path: str) -> type:
    cls = _REGISTRY[kind].get(name)
    if cls is None and ":" in name:
        cls = _import_dotted(name, path)
    if cls is None:
        available = ", ".join(sorted(_REGISTRY[kind])) or "none"
        raise ConfigError(f"{path}.name: unknown {kind} '{name}'. Registered {kind}s: {available}")
    return cls


def _coerce_spec(spec: Any, path: str) -> ComponentSpec:
    if isinstance(spec, ComponentSpec):
        return spec
    try:
        return ComponentSpec.model_validate(spec)
    except ValidationError as exc:
        raise ConfigError(format_validation_error(exc, path)) from None


def build(kind: str, spec: Any, build_ctx: BuildContext, path: str | None = None) -> object:
    """Build one component. ``path`` is its config path, used in error messages."""
    if kind not in _REGISTRY:
        raise ValueError(f"unknown component kind '{kind}'. Kinds: {', '.join(KINDS)}")
    load_builtin()
    path = path or f"pipeline.{kind}"
    component_spec = _coerce_spec(spec, path)
    cls = _resolve_class(kind, component_spec.name, path)
    params_model = getattr(cls, "Params", None)
    if not (isinstance(params_model, type) and issubclass(params_model, BaseModel)):
        raise ConfigError(f"{path}.name: component '{component_spec.name}' defines no Params model")
    try:
        params = params_model.model_validate(component_spec.params)
    except ValidationError as exc:
        raise ConfigError(format_validation_error(exc, f"{path}.params")) from None
    return cls(params, build_ctx)


def describe_components() -> dict[str, dict[str, list[tuple[str, str, str, str]]]]:
    """kind -> name -> [(param, type, default, description)] for ``list-components``."""
    load_builtin()
    out: dict[str, dict[str, list[tuple[str, str, str, str]]]] = {}
    for kind in KINDS:
        out[kind] = {}
        for name in sorted(_REGISTRY[kind]):
            params_model = getattr(_REGISTRY[kind][name], "Params", None)
            rows: list[tuple[str, str, str, str]] = []
            fields: Mapping[str, Any] = getattr(params_model, "model_fields", {})
            for field_name, info in fields.items():
                annotation = getattr(info.annotation, "__name__", str(info.annotation))
                default = (
                    "required"
                    if info.is_required()
                    else repr(info.get_default(call_default_factory=True))
                )
                rows.append((field_name, annotation, default, info.description or ""))
            out[kind][name] = rows
    return out
