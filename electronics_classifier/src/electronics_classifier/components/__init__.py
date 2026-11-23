"""Built-in components. Each module registers its classes on import."""

from __future__ import annotations

import importlib

# Modules are added here as the milestones that implement them land.
_BUILTIN_MODULES: tuple[str, ...] = ()


def load_builtin() -> None:
    for module in _BUILTIN_MODULES:
        importlib.import_module(f"{__name__}.{module}")
