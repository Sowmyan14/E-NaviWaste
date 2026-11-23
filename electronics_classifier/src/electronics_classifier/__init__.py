"""Image-only household electronics classifier."""

from __future__ import annotations

from typing import Any

from .schemas import ClassificationResult

__all__ = ["ClassificationResult", "ElectronicsClassificationPipeline"]


def __getattr__(name: str) -> Any:
    # Lazy so that `import electronics_classifier` stays light (no models, no heavy imports).
    if name == "ElectronicsClassificationPipeline":
        from .pipeline import ElectronicsClassificationPipeline

        return ElectronicsClassificationPipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
