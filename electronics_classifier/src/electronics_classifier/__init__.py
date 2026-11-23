"""Image-only household electronics classifier."""

from __future__ import annotations

import warnings
from typing import Any

# supervision emits a deprecation notice for its own internal decorators on every import;
# it is not actionable here. Only that message is hidden; other warnings are untouched.
warnings.filterwarnings("ignore", message=".*template_mgs.*", category=FutureWarning)

from .schemas import ClassificationResult  # noqa: E402

__all__ = ["ClassificationResult", "ElectronicsClassificationPipeline"]


def __getattr__(name: str) -> Any:
    # Lazy so that `import electronics_classifier` stays light (no models, no heavy imports).
    if name == "ElectronicsClassificationPipeline":
        from .pipeline import ElectronicsClassificationPipeline

        return ElectronicsClassificationPipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
