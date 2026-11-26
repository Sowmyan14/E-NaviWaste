"""The VLM system prompt. The category list always comes from the shared vocabulary."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from .config import FALLBACK_CATEGORY, ConfigError, builtin_config_dir

CATEGORIES_PLACEHOLDER = "{categories}"
EXAMPLE_PLACEHOLDER = "{example_category}"  # optional: replaced by the first category

DEFAULT_TEMPLATE = """\
You identify one household electronic device from the images you are given.

Return ONLY a JSON object with exactly these keys: "brand", "model", "category", "color".

Rules:
- "category" must be exactly one of these allowed labels: {categories}.
- Read "brand" and "model" only from text, logos or unmistakable design cues that are \
clearly visible on the device itself.
- Never guess a brand, and never fabricate or infer a model number.
- Write a "model" only if its exact name or number is printed on the device in the images. \
Recognizing the brand, a logo, or the product family is not enough: if no model text is \
visible, "model" must be "unknown". For example, a logo alone means a known brand but \
model "unknown".
- "color" is the main colour of the device body.
- If a value cannot be read from what is visible, use "unknown".
- Output the JSON object only: no markdown, no explanations.

Example output:
{"brand": "unknown", "model": "unknown", "category": "{example_category}", "color": "black"}
"""


def validate_template(template: str, source: str = "system prompt template") -> str:
    """A custom template must contain the ``{categories}`` placeholder."""
    if CATEGORIES_PLACEHOLDER not in template:
        raise ConfigError(f"{source} must contain the {CATEGORIES_PLACEHOLDER} placeholder")
    return template


def load_template(path: str | Path) -> str:
    """Read and validate a template file (``ConfigError`` on any problem).

    A relative path that does not exist from the current folder is also tried against the
    project folder next to ``config/``, so the built-in prompts load from any working directory.
    """
    candidate = Path(path)
    if not candidate.is_absolute() and not candidate.is_file():
        bundled = builtin_config_dir().parent / candidate
        if bundled.is_file():
            candidate = bundled
    try:
        text = candidate.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"cannot read system prompt template {path}: {exc.strerror}") from None
    return validate_template(text, f"system prompt template {path}")


def build_vlm_system_prompt(categories: Sequence[str], template: str | None = None) -> str:
    """Fill the template. Plain substitution, so braces elsewhere in a template are safe."""
    validate_template(DEFAULT_TEMPLATE if template is None else template)
    text = DEFAULT_TEMPLATE if template is None else template
    labels = ", ".join(f'"{category}"' for category in categories)
    example = categories[0] if categories else FALLBACK_CATEGORY
    return text.replace(CATEGORIES_PLACEHOLDER, labels).replace(EXAMPLE_PLACEHOLDER, example)
