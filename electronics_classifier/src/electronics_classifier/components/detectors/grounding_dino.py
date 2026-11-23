"""Grounding DINO zero-shot detector (default backend, Apache-2.0).

Returns raw, unfiltered detections above a low floor score; all real thresholding lives in
``detection_filters``. The model stack is imported lazily in the constructor.
"""

from __future__ import annotations

import inspect
import logging
from collections.abc import Callable, Sequence

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from ...interfaces import BuildContext
from ...registry import register
from ...schemas import RawDetection

log = logging.getLogger(__name__)


def build_prompt(phrases: Sequence[str]) -> str:
    """``"router . modem . power strip ."``: phrases joined by the DINO separator."""
    return " . ".join(phrases) + " ."


def chunk_vocabulary(
    phrases: Sequence[str], max_tokens: int, count_tokens: Callable[[str], int]
) -> list[list[str]]:
    """Greedily pack whole phrases into prompts of at most ``max_tokens`` tokens.

    A phrase is never split or truncated. A phrase that alone exceeds the limit gets its own
    prompt (with a warning) rather than being dropped.
    """
    chunks: list[list[str]] = []
    current: list[str] = []
    for phrase in phrases:
        trial = [*current, phrase]
        if current and count_tokens(build_prompt(trial)) > max_tokens:
            chunks.append(current)
            current = [phrase]
        else:
            current = trial
        if len(current) == 1 and count_tokens(build_prompt(current)) > max_tokens:
            log.warning("phrase '%s' alone exceeds max_prompt_tokens=%d", phrase, max_tokens)
    if current:
        chunks.append(current)
    return chunks


def map_label(label: str, phrases: Sequence[str]) -> str | None:
    """Longest vocabulary phrase contained in the returned label text, else ``None``."""
    text = f" {' '.join(str(label).lower().replace('.', ' ').split())} "
    best: str | None = None
    for phrase in phrases:
        if f" {phrase} " in text and (best is None or len(phrase) > len(best)):
            best = phrase
    return best


@register("detector", "grounding_dino")
class GroundingDinoDetector:
    """Open-vocabulary detector driven by the shared vocabulary."""

    class Params(BaseModel):
        model_config = ConfigDict(extra="forbid")
        model_id: str = Field(
            default="IDEA-Research/grounding-dino-tiny", description="Hugging Face model id."
        )
        raw_box_threshold: float = Field(
            default=0.15, ge=0.0, le=1.0, description="Low floor box score; filters do the rest."
        )
        raw_text_threshold: float = Field(
            default=0.15, ge=0.0, le=1.0, description="Low floor text score for label decoding."
        )
        max_prompt_tokens: int = Field(
            default=256, gt=0, description="Vocabulary is split into several prompts above this."
        )

    def __init__(self, params: Params, build: BuildContext) -> None:
        self.params = params
        self.vocabulary = tuple(build.categories)
        self.device = build.device
        self._load_backend()
        self.chunks = chunk_vocabulary(
            self.vocabulary, params.max_prompt_tokens, self._count_tokens
        )

    def _load_backend(self) -> None:
        try:
            import torch
            from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        except ImportError as exc:
            raise ImportError(
                "the grounding_dino detector needs torch and transformers: "
                "pip install electronics-classifier[dino]"
            ) from exc
        self._torch = torch
        self._processor = AutoProcessor.from_pretrained(self.params.model_id)
        self._model = (
            AutoModelForZeroShotObjectDetection.from_pretrained(self.params.model_id)
            .to(self.device)
            .eval()
        )
        tokenizer = self._processor.tokenizer
        self._count_tokens = lambda text: len(tokenizer(text)["input_ids"])

    def _run(self, image_rgb: np.ndarray, prompt: str) -> list[tuple[str, float, Sequence[float]]]:
        """One model call: returns ``(label_text, score, xyxy)`` in ``image_rgb`` pixels."""
        from PIL import Image

        torch = self._torch
        inputs = self._processor(
            images=Image.fromarray(image_rgb), text=prompt, return_tensors="pt"
        ).to(self.device)
        with torch.no_grad():
            outputs = self._model(**inputs)
        post = self._processor.post_process_grounded_object_detection
        threshold_arg = (
            "threshold" if "threshold" in inspect.signature(post).parameters else "box_threshold"
        )
        result = post(
            outputs,
            inputs.input_ids,
            **{threshold_arg: self.params.raw_box_threshold},
            text_threshold=self.params.raw_text_threshold,
            target_sizes=[image_rgb.shape[:2]],
        )[0]
        labels = result.get("text_labels") or result["labels"]
        return [
            (str(label), float(score), [float(v) for v in box])
            for label, score, box in zip(labels, result["scores"], result["boxes"], strict=False)
        ]

    def predict(self, image_rgb: np.ndarray) -> list[RawDetection]:
        found: list[RawDetection] = []
        for chunk in self.chunks:
            for label, score, box in self._run(image_rgb, build_prompt(chunk)):
                phrase = map_label(label, chunk)
                if phrase is None:
                    log.warning("dropping detection with unmapped label '%s'", label)
                    continue
                found.append(
                    RawDetection(
                        phrase=phrase,
                        confidence=min(max(score, 0.0), 1.0),
                        bbox_xyxy=tuple(round(v) for v in box),
                    )
                )
        return found
