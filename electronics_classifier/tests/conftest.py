"""Shared test helpers: synthetic images, scripted fakes, and a tiny plug-in component."""

from __future__ import annotations

import cv2
import numpy as np
import pytest
from pydantic import BaseModel, ConfigDict, Field

from electronics_classifier.config import load_config
from electronics_classifier.interfaces import BuildContext
from electronics_classifier.schemas import PreparedItem, RawDetection

IMAGE_KINDS = ("solid", "gradient", "noise", "dark", "blown-out", "blurred", "checkerboard")


def make_image(kind: str = "solid", h: int = 64, w: int = 64, seed: int = 0) -> np.ndarray:
    """Synthetic (h, w, 3) uint8 RGB image."""
    if kind == "solid":
        return np.full((h, w, 3), 128, dtype=np.uint8)
    if kind == "gradient":
        x = np.linspace(0, 255, w, dtype=np.float32)[None, :].repeat(h, axis=0)
        y = np.linspace(0, 255, h, dtype=np.float32)[:, None].repeat(w, axis=1)
        return np.stack([x, y, (x + y) / 2], axis=-1).astype(np.uint8)
    if kind == "noise":
        return np.random.default_rng(seed).integers(0, 256, (h, w, 3), dtype=np.uint8)
    if kind == "dark":
        return np.full((h, w, 3), 8, dtype=np.uint8)
    if kind in ("blown-out", "blown_out"):
        return np.full((h, w, 3), 255, dtype=np.uint8)
    if kind == "checkerboard":
        yy, xx = np.indices((h, w))
        board = (((yy // 8) + (xx // 8)) % 2 * 255).astype(np.uint8)
        return np.stack([board] * 3, axis=-1)
    if kind == "blurred":
        return cv2.GaussianBlur(make_image("checkerboard", h, w), (0, 0), sigmaX=3)
    raise ValueError(f"unknown image kind '{kind}'. Kinds: {', '.join(IMAGE_KINDS)}")


def make_prepared_item(
    box=(10, 10, 50, 50),
    det_id: str | None = "obj_1",
    phrase: str = "router",
    confidence: float = 0.9,
    padded=None,
    context: bool = True,
) -> PreparedItem:
    """A ready-made PreparedItem with a synthetic crop (and context image)."""
    x1, y1, x2, y2 = padded or box
    return PreparedItem(
        detection=RawDetection(
            detection_id=det_id, phrase=phrase, confidence=confidence, bbox_xyxy=box
        ),
        padded_bbox_xyxy=padded or box,
        crop_rgb=make_image("noise", y2 - y1, x2 - x1),
        context_rgb=make_image("gradient", 60, 80) if context else None,
    )


class FakeDetector:
    """Returns exactly the raw detections it was given, on every call."""

    def __init__(self, outputs: list[RawDetection] | None = None) -> None:
        self.outputs = list(outputs or [])
        self.calls: list[np.ndarray] = []

    def predict(self, image_rgb: np.ndarray) -> list[RawDetection]:
        self.calls.append(image_rgb)
        return [d.model_copy(deep=True) for d in self.outputs]


class FakeVLMClient:
    """Returns scripted strings in call order; an entry that is an Exception is raised.

    With ``cycle=True`` the script repeats once exhausted (useful for repeated runs).
    """

    def __init__(self, responses: list[str | Exception] | None = None, cycle: bool = False) -> None:
        self.responses = list(responses or [])
        self.cycle = cycle
        self.calls: list[tuple[str, list[np.ndarray], str]] = []

    def generate(self, system_prompt: str, images: list[np.ndarray], user_text: str) -> str:
        index = len(self.calls)
        self.calls.append((system_prompt, images, user_text))
        if not self.responses:
            raise AssertionError("FakeVLMClient has no scripted responses")
        if index >= len(self.responses):
            if not self.cycle:
                raise AssertionError(
                    f"FakeVLMClient exhausted after {len(self.responses)} responses"
                )
            index %= len(self.responses)
        response = self.responses[index]
        if isinstance(response, Exception):
            raise response
        return response


class TinyDetector:
    """A third-party style plug-in, referenced by dotted path ``conftest:TinyDetector``."""

    class Params(BaseModel):
        model_config = ConfigDict(extra="forbid")
        phrase: str = Field(default="router", description="Phrase reported for the fixed box.")
        score: float = Field(default=0.9, ge=0.0, le=1.0, description="Reported confidence.")

    def __init__(self, params: Params, build: BuildContext) -> None:
        self.params = params
        self.build = build

    def predict(self, image_rgb: np.ndarray) -> list[RawDetection]:
        h, w = image_rgb.shape[:2]
        return [
            RawDetection(
                phrase=self.params.phrase,
                confidence=self.params.score,
                bbox_xyxy=(w // 4, h // 4, w // 2, h // 2),
            )
        ]


@pytest.fixture
def resolved_config() -> dict:
    return load_config()


@pytest.fixture
def build_ctx(resolved_config: dict) -> BuildContext:
    return BuildContext.from_config({**resolved_config, "run": {"device": "cpu", "seed": 0}})


VALID_REPLY = '{"brand": "TP-Link", "model": "Archer", "category": "router", "color": "black"}'


def det(phrase="router", conf=0.9, box=(100, 100, 300, 300)) -> RawDetection:
    return RawDetection(phrase=phrase, confidence=conf, bbox_xyxy=box)


def make_pipeline(detections=None, replies=None, sets=(), presets=(), **components):
    """A pipeline whose detector and VLM are scripted fakes (unless passed explicitly)."""
    from electronics_classifier.pipeline import ElectronicsClassificationPipeline

    components.setdefault("detector", FakeDetector(detections))
    components.setdefault("vlm_client", FakeVLMClient(replies or [VALID_REPLY], cycle=True))
    return ElectronicsClassificationPipeline(presets=presets, sets=sets, **components)
