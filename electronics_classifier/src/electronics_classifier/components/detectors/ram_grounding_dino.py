"""RAM++ tagger + Grounding DINO box grounder (the front half of Grounded-SAM + RAM).

RAM++ (Recognize Anything) lists what is in the photo, those tags become the Grounding DINO
prompt, and DINO returns one box per tag it can find. This is the flow of the Grounded-SAM
``automatic_label_ram_demo.py``. Masks are made later by a segmenter (``hf_sam``).

Phrases come from the image, not from the vocabulary, so this detector declares
``open_vocabulary`` and the pipeline's sanitize step does not drop its phrases. The vocabulary
can still be added to the prompt (``include_vocabulary``) so known devices are never missed.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterable

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from ...config import ConfigError
from ...interfaces import BuildContext
from ...registry import register
from ...schemas import RawDetection
from .grounding_dino import GroundingDinoDetector, build_prompt, chunk_vocabulary, map_label

log = logging.getLogger(__name__)

TAG_SEPARATOR = "|"
MAX_TAG_WORDS = 4  # longer RAM tags are scene descriptions, not object names

# RAM tags the whole scene. These are never a device, so prompting DINO with them only makes
# boxes for the table, the wall or the colour. The VLM verdict catches whatever is left.
DEFAULT_DROP_TAGS = (
    "black", "white", "gray", "grey", "green", "blue", "red", "yellow", "orange", "brown",
    "pink", "purple", "silver", "gold", "beige", "colorful",
    "room", "living room", "bedroom", "bathroom", "office", "house", "home", "indoor",
    "outdoor", "interior", "apartment", "building", "scene",
    "floor", "wall", "ceiling", "table", "desk", "shelf", "chair", "couch", "sofa", "bed",
    "cabinet", "counter", "surface", "wood", "wooden", "wood floor", "carpet", "rug",
    "sit", "stand", "lay", "hold", "place", "use", "face", "close-up", "top",
)  # fmt: skip


def clean_tags(
    raw: str, drop: Iterable[str] = (), max_tags: int | None = None
) -> list[str]:
    """RAM tag string (``"a | b | c"``) to clean phrases for the DINO prompt.

    Lower-cased, de-duplicated (first seen wins) and without dots or commas, because DINO
    separates phrases with ``.`` and the RAM demo with ``,``. Stoplisted and over-long tags
    are dropped. Anything that is not a string gives an empty list.
    """
    if not isinstance(raw, str):
        return []
    stop = {" ".join(str(t).lower().split()) for t in drop}
    tags: list[str] = []
    for piece in raw.split(TAG_SEPARATOR):
        tag = " ".join(piece.lower().replace(".", " ").replace(",", " ").split())
        if not tag or tag in stop or tag in tags or len(tag.split()) > MAX_TAG_WORDS:
            continue
        tags.append(tag)
    return tags[:max_tags] if max_tags is not None else tags


@register("detector", "ram_grounding_dino")
class RamGroundingDinoDetector(GroundingDinoDetector):
    """RAM++ proposes the phrases, Grounding DINO draws the boxes."""

    open_vocabulary = True  # the pipeline keeps phrases that are not in the vocabulary

    class Params(BaseModel):
        model_config = ConfigDict(extra="forbid")
        ram_checkpoint: str = Field(
            description="Path to the RAM++ checkpoint (ram_plus_swin_large_14m.pth)."
        )
        ram_image_size: int = Field(default=384, description="RAM input size: 224 or 384.")
        ram_on_cpu: bool = Field(
            default=False, description="Run only RAM on the CPU (a fallback if MPS fails)."
        )
        ram_half: bool = Field(
            default=False, description="Run RAM in float16: about half the memory."
        )
        mps_fallback: bool = Field(
            default=True,
            description="On Apple GPUs let unsupported ops fall back to the CPU "
            "(sets PYTORCH_ENABLE_MPS_FALLBACK).",
        )
        model_id: str = Field(
            default="IDEA-Research/grounding-dino-tiny", description="Hugging Face DINO model id."
        )
        raw_box_threshold: float = Field(
            default=0.25, ge=0.0, le=1.0, description="Floor box score (Grounded-SAM demo value)."
        )
        raw_text_threshold: float = Field(
            default=0.2, ge=0.0, le=1.0, description="Floor text score (Grounded-SAM demo value)."
        )
        max_prompt_tokens: int = Field(
            default=256, gt=0, description="The tag list is split into several prompts above this."
        )
        max_tags: int = Field(default=40, gt=0, description="Keep at most this many RAM tags.")
        drop_tags: list[str] = Field(
            default_factory=lambda: list(DEFAULT_DROP_TAGS),
            description="RAM tags that are never devices (colours, rooms, furniture, verbs).",
        )
        include_vocabulary: bool = Field(
            default=True,
            description="Add vocabulary.categories to the tags so known devices are never missed.",
        )

    def __init__(self, params: Params, build: BuildContext) -> None:
        if not os.path.isfile(os.path.expanduser(params.ram_checkpoint)):
            raise ConfigError(f"params.ram_checkpoint: file not found: {params.ram_checkpoint}")
        self.params = params
        self.vocabulary = tuple(build.categories)
        self.device = build.device
        if build.device == "mps" and params.mps_fallback:
            os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        self._load_ram()
        self._load_backend()  # Grounding DINO, inherited

    # --- RAM ------------------------------------------------------------------------

    def _load_ram(self) -> None:
        try:
            import torch
            from ram import get_transform, inference_ram
            from ram.models import ram_plus
        except ImportError as exc:
            raise ImportError(
                "the ram_grounding_dino detector needs recognize-anything, torch and "
                "transformers: pip install electronics-classifier[grounded]"
            ) from exc
        self._ram_device = "cpu" if self.params.ram_on_cpu else self.device
        model = ram_plus(
            pretrained=os.path.expanduser(self.params.ram_checkpoint),
            image_size=self.params.ram_image_size,
            vit="swin_l",
        ).eval()
        if self.params.ram_half and self._ram_device != "cpu":
            model = model.half()
        self._ram_model = model.to(self._ram_device)
        self._ram_dtype = torch.float16 if self.params.ram_half else torch.float32
        self._ram_transform = get_transform(image_size=self.params.ram_image_size)
        self._inference_ram = inference_ram

    def _tag_image(self, image_rgb: np.ndarray) -> str:
        """One RAM call: the raw English tag string (tags joined by ``" | "``)."""
        from PIL import Image

        tensor = self._ram_transform(Image.fromarray(image_rgb)).unsqueeze(0)
        tensor = tensor.to(self._ram_device).to(self._ram_dtype)
        return self._inference_ram(tensor, self._ram_model)[0]

    # --- detect ---------------------------------------------------------------------

    def prompt_phrases(self, image_rgb: np.ndarray) -> list[str]:
        """The phrases for this image: cleaned RAM tags, then the vocabulary (if enabled)."""
        try:
            raw = self._tag_image(image_rgb)
        except Exception as exc:  # RAM trouble must not abort the run: use the vocabulary
            log.warning("RAM tagging failed, using the vocabulary only: %s", exc)
            raw = ""
        phrases = clean_tags(raw, self.params.drop_tags, self.params.max_tags)
        if self.params.include_vocabulary:
            phrases += [v for v in self.vocabulary if v not in phrases]
        if not phrases:
            log.warning("no prompt phrases for this image (RAM gave no usable tags)")
        return phrases

    def predict(self, image_rgb: np.ndarray) -> list[RawDetection]:
        phrases = self.prompt_phrases(image_rgb)
        if not phrases:
            return []
        found: list[RawDetection] = []
        for chunk in chunk_vocabulary(phrases, self.params.max_prompt_tokens, self._count_tokens):
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

