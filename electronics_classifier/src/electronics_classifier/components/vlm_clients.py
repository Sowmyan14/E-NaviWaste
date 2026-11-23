"""VLM clients (kind ``vlm_client``): ``transformers`` (real) and ``fake`` (canned replies)."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from ..interfaces import BuildContext
from ..registry import register


def _default_reply() -> dict[str, Any]:
    return {"brand": "unknown", "model": "unknown", "category": "router", "color": "black"}


@register("vlm_client", "fake")
class FakeVLMClient:
    """Returns the same canned reply for every call. Not a real model."""

    class Params(BaseModel):
        model_config = ConfigDict(extra="forbid")
        response: str | dict[str, Any] = Field(
            default_factory=_default_reply,
            description="Reply text, or a JSON object that is serialised to the reply text.",
        )

    def __init__(self, params: Params, build: BuildContext) -> None:
        self.params = params

    def generate(self, system_prompt: str, images: list[np.ndarray], user_text: str) -> str:
        response = self.params.response
        return response if isinstance(response, str) else json.dumps(response)


@register("vlm_client", "transformers")
class TransformersVLMClient:
    """Any Hugging Face image-text-to-text chat model (default: Qwen2.5-VL-3B-Instruct)."""

    class Params(BaseModel):
        model_config = ConfigDict(extra="forbid")
        model_name: str = Field(
            default="Qwen/Qwen2.5-VL-3B-Instruct", description="Hugging Face model id."
        )
        temperature: float = Field(
            default=0.0, ge=0.0, description="Sampling temperature; 0 means greedy decoding."
        )
        max_new_tokens: int = Field(default=256, gt=0, description="Maximum tokens to generate.")

    def __init__(self, params: Params, build: BuildContext) -> None:
        self.params = params
        self.device = build.device
        self._load_backend()

    def _load_backend(self) -> None:
        try:
            import torch
            from transformers import AutoModelForImageTextToText, AutoProcessor
        except ImportError as exc:
            raise ImportError(
                "the transformers VLM client needs torch and transformers: "
                "pip install electronics-classifier[vlm]"
            ) from exc
        self._inference_mode = torch.inference_mode
        self._processor = AutoProcessor.from_pretrained(self.params.model_name)
        self._model = (
            AutoModelForImageTextToText.from_pretrained(self.params.model_name, torch_dtype="auto")
            .to(self.device)
            .eval()
        )

    def generate(self, system_prompt: str, images: list[np.ndarray], user_text: str) -> str:
        from PIL import Image

        content: list[dict[str, Any]] = [
            {"type": "image", "image": Image.fromarray(image)} for image in images
        ]
        content.append({"type": "text", "text": user_text})
        messages = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user", "content": content},
        ]
        inputs = self._processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.device)

        options: dict[str, Any] = {"max_new_tokens": self.params.max_new_tokens}
        if self.params.temperature > 0:
            options.update(do_sample=True, temperature=self.params.temperature)
        else:
            options.update(do_sample=False)
        with self._inference_mode():
            output = self._model.generate(**inputs, **options)
        new_tokens = output[:, inputs["input_ids"].shape[1] :]
        return self._processor.batch_decode(new_tokens, skip_special_tokens=True)[0]
