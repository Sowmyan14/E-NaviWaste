"""VLM clients (kind ``vlm_client``).

``transformers`` runs a model in this process, ``openai_compatible`` calls a model server
(LM Studio, Ollama local or cloud, vLLM, ...), and ``fake`` returns canned replies.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import os
from typing import Any
from urllib.parse import urlparse

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator

from ..config import ConfigError
from ..interfaces import BuildContext
from ..registry import register

log = logging.getLogger(__name__)
LOCAL_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})
CHECK_TIMEOUT_S = 10.0  # the build-time server check never waits longer than this


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


def _data_url(image: np.ndarray) -> str:
    """A lossless PNG data URL: small label text must survive the trip to the server."""
    from PIL import Image

    buffer = io.BytesIO()
    Image.fromarray(image).save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


@register("vlm_client", "openai_compatible")
class OpenAICompatibleVLMClient:
    """Any server that speaks the OpenAI chat API: LM Studio, Ollama (local or cloud), vLLM.

    Crops and context images are sent to ``base_url``. When that host is not this machine a
    warning says so, because the photos then leave your computer. The API key is never part
    of the config: ``api_key_env`` names the environment variable that holds it.
    """

    class Params(BaseModel):
        model_config = ConfigDict(extra="forbid")
        base_url: str = Field(
            default="http://localhost:1234/v1",
            description="Server URL ending in /v1. LM Studio: http://localhost:1234/v1, "
            "local Ollama: http://localhost:11434/v1, Ollama cloud: https://ollama.com/v1.",
        )
        model: str = Field(description="Model name exactly as the server lists it.")
        api_key_env: str | None = Field(
            default=None,
            description="NAME of the environment variable holding the API key (never the key "
            "itself). Leave empty for local servers, which ignore the key.",
        )
        temperature: float = Field(
            default=0.0,
            ge=0.0,
            description="Sampling temperature; 0 means as deterministic as the server allows.",
        )
        max_tokens: int = Field(default=256, gt=0, description="Maximum tokens to generate.")
        timeout_s: float = Field(default=120.0, gt=0, description="Per-request timeout, seconds.")
        max_retries: int = Field(
            default=2,
            ge=0,
            description="Automatic retries (with backoff) for connection errors, 429 and 5xx.",
        )
        check_server: bool = Field(
            default=True,
            description="At build time, check the server answers and lists the model. Turn off "
            "for servers that do not provide a model list.",
        )

        @field_validator("base_url")
        @classmethod
        def _http_url(cls, value: str) -> str:
            parsed = urlparse(value)
            if parsed.scheme not in ("http", "https") or not parsed.netloc:
                raise ValueError("must be an http(s) URL such as http://localhost:1234/v1")
            return value.rstrip("/")

    def __init__(self, params: Params, build: BuildContext) -> None:
        self.params = params
        api_key = self._read_api_key()
        host = urlparse(params.base_url).hostname
        if host not in LOCAL_HOSTS:
            log.warning("openai_compatible sends crops and context images to %s", host)
        self._client = self._make_client(api_key)
        if params.check_server:
            self._check_server()

    def _read_api_key(self) -> str:
        name = self.params.api_key_env
        if name is None:
            return "not-needed"  # the SDK wants a non-empty key; local servers ignore it
        key = os.environ.get(name)
        if not key:
            raise ConfigError(
                f"params.api_key_env: environment variable {name} is not set or empty"
            )
        return key

    def _make_client(self, api_key: str) -> Any:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError(
                "the openai_compatible VLM client needs the openai package: "
                "pip install electronics-classifier[server]"
            ) from exc
        return OpenAI(
            base_url=self.params.base_url,
            api_key=api_key,
            timeout=self.params.timeout_s,
            max_retries=self.params.max_retries,
        )

    def _check_server(self) -> None:
        url, model = self.params.base_url, self.params.model
        try:
            probe = self._client.with_options(timeout=CHECK_TIMEOUT_S, max_retries=0)
            listing = probe.models.list()
        except Exception as exc:  # connection refused, DNS failure, 401, ...
            raise ConfigError(
                f"params.base_url: cannot reach a model server at {url} ({type(exc).__name__}). "
                "Is it running, and is the key valid? "
                "Set params.check_server: false to skip this check."
            ) from None
        available = [entry.id for entry in listing.data]
        if available and model not in available:
            shown = ", ".join(available[:10]) + (" ..." if len(available) > 10 else "")
            raise ConfigError(
                f"params.model: '{model}' is not available on {url}. Available: {shown}. "
                "Set params.check_server: false if this server does not list its models."
            )

    def generate(self, system_prompt: str, images: list[np.ndarray], user_text: str) -> str:
        content: list[dict[str, Any]] = [
            {"type": "image_url", "image_url": {"url": _data_url(image)}} for image in images
        ]
        content.append({"type": "text", "text": user_text})
        response = self._client.chat.completions.create(
            model=self.params.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content},
            ],
            temperature=self.params.temperature,
            max_tokens=self.params.max_tokens,
        )
        return response.choices[0].message.content or ""
