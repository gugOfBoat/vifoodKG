from __future__ import annotations

import base64
import mimetypes
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class VisionModel(ABC):
    @abstractmethod
    def generate(
        self,
        messages: list[dict[str, Any]],
        *,
        max_new_tokens: int,
        temperature: float,
        response_format: dict[str, Any] | None = None,
    ) -> str:
        raise NotImplementedError


def make_model(name: str, cfg: dict[str, Any]) -> VisionModel:
    model_type = cfg.get("type")
    if model_type == "openai_compatible":
        return OpenAICompatibleModel(cfg)
    if model_type == "hf":
        return HFVisionModel(cfg)
    raise ValueError(f"Unsupported model type for {name}: {model_type}")


class OpenAICompatibleModel(VisionModel):
    def __init__(self, cfg: dict[str, Any]) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("Install the 'api' extra to use OpenAI-compatible models.") from exc

        api_key = os.getenv(cfg.get("api_key_env", "OPENAI_COMPAT_API_KEY"))
        base_url = os.getenv(cfg.get("base_url_env", "OPENAI_COMPAT_BASE_URL")) or None
        if not api_key:
            raise RuntimeError("Missing OpenAI-compatible API key environment variable.")

        self.model_id = cfg["model_id"]
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.use_response_format = bool(cfg.get("json_response_format", True))
        self.response_format_fallback = bool(cfg.get("json_response_format_fallback", True))

    def generate(
        self,
        messages: list[dict[str, Any]],
        *,
        max_new_tokens: int,
        temperature: float,
        response_format: dict[str, Any] | None = None,
    ) -> str:
        request: dict[str, Any] = {
            "model": self.model_id,
            "messages": _messages_to_openai(messages),
            "temperature": temperature,
            "max_tokens": max_new_tokens,
        }
        if response_format and self.use_response_format:
            request["response_format"] = response_format

        try:
            response = self.client.chat.completions.create(**request)
        except Exception as exc:
            if (
                "response_format" not in request
                or not self.response_format_fallback
                or not _looks_like_response_format_error(exc)
            ):
                raise
            request.pop("response_format", None)
            response = self.client.chat.completions.create(**request)
        return response.choices[0].message.content or ""


class HFVisionModel(VisionModel):
    def __init__(self, cfg: dict[str, Any]) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoProcessor
        except ImportError as exc:
            raise RuntimeError("Install the 'hf' extra to use local Hugging Face models.") from exc

        self.torch = torch
        self.adapter = cfg.get("adapter", "qwen_vl")
        self.processor = AutoProcessor.from_pretrained(
            cfg["model_id"],
            trust_remote_code=cfg.get("trust_remote_code", True),
        )
        model_kwargs = {
            "device_map": cfg.get("device_map", "auto"),
            "trust_remote_code": cfg.get("trust_remote_code", True),
        }
        if cfg.get("torch_dtype", "auto") == "auto":
            model_kwargs["torch_dtype"] = "auto"

        try:
            self.model = AutoModelForImageTextToText.from_pretrained(cfg["model_id"], **model_kwargs)
        except Exception:
            self.model = AutoModelForCausalLM.from_pretrained(cfg["model_id"], **model_kwargs)
        self.model.eval()

    def generate(
        self,
        messages: list[dict[str, Any]],
        *,
        max_new_tokens: int,
        temperature: float,
        response_format: dict[str, Any] | None = None,
    ) -> str:
        _ = response_format
        if self.adapter == "phi3_vision":
            prompt, images = _messages_to_phi(messages)
        else:
            prompt, images = _messages_to_chat_template(self.processor, messages)

        inputs = self.processor(text=[prompt], images=images, return_tensors="pt")
        inputs = {key: value.to(self.model.device) for key, value in inputs.items()}
        generate_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0,
        }
        if temperature > 0:
            generate_kwargs["temperature"] = temperature

        with self.torch.no_grad():
            output_ids = self.model.generate(**inputs, **generate_kwargs)

        input_len = inputs["input_ids"].shape[-1]
        new_tokens = output_ids[:, input_len:]
        decoded = self.processor.batch_decode(new_tokens, skip_special_tokens=True)
        return decoded[0].strip()


def _messages_to_openai(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    converted: list[dict[str, Any]] = []
    for message in messages:
        parts: list[dict[str, Any]] = []
        for part in message["content"]:
            if part["type"] == "text":
                parts.append({"type": "text", "text": part["text"]})
            elif part["type"] == "image":
                parts.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": _image_to_data_url(Path(part["path"]))},
                    }
                )
        converted.append({"role": message["role"], "content": parts})
    return converted


def _looks_like_response_format_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return "response_format" in message or "response format" in message


def _messages_to_chat_template(processor: Any, messages: list[dict[str, Any]]) -> tuple[str, list[Any]]:
    images = []
    converted = []
    for message in messages:
        content = []
        for part in message["content"]:
            if part["type"] == "text":
                content.append({"type": "text", "text": part["text"]})
            elif part["type"] == "image":
                image = _load_image(Path(part["path"]))
                images.append(image)
                content.append({"type": "image"})
        converted.append({"role": message["role"], "content": content})
    prompt = processor.apply_chat_template(converted, tokenize=False, add_generation_prompt=True)
    return prompt, images


def _messages_to_phi(messages: list[dict[str, Any]]) -> tuple[str, list[Any]]:
    images = []
    chunks: list[str] = []
    image_idx = 1
    for message in messages:
        if message["role"] == "system":
            text = "\n".join(part["text"] for part in message["content"] if part["type"] == "text")
            chunks.append(f"<|system|>\n{text}<|end|>\n")
            continue
        if message["role"] == "assistant":
            text = "\n".join(part["text"] for part in message["content"] if part["type"] == "text")
            chunks.append(f"<|assistant|>\n{text}<|end|>\n")
            continue

        user_parts = ["<|user|>\n"]
        for part in message["content"]:
            if part["type"] == "image":
                images.append(_load_image(Path(part["path"])))
                user_parts.append(f"<|image_{image_idx}|>\n")
                image_idx += 1
            elif part["type"] == "text":
                user_parts.append(part["text"])
        user_parts.append("<|end|>\n")
        chunks.append("".join(user_parts))
    chunks.append("<|assistant|>\n")
    return "".join(chunks), images


def _image_to_data_url(path: Path) -> str:
    mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def _load_image(path: Path) -> Any:
    from PIL import Image

    return Image.open(path).convert("RGB")
