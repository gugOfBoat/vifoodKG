from __future__ import annotations

import os
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vifood_eval.models import HFVisionModel, OpenAICompatibleModel, _patch_dynamic_cache_legacy_api


class FakeCompletions:
    def __init__(self) -> None:
        self.requests: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> object:
        self.requests.append(kwargs)
        if kwargs.get("response_format") and len(self.requests) == 1:
            raise RuntimeError("unsupported response_format")
        return types.SimpleNamespace(
            choices=[types.SimpleNamespace(message=types.SimpleNamespace(content="Answer: B"))]
        )


class FakeOpenAI:
    last_client: "FakeOpenAI | None" = None

    def __init__(self, *, api_key: str, base_url: str | None = None) -> None:
        self.api_key = api_key
        self.base_url = base_url
        self.completions = FakeCompletions()
        self.chat = types.SimpleNamespace(completions=self.completions)
        FakeOpenAI.last_client = self


class FakeProcessorLoader:
    requests: list[dict[str, object]] = []
    processor: "FakeProcessor | object" = object()

    @classmethod
    def from_pretrained(cls, model_id: str, **kwargs: object) -> object:
        cls.requests.append({"model_id": model_id, **kwargs})
        return cls.processor


class FakeTensor:
    shape = (1, 1)

    def to(self, device: str) -> "FakeTensor":
        return self

    def __getitem__(self, key: object) -> "FakeTensor":
        return self


class FakeProcessor:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def __call__(self, *, text: object, images: object, return_tensors: str) -> dict[str, FakeTensor]:
        self.calls.append({"text": text, "images": images, "return_tensors": return_tensors})
        return {"input_ids": FakeTensor()}

    def batch_decode(self, tokens: object, skip_special_tokens: bool) -> list[str]:
        return ["Answer: B"]


class FakeModelConfig:
    def __init__(self) -> None:
        self._attn_implementation_internal = "flash_attention_2"
        self.use_flash_attention_2 = True


class FakeConfigLoader:
    requests: list[dict[str, object]] = []
    last_config: FakeModelConfig | None = None

    @classmethod
    def from_pretrained(cls, model_id: str, **kwargs: object) -> FakeModelConfig:
        cls.requests.append({"model_id": model_id, **kwargs})
        cls.last_config = FakeModelConfig()
        return cls.last_config


class FakeLoadedModel:
    device = "cpu"

    def eval(self) -> None:
        return None

    def generate(self, **kwargs: object) -> FakeTensor:
        return FakeTensor()


class FakeNoGrad:
    def __enter__(self) -> None:
        return None

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        return None


class FakeTorch:
    @staticmethod
    def no_grad() -> FakeNoGrad:
        return FakeNoGrad()


class FakeDynamicCache:
    calls: list[object] = []

    def __init__(self, ddp_cache_data: object = None) -> None:
        self.ddp_cache_data = ddp_cache_data


class FakeCausalLM:
    requests: list[dict[str, object]] = []

    @classmethod
    def from_pretrained(cls, model_id: str, **kwargs: object) -> FakeLoadedModel:
        cls.requests.append({"model_id": model_id, **kwargs})
        return FakeLoadedModel()


class FakeImageTextToText:
    requests: list[dict[str, object]] = []

    @classmethod
    def from_pretrained(cls, model_id: str, **kwargs: object) -> FakeLoadedModel:
        cls.requests.append({"model_id": model_id, **kwargs})
        return FakeLoadedModel()


class OpenAICompatibleModelTests(unittest.TestCase):
    def test_json_response_format_falls_back_when_provider_rejects_it(self) -> None:
        fake_openai = types.SimpleNamespace(OpenAI=FakeOpenAI)
        messages = [{"role": "user", "content": [{"type": "text", "text": "Return JSON."}]}]

        with patch.dict(sys.modules, {"openai": fake_openai}):
            with patch.dict(os.environ, {"TEST_API_KEY": "secret"}, clear=False):
                model = OpenAICompatibleModel({"model_id": "fake", "api_key_env": "TEST_API_KEY"})
                response = model.generate(
                    messages,
                    max_new_tokens=16,
                    temperature=0,
                    response_format={"type": "json_object"},
                )

        self.assertEqual(response, "Answer: B")
        assert FakeOpenAI.last_client is not None
        requests = FakeOpenAI.last_client.completions.requests
        self.assertEqual(requests[0]["response_format"], {"type": "json_object"})
        self.assertNotIn("response_format", requests[1])


class HFVisionModelTests(unittest.TestCase):
    def setUp(self) -> None:
        FakeConfigLoader.requests = []
        FakeConfigLoader.last_config = None
        FakeProcessorLoader.requests = []
        FakeProcessorLoader.processor = object()
        FakeCausalLM.requests = []
        FakeImageTextToText.requests = []

    def test_phi_config_can_force_causal_lm_without_flash_attention(self) -> None:
        fake_transformers = types.SimpleNamespace(
            AutoConfig=FakeConfigLoader,
            AutoProcessor=FakeProcessorLoader,
            AutoModelForCausalLM=FakeCausalLM,
            AutoModelForImageTextToText=FakeImageTextToText,
        )

        with patch.dict(sys.modules, {"torch": types.SimpleNamespace(), "transformers": fake_transformers}):
            HFVisionModel(
                {
                    "model_id": "microsoft/Phi-3.5-vision-instruct",
                    "adapter": "phi3_vision",
                    "auto_model": "causal_lm",
                    "load_config_first": True,
                    "device_map": "auto",
                    "torch_dtype": "auto",
                    "attn_implementation": "eager",
                    "processor_use_fast": False,
                    "trust_remote_code": True,
                }
            )

        self.assertEqual(FakeImageTextToText.requests, [])
        assert FakeConfigLoader.last_config is not None
        self.assertEqual(FakeConfigLoader.last_config._attn_implementation_internal, "eager")
        self.assertEqual(FakeConfigLoader.last_config._attn_implementation, "eager")
        self.assertEqual(FakeConfigLoader.last_config.use_flash_attention_2, False)
        self.assertEqual(FakeProcessorLoader.requests[0]["use_fast"], False)
        self.assertIs(FakeCausalLM.requests[0]["config"], FakeConfigLoader.last_config)
        self.assertEqual(FakeCausalLM.requests[0]["attn_implementation"], "eager")
        self.assertEqual(FakeCausalLM.requests[0]["torch_dtype"], "auto")

    def test_phi_processor_receives_single_text_prompt(self) -> None:
        processor = FakeProcessor()
        FakeProcessorLoader.processor = processor
        fake_transformers = types.SimpleNamespace(
            AutoConfig=FakeConfigLoader,
            AutoProcessor=FakeProcessorLoader,
            AutoModelForCausalLM=FakeCausalLM,
            AutoModelForImageTextToText=FakeImageTextToText,
        )

        with patch.dict(sys.modules, {"torch": FakeTorch, "transformers": fake_transformers}):
            model = HFVisionModel(
                {
                    "model_id": "microsoft/Phi-3.5-vision-instruct",
                    "adapter": "phi3_vision",
                    "auto_model": "causal_lm",
                    "load_config_first": True,
                    "device_map": "auto",
                    "torch_dtype": "auto",
                    "attn_implementation": "eager",
                    "processor_use_fast": False,
                    "trust_remote_code": True,
                }
            )
            response = model.generate(
                [{"role": "user", "content": [{"type": "text", "text": "Question?"}]}],
                max_new_tokens=16,
                temperature=0,
            )

        self.assertEqual(response, "Answer: B")
        self.assertIsInstance(processor.calls[0]["text"], str)
        self.assertNotIsInstance(processor.calls[0]["text"], list)

    def test_phi_dynamic_cache_legacy_api_is_patched_for_transformers_5(self) -> None:
        if hasattr(FakeDynamicCache, "from_legacy_cache"):
            delattr(FakeDynamicCache, "from_legacy_cache")

        fake_cache_utils = types.SimpleNamespace(DynamicCache=FakeDynamicCache)
        with patch.dict(sys.modules, {"transformers.cache_utils": fake_cache_utils}):
            _patch_dynamic_cache_legacy_api()

        self.assertTrue(hasattr(FakeDynamicCache, "from_legacy_cache"))
        empty_cache = FakeDynamicCache.from_legacy_cache()
        self.assertIsInstance(empty_cache, FakeDynamicCache)
        legacy_cache = [("key", "value")]
        converted_cache = FakeDynamicCache.from_legacy_cache(legacy_cache)
        self.assertIs(converted_cache.ddp_cache_data, legacy_cache)


if __name__ == "__main__":
    unittest.main()
