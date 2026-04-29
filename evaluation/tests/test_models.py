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


class FakeBatchInputs(dict):
    def __init__(self) -> None:
        super().__init__({"input_ids": FakeTensor()})
        self.device: str | None = None

    def to(self, device: str) -> "FakeBatchInputs":
        self.device = device
        return self


class FakeProcessor:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []
        self.tokenizer = types.SimpleNamespace(eos_token_id=32000, pad_token_id=0)

    def __call__(self, *, text: object, images: object, return_tensors: str) -> dict[str, FakeTensor]:
        self.calls.append({"text": text, "images": images, "return_tensors": return_tensors})
        return {"input_ids": FakeTensor()}

    def apply_chat_template(self, messages: object, **kwargs: object) -> FakeBatchInputs | str:
        self.calls.append({"messages": messages, **kwargs})
        if kwargs.get("tokenize"):
            return FakeBatchInputs()
        return "templated prompt"

    def batch_decode(
        self,
        tokens: object,
        skip_special_tokens: bool,
        clean_up_tokenization_spaces: bool | None = None,
    ) -> list[str]:
        self.calls.append(
            {
                "tokens": tokens,
                "skip_special_tokens": skip_special_tokens,
                "clean_up_tokenization_spaces": clean_up_tokenization_spaces,
            }
        )
        return ["Answer: B"]


class FakeModelConfig:
    def __init__(self) -> None:
        self._attn_implementation_internal = "flash_attention_2"
        self.use_flash_attention_2 = True
        self.use_cache = True


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
    generate_calls: list[dict[str, object]] = []

    def __init__(self) -> None:
        self.config = FakeModelConfig()

    def eval(self) -> None:
        return None

    def generate(self, **kwargs: object) -> FakeTensor:
        self.generate_calls.append(kwargs)
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
    max_cache_shape = 8

    def __init__(self, ddp_cache_data: object = None) -> None:
        self.ddp_cache_data = ddp_cache_data

    def get_seq_length(self, layer_idx: int = 0) -> int:
        return 5

    def get_max_cache_shape(self, layer_idx: int = 0) -> int:
        return self.max_cache_shape


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


class FakeQwen3VL:
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
        FakeLoadedModel.generate_calls = []
        FakeImageTextToText.requests = []
        FakeQwen3VL.requests = []

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
                    "use_cache": True,
                    "trust_remote_code": True,
                }
            )

        self.assertEqual(FakeImageTextToText.requests, [])
        assert FakeConfigLoader.last_config is not None
        self.assertEqual(FakeConfigLoader.last_config._attn_implementation_internal, "eager")
        self.assertEqual(FakeConfigLoader.last_config._attn_implementation, "eager")
        self.assertEqual(FakeConfigLoader.last_config.use_flash_attention_2, False)
        self.assertEqual(FakeConfigLoader.last_config.use_cache, True)
        self.assertEqual(FakeProcessorLoader.requests[0]["use_fast"], False)
        self.assertEqual(FakeProcessorLoader.requests[0]["num_crops"], 16)
        self.assertIs(FakeCausalLM.requests[0]["config"], FakeConfigLoader.last_config)
        self.assertEqual(FakeCausalLM.requests[0]["attn_implementation"], "eager")
        self.assertEqual(FakeCausalLM.requests[0]["torch_dtype"], "auto")

    def test_phi_loads_with_older_transformers_without_image_text_to_text_auto_model(self) -> None:
        fake_transformers = types.SimpleNamespace(
            AutoConfig=FakeConfigLoader,
            AutoProcessor=FakeProcessorLoader,
            AutoModelForCausalLM=FakeCausalLM,
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
                    "num_crops": 16,
                    "use_cache": True,
                    "trust_remote_code": True,
                }
            )

        self.assertEqual(len(FakeCausalLM.requests), 1)
        self.assertEqual(FakeCausalLM.requests[0]["model_id"], "microsoft/Phi-3.5-vision-instruct")

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
                    "use_cache": True,
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
        self.assertEqual(FakeLoadedModel.generate_calls[0]["use_cache"], True)
        self.assertEqual(FakeLoadedModel.generate_calls[0]["eos_token_id"], 32000)
        self.assertEqual(processor.calls[-1]["clean_up_tokenization_spaces"], False)

    def test_qwen3_uses_processor_native_chat_template(self) -> None:
        processor = FakeProcessor()
        FakeProcessorLoader.processor = processor
        fake_transformers = types.SimpleNamespace(
            AutoConfig=FakeConfigLoader,
            AutoProcessor=FakeProcessorLoader,
            AutoModelForCausalLM=FakeCausalLM,
            AutoModelForImageTextToText=FakeImageTextToText,
            Qwen3VLForConditionalGeneration=FakeQwen3VL,
        )

        with patch.dict(sys.modules, {"torch": FakeTorch, "transformers": fake_transformers}):
            model = HFVisionModel(
                {
                    "model_id": "Qwen/Qwen3-VL-2B-Instruct",
                    "adapter": "qwen3_vl",
                    "device_map": "auto",
                    "torch_dtype": "auto",
                    "generation_kwargs": {
                        "repetition_penalty": 1.05,
                        "eos_token_id": "tokenizer",
                    },
                }
            )
            response = model.generate(
                [{"role": "user", "content": [{"type": "text", "text": "Question?"}]}],
                max_new_tokens=16,
                temperature=0,
            )

        self.assertEqual(response, "Answer: B")
        self.assertEqual(FakeQwen3VL.requests[0]["dtype"], "auto")
        self.assertNotIn("torch_dtype", FakeQwen3VL.requests[0])
        self.assertEqual(processor.calls[0]["tokenize"], True)
        self.assertEqual(processor.calls[0]["return_dict"], True)
        self.assertEqual(processor.calls[0]["return_tensors"], "pt")
        self.assertEqual(FakeLoadedModel.generate_calls[0]["repetition_penalty"], 1.05)
        self.assertEqual(FakeLoadedModel.generate_calls[0]["eos_token_id"], 32000)

    def test_phi_dynamic_cache_legacy_api_is_patched_for_transformers_5(self) -> None:
        if hasattr(FakeDynamicCache, "from_legacy_cache"):
            delattr(FakeDynamicCache, "from_legacy_cache")
        if hasattr(FakeDynamicCache, "get_usable_length"):
            delattr(FakeDynamicCache, "get_usable_length")
        if hasattr(FakeDynamicCache, "seen_tokens"):
            delattr(FakeDynamicCache, "seen_tokens")
        if hasattr(FakeDynamicCache, "get_max_length"):
            delattr(FakeDynamicCache, "get_max_length")
        if hasattr(FakeDynamicCache, "to_legacy_cache"):
            delattr(FakeDynamicCache, "to_legacy_cache")

        fake_cache_utils = types.SimpleNamespace(DynamicCache=FakeDynamicCache)
        with patch.dict(sys.modules, {"transformers.cache_utils": fake_cache_utils}):
            _patch_dynamic_cache_legacy_api()

        self.assertTrue(hasattr(FakeDynamicCache, "from_legacy_cache"))
        empty_cache = FakeDynamicCache.from_legacy_cache()
        self.assertIsInstance(empty_cache, FakeDynamicCache)
        legacy_cache = [("key", "value")]
        converted_cache = FakeDynamicCache.from_legacy_cache(legacy_cache)
        self.assertIs(converted_cache.ddp_cache_data, legacy_cache)
        self.assertEqual(empty_cache.get_usable_length(2), 5)
        self.assertEqual(empty_cache.get_usable_length(4), 4)
        self.assertEqual(empty_cache.seen_tokens, 5)
        self.assertEqual(empty_cache.get_max_length(), 8)
        FakeDynamicCache.max_cache_shape = -1
        self.assertIsNone(empty_cache.get_max_length())
        self.assertEqual(empty_cache.get_usable_length(4), 5)
        FakeDynamicCache.max_cache_shape = 8
        self.assertEqual(converted_cache.to_legacy_cache(), tuple(legacy_cache))


if __name__ == "__main__":
    unittest.main()
