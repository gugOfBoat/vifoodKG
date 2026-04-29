from __future__ import annotations

import os
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vifood_eval.models import OpenAICompatibleModel


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


if __name__ == "__main__":
    unittest.main()
