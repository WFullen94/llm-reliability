"""Tests for model adapters — uses mocks, no real API calls."""

from unittest.mock import MagicMock, patch
import pytest

from llm_reliability.adapters import (
    ModelAdapter,
    ModelResponse,
    OpenAIAdapter,
    AnthropicAdapter,
    OllamaAdapter,
)


# ---------------------------------------------------------------------------
# ModelResponse
# ---------------------------------------------------------------------------

def test_model_response_defaults():
    r = ModelResponse(text="hello")
    assert r.text == "hello"
    assert r.confidence is None
    assert r.token_logprobs is None
    assert r.logits is None
    assert r.latency == 0.0


def test_model_response_with_confidence():
    r = ModelResponse(text="yes", confidence=0.87, token_logprobs=[-0.14])
    assert r.confidence == pytest.approx(0.87)
    assert r.token_logprobs == [-0.14]


# ---------------------------------------------------------------------------
# ModelAdapter base: __call__ delegates to generate()
# ---------------------------------------------------------------------------

class _EchoAdapter(ModelAdapter):
    def generate(self, prompt, temperature=0.0, max_tokens=256):
        return ModelResponse(text=f"echo:{prompt}")


def test_adapter_callable():
    adapter = _EchoAdapter()
    assert adapter("hello") == "echo:hello"


def test_adapter_callable_used_as_model_fn():
    from llm_reliability import consistency
    adapter = _EchoAdapter()
    questions = ["q1", "q2", "q3"]
    labels = ["echo:q1", "echo:q2", "echo:q3"]
    result = consistency(adapter, questions, labels, n_samples=3)
    assert result.n_samples == 3


# ---------------------------------------------------------------------------
# OpenAIAdapter (mocked)
# ---------------------------------------------------------------------------

def _mock_openai_response(text: str, logprob: float = -0.1):
    token = MagicMock()
    token.logprob = logprob

    logprobs_content = MagicMock()
    logprobs_content.__iter__ = MagicMock(return_value=iter([token]))

    choice = MagicMock()
    choice.message.content = text
    choice.logprobs.content = [token]

    response = MagicMock()
    response.choices = [choice]
    return response


@patch("llm_reliability.adapters.OpenAIAdapter.__init__", lambda self, **kw: None)
def test_openai_adapter_extracts_text():
    adapter = OpenAIAdapter.__new__(OpenAIAdapter)
    adapter.model = "gpt-4o-mini"
    adapter.logprobs = True
    adapter._client = MagicMock()
    adapter._client.chat.completions.create.return_value = _mock_openai_response("Paris")

    response = adapter.generate("What is the capital of France?")
    assert response.text == "Paris"


@patch("llm_reliability.adapters.OpenAIAdapter.__init__", lambda self, **kw: None)
def test_openai_adapter_computes_confidence():
    import numpy as np
    adapter = OpenAIAdapter.__new__(OpenAIAdapter)
    adapter.model = "gpt-4o-mini"
    adapter.logprobs = True
    adapter._client = MagicMock()
    logprob = -0.2
    adapter._client.chat.completions.create.return_value = _mock_openai_response(
        "Paris", logprob=logprob
    )

    response = adapter.generate("Capital of France?")
    expected_conf = float(np.exp(logprob))
    assert response.confidence == pytest.approx(expected_conf, abs=1e-4)


@patch("llm_reliability.adapters.OpenAIAdapter.__init__", lambda self, **kw: None)
def test_openai_adapter_no_logprobs():
    adapter = OpenAIAdapter.__new__(OpenAIAdapter)
    adapter.model = "gpt-4o-mini"
    adapter.logprobs = False
    adapter._client = MagicMock()

    choice = MagicMock()
    choice.message.content = "Berlin"
    choice.logprobs = None
    response_mock = MagicMock()
    response_mock.choices = [choice]
    adapter._client.chat.completions.create.return_value = response_mock

    response = adapter.generate("Capital?")
    assert response.text == "Berlin"
    assert response.confidence is None


# ---------------------------------------------------------------------------
# AnthropicAdapter (mocked)
# ---------------------------------------------------------------------------

@patch("llm_reliability.adapters.AnthropicAdapter.__init__", lambda self, **kw: None)
def test_anthropic_adapter_text_only():
    adapter = AnthropicAdapter.__new__(AnthropicAdapter)
    adapter.model = "claude-haiku-4-5-20251001"
    adapter._client = MagicMock()

    content_block = MagicMock()
    content_block.text = "42"
    message = MagicMock()
    message.content = [content_block]
    adapter._client.messages.create.return_value = message

    response = adapter.generate("What is 6 * 7?")
    assert response.text == "42"
    assert response.confidence is None  # Tier 3 — no logprobs
    assert response.token_logprobs is None


@patch("llm_reliability.adapters.AnthropicAdapter.__init__", lambda self, **kw: None)
def test_anthropic_adapter_callable():
    adapter = AnthropicAdapter.__new__(AnthropicAdapter)
    adapter.model = "claude-haiku-4-5-20251001"
    adapter._client = MagicMock()

    content_block = MagicMock()
    content_block.text = "Answer: 4\nConfidence: 90"
    message = MagicMock()
    message.content = [content_block]
    adapter._client.messages.create.return_value = message

    assert adapter("some prompt") == "Answer: 4\nConfidence: 90"


# ---------------------------------------------------------------------------
# OllamaAdapter (mocked)
# ---------------------------------------------------------------------------

@patch("llm_reliability.adapters.OllamaAdapter.__init__", lambda self, **kw: None)
def test_ollama_adapter_with_logprobs():
    import numpy as np
    adapter = OllamaAdapter.__new__(OllamaAdapter)
    adapter.model = "llama3.2"
    adapter.host = "http://localhost:11434"
    adapter.logprobs = True
    adapter._ollama = MagicMock()
    adapter._ollama.generate.return_value = {
        "response": "Paris",
        "logprobs": [{"logprob": -0.1}, {"logprob": -0.2}],
    }

    response = adapter.generate("Capital of France?")
    assert response.text == "Paris"
    assert response.token_logprobs == [-0.1, -0.2]
    expected_conf = float(np.exp(np.mean([-0.1, -0.2])))
    assert response.confidence == pytest.approx(expected_conf, abs=1e-4)


@patch("llm_reliability.adapters.OllamaAdapter.__init__", lambda self, **kw: None)
def test_ollama_adapter_no_logprobs():
    adapter = OllamaAdapter.__new__(OllamaAdapter)
    adapter.model = "llama3.2"
    adapter.host = "http://localhost:11434"
    adapter.logprobs = False
    adapter._ollama = MagicMock()
    adapter._ollama.generate.return_value = {"response": "Berlin"}

    response = adapter.generate("Capital of Germany?")
    assert response.text == "Berlin"
    assert response.confidence is None


# ---------------------------------------------------------------------------
# Import errors give helpful messages
# ---------------------------------------------------------------------------

def test_openai_missing_import():
    with patch.dict("sys.modules", {"openai": None}):
        with pytest.raises(ImportError, match="pip install openai"):
            OpenAIAdapter(model="gpt-4o-mini")


def test_anthropic_missing_import():
    with patch.dict("sys.modules", {"anthropic": None}):
        with pytest.raises(ImportError, match="pip install anthropic"):
            AnthropicAdapter(model="claude-haiku-4-5-20251001")
