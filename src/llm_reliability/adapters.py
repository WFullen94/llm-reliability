"""Model adapters — standardized interface for calling LLMs across providers.

Three tiers by logit access:
  Tier 1 (full logits):   MLXAdapter, HuggingFaceAdapter
  Tier 2 (log-probs):     OpenAIAdapter, OllamaAdapter
  Tier 3 (text only):     AnthropicAdapter

All adapters implement __call__(prompt) -> str so they drop straight into
any calibration function that takes a model_fn argument.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Response type
# ---------------------------------------------------------------------------

@dataclass
class ModelResponse:
    text: str
    # Sequence-level confidence in [0, 1] — None if not available for this tier.
    confidence: float | None = None
    # Token-level log-probs — None for Tier 3 providers.
    token_logprobs: list[float] | None = None
    # Raw pre-softmax logits — Tier 1 only.
    logits: list[list[float]] | None = None
    latency: float = 0.0


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class ModelAdapter(ABC):
    """Common interface for all model providers.

    Subclasses must implement generate(). __call__ delegates to generate()
    so adapters work as drop-in model_fn callables.
    """

    @abstractmethod
    def generate(self, prompt: str, temperature: float = 0.0, max_tokens: int = 256) -> ModelResponse:
        ...

    def __call__(self, prompt: str) -> str:
        return self.generate(prompt).text


# ---------------------------------------------------------------------------
# OpenAI (Tier 2 — token log-probs)
# ---------------------------------------------------------------------------

class OpenAIAdapter(ModelAdapter):
    """OpenAI chat completions with optional token log-probs.

    Args:
        model: Model ID, e.g. "gpt-4o", "gpt-4o-mini".
        api_key: OpenAI API key. Defaults to OPENAI_API_KEY env var.
        logprobs: If True, request top-1 token log-probs and compute
                  sequence confidence as mean token probability.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: str | None = None,
        logprobs: bool = True,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("pip install openai")
        self._client = OpenAI(api_key=api_key)
        self.model = model
        self.logprobs = logprobs

    def generate(self, prompt: str, temperature: float = 0.0, max_tokens: int = 256) -> ModelResponse:
        import numpy as np

        t0 = time.perf_counter()
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
            logprobs=self.logprobs,
            top_logprobs=1 if self.logprobs else None,
        )
        latency = time.perf_counter() - t0

        choice = response.choices[0]
        text = choice.message.content or ""

        token_logprobs = None
        confidence = None
        if self.logprobs and choice.logprobs and choice.logprobs.content:
            token_logprobs = [t.logprob for t in choice.logprobs.content]
            # Mean token probability as sequence confidence
            mean_logprob = float(np.mean(token_logprobs))
            confidence = float(np.exp(mean_logprob))

        return ModelResponse(
            text=text,
            confidence=confidence,
            token_logprobs=token_logprobs,
            latency=latency,
        )


# ---------------------------------------------------------------------------
# Anthropic (Tier 3 — text only)
# ---------------------------------------------------------------------------

class AnthropicAdapter(ModelAdapter):
    """Anthropic Messages API. No logprob access — text only.

    Use verbalized(), consistency(), p_true(), or semantic_entropy()
    for calibration; these are all text-based.

    Args:
        model: Model ID, e.g. "claude-sonnet-4-6", "claude-haiku-4-5-20251001".
        api_key: Anthropic API key. Defaults to ANTHROPIC_API_KEY env var.
    """

    def __init__(
        self,
        model: str = "claude-haiku-4-5-20251001",
        api_key: str | None = None,
    ) -> None:
        try:
            import anthropic
        except ImportError:
            raise ImportError("pip install anthropic")
        self._client = anthropic.Anthropic(api_key=api_key)
        self.model = model

    def generate(self, prompt: str, temperature: float = 0.0, max_tokens: int = 256) -> ModelResponse:
        t0 = time.perf_counter()
        message = self._client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        latency = time.perf_counter() - t0
        text = message.content[0].text if message.content else ""
        return ModelResponse(text=text, latency=latency)


# ---------------------------------------------------------------------------
# Ollama (Tier 2 — token log-probs)
# ---------------------------------------------------------------------------

class OllamaAdapter(ModelAdapter):
    """Ollama local server with optional log-probs.

    Args:
        model: Model name as listed in `ollama list`, e.g. "llama3.2", "mistral".
        host: Ollama server URL. Defaults to http://localhost:11434.
        logprobs: If True, request token log-probs from Ollama's API.
    """

    def __init__(
        self,
        model: str = "llama3.2",
        host: str = "http://localhost:11434",
        logprobs: bool = True,
    ) -> None:
        try:
            import ollama
        except ImportError:
            raise ImportError("pip install ollama")
        self._ollama = ollama
        self.model = model
        self.host = host
        self.logprobs = logprobs

    def generate(self, prompt: str, temperature: float = 0.0, max_tokens: int = 256) -> ModelResponse:
        import numpy as np

        t0 = time.perf_counter()
        response = self._ollama.generate(
            model=self.model,
            prompt=prompt,
            options={
                "temperature": temperature,
                "num_predict": max_tokens,
                "logprobs": self.logprobs,
            },
        )
        latency = time.perf_counter() - t0
        text = response.get("response", "")

        token_logprobs = None
        confidence = None
        if self.logprobs:
            raw = response.get("logprobs") or []
            if raw:
                token_logprobs = [entry.get("logprob", 0.0) for entry in raw if "logprob" in entry]
                if token_logprobs:
                    confidence = float(np.exp(np.mean(token_logprobs)))

        return ModelResponse(
            text=text,
            confidence=confidence,
            token_logprobs=token_logprobs,
            latency=latency,
        )


# ---------------------------------------------------------------------------
# MLX (Tier 1 — full logits, Apple Silicon)
# ---------------------------------------------------------------------------

class MLXAdapter(ModelAdapter):
    """MLX-LM backend for Apple Silicon. Full logit access.

    Args:
        model: HuggingFace model ID or local path, e.g.
               "mlx-community/Llama-3.2-3B-Instruct-4bit".
    """

    def __init__(self, model: str) -> None:
        try:
            import mlx_lm
        except ImportError:
            raise ImportError("pip install mlx-lm")
        self.model_id = model
        self._model, self._tokenizer = mlx_lm.load(model)

    def generate(self, prompt: str, temperature: float = 0.0, max_tokens: int = 256) -> ModelResponse:
        import mlx_lm
        import mlx.core as mx
        import numpy as np
        import inspect

        messages = [{"role": "user", "content": prompt}]
        formatted = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        gen_kwargs: dict = {"max_tokens": max_tokens}
        from mlx_lm.generate import generate_step
        gs_sig = inspect.signature(generate_step)
        if "sampler" in gs_sig.parameters:
            from mlx_lm.sample_utils import make_sampler
            gen_kwargs["sampler"] = make_sampler(temp=temperature)
        elif "temperature" in gs_sig.parameters:
            gen_kwargs["temperature"] = temperature
        else:
            gen_kwargs["temp"] = temperature

        t0 = time.perf_counter()
        output_text = ""
        for chunk in mlx_lm.stream_generate(
            self._model, self._tokenizer, prompt=formatted, **gen_kwargs
        ):
            if hasattr(chunk, "text"):
                output_text += chunk.text
            elif isinstance(chunk, tuple):
                output_text += chunk[0]
            else:
                output_text += chunk
        latency = time.perf_counter() - t0

        return ModelResponse(text=output_text, latency=latency)

    def logprobs_for(self, text: str) -> list[float]:
        """Return per-token log-probs for a complete text string."""
        import mlx.core as mx
        import mlx.nn as nn
        import numpy as np

        tokens = self._tokenizer.encode(text)
        tokens_mx = mx.array(tokens)
        logits = self._model(tokens_mx[None, :-1])
        log_probs = nn.log_softmax(logits[0], axis=-1)
        mx.eval(log_probs)
        per_token = [float(log_probs[i, tokens[i + 1]]) for i in range(len(tokens) - 1)]
        return per_token


# ---------------------------------------------------------------------------
# HuggingFace Transformers (Tier 1 — full logits)
# ---------------------------------------------------------------------------

class HuggingFaceAdapter(ModelAdapter):
    """HuggingFace Transformers backend. Full logit access via MPS or CPU.

    Args:
        model: HuggingFace model ID, e.g. "meta-llama/Llama-3.2-3B-Instruct".
        device: "mps", "cuda", or "cpu". Auto-detected if None.
    """

    def __init__(self, model: str, device: str | None = None) -> None:
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
        except ImportError:
            raise ImportError("pip install transformers torch")

        import torch
        if device is None:
            device = "mps" if torch.backends.mps.is_available() else \
                     "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device
        self._tokenizer = AutoTokenizer.from_pretrained(model)
        self._model = AutoModelForCausalLM.from_pretrained(
            model, torch_dtype=torch.float16
        ).to(device)
        self._model.eval()

    def generate(self, prompt: str, temperature: float = 0.0, max_tokens: int = 256) -> ModelResponse:
        import torch
        import numpy as np

        inputs = self._tokenizer(prompt, return_tensors="pt").to(self.device)
        t0 = time.perf_counter()
        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=temperature > 0,
                temperature=temperature if temperature > 0 else 1.0,
                return_dict_in_generate=True,
                output_scores=True,
            )
        latency = time.perf_counter() - t0

        input_len = inputs["input_ids"].shape[1]
        generated_ids = outputs.sequences[0][input_len:]
        text = self._tokenizer.decode(generated_ids, skip_special_tokens=True)

        # Compute per-token log-probs from scores
        token_logprobs = None
        confidence = None
        if outputs.scores:
            log_probs = [
                torch.log_softmax(score, dim=-1)[0, tok].item()
                for score, tok in zip(outputs.scores, generated_ids)
            ]
            token_logprobs = log_probs
            confidence = float(np.exp(np.mean(log_probs)))

        return ModelResponse(
            text=text,
            confidence=confidence,
            token_logprobs=token_logprobs,
            latency=latency,
        )
