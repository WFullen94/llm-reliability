# llm-reliability

Measure whether you can trust your LLM — calibration, drift detection, and adversarial robustness in one library. Works with black-box APIs (OpenAI, Anthropic) and local models (MLX, Ollama, HuggingFace).

---

## Why

Most LLM tooling treats reliability as an afterthought. Benchmarks measure accuracy. Observability tools log latency. Nobody answers the harder questions:

- **When the model says it's confident, is it right?** (calibration)
- **Did the model's behavior silently change after an update?** (drift)
- **Does it give consistent answers to the same question asked differently?** (robustness)

llm-reliability answers these with statistical rigor, not vibes. No logit access required — all methods work against black-box APIs.

---

## Installation

```bash
git clone https://github.com/WFullen94/llm-reliability
cd llm-reliability
pip install -e .
```

For embedding-based methods (semantic entropy, semantic dispersion):

```bash
pip install sentence-transformers
```

---

## What's available

### Calibration metrics (Phase 1)

Compute how well a model's confidence tracks its actual accuracy.

```python
from llm_reliability import ece, mce, calibration_result, reliability_diagram

# You already have confidence scores (e.g. from logprobs)
y_true = [1, 0, 1, 1, 0, 1, 0, 0, 1, 1]
confidences = [0.9, 0.8, 0.7, 0.95, 0.6, 0.85, 0.75, 0.55, 0.9, 0.8]

print(ece(y_true, confidences))          # Expected Calibration Error
print(mce(y_true, confidences))          # Maximum Calibration Error

result = calibration_result(y_true, confidences)
print(result.summary())

fig = reliability_diagram(y_true, confidences, title="My Model")
fig.savefig("reliability.png")
```

### Verbalized confidence (Phase 1)

Ask the model to self-report its confidence, then measure how calibrated that self-report is.

```python
from llm_reliability import verbalized

def my_model(prompt: str) -> str:
    # your model call here
    return call_openai(prompt)

result = verbalized(my_model, questions, labels)
print(result.summary())
```

### Temperature scaling (Phase 1)

Post-hoc calibration for models where you have logits. Finds the optimal temperature T that minimizes calibration error.

```python
from llm_reliability import temperature_scale, apply_temperature

T = temperature_scale(logits, labels)        # fit T on a held-out set
probs = apply_temperature(logits, T)         # apply to new predictions
print(f"Optimal temperature: {T:.3f}")
```

### Conformal prediction (Phase 1)

Distribution-free coverage guarantee: given a calibration set, compute a threshold that covers true labels at rate 1-alpha.

```python
from llm_reliability import conformal_threshold

q = conformal_threshold(calibration_scores, alpha=0.1)  # 90% coverage
```

---

### Black-box calibration — no logits required (Phase 2)

All four methods work against any text API. They infer confidence from model behavior rather than internal probabilities.

#### Consistency / SelfCheckGPT (Manakul et al. 2023)

Ask the same question N times. Agreement rate = confidence proxy.

```python
from llm_reliability import consistency

result = consistency(my_model, questions, labels, n_samples=10)
print(result.summary())
```

#### P(True) (Kadavath et al. 2022)

Generate an answer, then ask "is this correct?" N times. Fraction of Yes = confidence.

```python
from llm_reliability import p_true

result = p_true(my_model, questions, labels, n_samples=5)
print(result.summary())
```

#### Semantic Dispersion (Lin et al. 2024)

Sample N responses, embed them, use mean pairwise similarity as confidence. High similarity = low uncertainty.

```python
from llm_reliability import semantic_dispersion

result = semantic_dispersion(my_model, questions, labels, n_samples=10)
print(result.summary())
```

#### Semantic Entropy (Kuhn et al. 2023)

Sample N responses, cluster by semantic equivalence, compute entropy over clusters. Low entropy = high confidence. Most rigorous of the four.

```python
from llm_reliability import semantic_entropy

result = semantic_entropy(my_model, questions, labels, n_samples=10)
print(result.summary())
```

---

### Comparing methods on the same model

All black-box methods store per-question scores for cross-method comparison:

```python
r1 = consistency(my_model, questions, labels)
r2 = semantic_entropy(my_model, questions, labels)

# raw_scores: list of (correct, confidence) per question
print(r1.raw_scores)   # [(1, 0.9), (0, 0.7), ...]
print(r2.raw_scores)   # [(1, 0.85), (0, 0.6), ...]
```

---

### CLI

```bash
# Calibration from a scores file
llm-reliability calibrate scores.json --plot reliability.png --output results.json

# Temperature scaling from logits
llm-reliability temperature-scale logits.json
```

`scores.json` format:
```json
[{"label": 1, "confidence": 0.9}, {"label": 0, "confidence": 0.7}]
```

`logits.json` format:
```json
{"logits": [[2.1, 0.3], [0.5, 1.8]], "labels": [0, 1]}
```

---

## API call cost

Black-box methods make multiple model calls per question:

| Method | Calls per question |
|---|---|
| `verbalized` | 1 |
| `consistency` | n_samples (default 10) |
| `p_true` | 1 + n_samples (default 6) |
| `semantic_dispersion` | n_samples (default 10) |
| `semantic_entropy` | n_samples (default 10) |

For 100 questions with n_samples=10: ~1000 API calls. Plan accordingly.

---

## Roadmap

| Phase | Status | What |
|---|---|---|
| 1 | ✅ Done | Calibration core — ECE, MCE, verbalized, temperature scaling, conformal |
| 2 | ✅ Done | Black-box calibration — semantic entropy, consistency, P(True), semantic dispersion |
| 3 | 🔧 In progress | Model adapters — OpenAI, Anthropic, MLX, Ollama |
| 4 | Planned | Drift detection — KS tests, MMD, behavioral fingerprinting |
| 5 | Planned | Adversarial robustness — consistency attacks, perturbation suite |
| 6 | Planned | Unified report + GitHub Action |

---

## Research

This project is also a research platform. Open directions tracked in [RESEARCH.md](RESEARCH.md):

1. **Meta-uncertainty** — when black-box methods disagree, does that disagreement predict errors?
2. **Calibration drift as fingerprint** — can ECE curve shape detect silent model updates?
3. **Semantic stability distance** — minimum perturbation to flip an answer as a reliability metric
4. **Method ranking consistency** — do different methods agree on which model is best calibrated?

---

## License

MIT
