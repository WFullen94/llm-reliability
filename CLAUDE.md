# llm-reliability

Toolkit for measuring and monitoring the reliability of LLMs in production — calibration, drift detection, and adversarial robustness in one composable library.

**GitHub:** `github.com/wfullen/llm-reliability`
**Thesis:** We are scaling AI rapidly. This project answers "can we trust what the model says?" and "did the model's behavior change?" with statistical rigor, not vibes.

---

## Three modules

### 1. `calibration` — Does the model know what it doesn't know?
When a model says it's 90% confident, is it right 90% of the time? Calibration measures the gap between stated confidence and actual accuracy. Poor calibration means you can't trust uncertainty signals — dangerous for routing, human-in-the-loop decisions, or any system that acts on model confidence.

**Techniques:**
- Expected Calibration Error (ECE) and reliability diagrams
- Verbalized confidence (asking the model "how confident are you?")
- Logit-based confidence (token probabilities where API exposes them)
- Temperature scaling and Platt calibration
- Conformal prediction intervals — distribution-free coverage guarantees

**Prior art gap:** Academic papers (Kadavath et al., 2022 on verbalized uncertainty; Angelopoulos et al. on conformal) but no production-grade open tooling.

### 2. `drift` — Did the model's behavior change?
Detects behavioral regressions when a model or prompt changes — using statistical distribution tests on output distributions, not point-in-time evals. Answers "did this update meaningfully shift how the model behaves?" with a p-value.

**Techniques:**
- KS test on output length distributions
- Jensen-Shannon divergence on embedding distributions
- Semantic drift via sentence embeddings
- Silent model update detection (API provider swaps model without notice)

**Prior art gap:** DeepEval/Promptfoo run evals but treat each run independently — no regression detection, no distribution comparison, no significance testing.

### 3. `adversarial` — Where does the model break?
Structured red-teaming: generate adversarial inputs, measure failure modes, report a robustness score. Not a jailbreak tool — a reliability audit.

**Techniques:**
- Perturbation testing (paraphrase, synonym swap, typo injection)
- Semantic equivalence checking (same question, different phrasing — does the model answer consistently?)
- Out-of-distribution input detection
- Failure mode taxonomy and classification

**Prior art gap:** Garak does jailbreak testing. Nobody does systematic consistency/robustness testing oriented toward production reliability.

---

## Phased Roadmap

### Phase 1 — Calibration foundation
- [ ] `calibration.ece(predictions, confidences)` — Expected Calibration Error
- [ ] `calibration.reliability_diagram(predictions, confidences)` — plot
- [ ] `calibration.verbalized(model_fn, prompts, labels)` — ask model for confidence, measure ECE
- [ ] `calibration.temperature_scale(logits, labels)` — fit temperature parameter
- [ ] CLI: `llm-reliability calibrate --model gpt-4o --dataset mmlu`

### Phase 2 — Drift detection
- [ ] `drift.capture(model_fn, prompts, n=100)` — store output distribution baseline
- [ ] `drift.compare(baseline, current)` — KS test + JS divergence + semantic drift
- [ ] `drift.report(result)` — human-readable with p-values and example diffs
- [ ] CLI: `llm-reliability drift compare --baseline baseline.json --current current.json`

### Phase 3 — Adversarial robustness
- [ ] `adversarial.perturb(prompts)` — generate paraphrase/typo/synonym variants
- [ ] `adversarial.consistency_score(model_fn, prompt_variants)` — measure answer stability
- [ ] `adversarial.ood_score(model_fn, prompts)` — out-of-distribution input detection
- [ ] CLI: `llm-reliability audit --model gpt-4o --prompts prompts.jsonl`

### Phase 4 — Unified report
- [ ] Single `llm-reliability report` command: runs calibration + drift + adversarial, outputs HTML/Markdown report
- [ ] JSON schema for reliability scores — composable with benchpress speed results

### Phase 5 — Distribution
- [ ] PyPI package
- [ ] GitHub Actions action: run reliability checks on model update PRs
- [ ] Integration guide with LangChain, LlamaIndex, raw OpenAI SDK

---

## Related work in this repo

- `~/llm-uncertainty-reliability/` — 12 notebooks covering verbalized uncertainty, logit-based confidence, semantic entropy, conformal prediction, hallucination detection, RAG faithfulness. These are the research foundation for the `calibration` module.
- `~/statistical-ml-evaluation/` — 17 notebooks on bootstrap CIs, calibration, distribution shift detection, proper scoring rules. Foundation for the `drift` module's statistical tests.
- `~/drift/` — earlier standalone scoping of the drift detection concept. Superseded by this project.
- `~/benchpress/` — sister project for speed benchmarking. llm-reliability is the quality/trust counterpart.

---

## Conventions

- Conventional commits: `feat:`, `fix:`, `docs:`, `refactor:`, `chore:`
- Tag each phase: `v0.1.0`, `v0.2.0`, etc.
- Model-agnostic: works with any callable `model_fn(prompt) -> str` — OpenAI, Anthropic, local MLX, Ollama
- scipy/statsmodels for statistical tests
- sentence-transformers for semantic drift
- Outputs interpretable by non-statisticians — p-values + plain-language verdict

## Current Status

Phase 1 (calibration) — not started. Start here.
