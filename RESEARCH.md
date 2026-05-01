# Research Directions

Open research questions that emerged from building llm-reliability. Each one could become
a workshop paper, arXiv preprint, or full venue submission. Status tracked here as the
implementation matures and produces data.

---

## 1. Meta-Uncertainty: When Black-Box Methods Disagree

**Hypothesis:** When multiple black-box uncertainty methods (semantic entropy,
SelfCheckGPT, P(True), verbalized confidence) disagree on the same question, that
disagreement is itself a reliable signal — predictive of model errors in ways no single
method is. The cross-method variance is a form of *epistemic uncertainty about the
uncertainty estimate*.

**Novel claim:** Nobody has studied cross-method agreement as a formal signal. Individual
papers compare methods on accuracy, not on what their disagreement means.

**What we need to build:**
- Phase 2: all four black-box calibration methods implemented
- Phase 3: model adapters (OpenAI, Anthropic, MLX) to run against real models
- Analysis: for a fixed test set, compute all four method scores per question; measure
  cross-method variance; check if high variance correlates with actual model errors

**Potential finding:** A meta-uncertainty score (variance across methods) outperforms any
single method at predicting real errors, especially in the hard middle confidence range
(40–70%).

**Target venue:** NeurIPS Workshop on Uncertainty in AI, or arXiv preprint
**Status:** Blocked on Phase 2 + 3 implementation

---

## 2. Calibration Drift as a Behavioral Fingerprint

**Hypothesis:** A model's ECE curve shape (not just scalar ECE, but the per-bin
accuracy-vs-confidence profile) is a more stable and sensitive fingerprint for detecting
silent model updates than output text statistics (length, vocabulary, etc.). Two versions
of the same model will have different calibration signatures; the same version will be
consistent.

**Novel claim:** Current silent-update detection (Karimi et al. 2024) uses linguistic
features. Calibration-based fingerprinting is unexplored and likely more sensitive because
it captures *how the model is wrong*, not just *what it says*.

**What we need to build:**
- Phase 2: black-box calibration methods
- Phase 4: drift detection module with snapshot storage
- Experiment: take calibration snapshots of the same model endpoint over time (or across
  known version pairs); measure whether calibration curves distinguish versions better than
  text statistics

**Potential finding:** ECE curve shape identifies model versions with higher precision than
KS/JS tests on output distributions, and detects finer-grained behavioral changes that
text statistics miss.

**Practical implication:** A developer can run `llm-reliability fingerprint --model gpt-4o`
weekly and get a statistical alert if the model's calibration signature has shifted —
without knowing the model version string.

**Target venue:** EMNLP System Demonstrations, or empirical paper at ACL Findings
**Status:** Blocked on Phase 2 + 4 implementation

---

## 3. Semantic Stability Distance

**Hypothesis:** The minimum semantic perturbation required to flip a model's answer is a
novel, interpretable reliability metric. Models with high semantic stability distance are
robust; those that flip on near-synonym swaps are not. This metric correlates with
calibration — well-calibrated models are also more semantically stable.

**Novel claim:** Current robustness metrics measure consistency *at a fixed perturbation
level*. A distance-based metric (minimum perturbation to flip) gives a continuous,
comparable score across models and tasks. Not defined or measured in existing literature.

**What we need to build:**
- Phase 3: model adapters
- Phase 5: adversarial module with semantic perturbation generation
- Metric: binary search over perturbation magnitude to find the flip threshold per
  question; average across a test set

**Potential finding:** Semantic stability distance and ECE are correlated — a joint
reliability score combining both is more predictive than either alone.

**Target venue:** EACL Findings, EMNLP, or arXiv
**Status:** Blocked on Phase 5 implementation

---

## 4. Method Ranking Consistency: Which Black-Box Method Should You Trust?

**Hypothesis:** Different black-box calibration methods rank frontier models differently.
The rankings are inconsistent, and the inconsistency is predictable — certain methods are
better for certain task types. A practitioner currently has no principled way to choose
between them.

**Novel claim:** This is a meta-evaluation / benchmark contribution. No paper has run
all major black-box calibration methods (semantic entropy, SelfCheckGPT, P(True),
verbalized, semantic dispersion) on the same frontier models and the same tasks and asked:
do they agree? When they disagree, which one is right?

**What we need to build:**
- Phase 2 + 3: all methods implemented, all adapters working
- Experiment: run all methods on 6–8 models (GPT-4o, Claude 3.5 Sonnet, Llama-3 8B/70B,
  Mistral, Gemma 2) × 3 tasks (MMLU, TruthfulQA, HellaSwag) × all calibration methods
- Measure: Kendall's tau between method rankings; identify which method best predicts
  human-eval reliability scores

**Potential finding:** Methods diverge most on tasks requiring multi-step reasoning.
Semantic entropy is the most consistent ranker; verbalized confidence is the least — but
verbalized is the cheapest. Practitioners can use a decision rule: start with verbalized,
escalate to semantic entropy when variance is high.

**Target venue:** ACL Findings, or NeurIPS Datasets & Benchmarks
**Status:** Blocked on Phase 2 + 3 implementation. This is the most naturally produced
by building the tool — running it on real models is the experiment.

---

## Dependency map

```
Phase 2 (black-box calibration) ──► Ideas 1, 2, 4
Phase 3 (model adapters)         ──► Ideas 1, 2, 4
Phase 4 (drift detection)        ──► Idea 2
Phase 5 (adversarial)            ──► Idea 3
```

Ideas 1, 2, and 4 can all be pursued once Phases 2–3 are done. Idea 3 requires Phase 5.
The natural first paper is **Idea 4** — it's produced as a byproduct of running the tool,
requires no new experimental infrastructure beyond what we're building, and the findings
would be immediately useful to the community.
