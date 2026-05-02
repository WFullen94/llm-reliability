"""Adversarial robustness — consistency testing and perturbation-based reliability auditing.

Tests whether a model gives stable answers to semantically equivalent inputs.
Not a jailbreak tool — a reliability audit for production systems.

Methods:
  perturb()            — generate prompt variants via typo/synonym/reorder/format
  consistency_score()  — measure answer stability across perturbation types
  contradiction_probe() — detect self-contradiction via negation challenges

Research Idea 3 instrumentation: each ConsistencyResult stores flip_level —
the minimum perturbation level that changed the model's answer. The mean
flip_level across questions is the Semantic Stability Distance metric.
Higher = more robust. Comparable across models and tasks.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from typing import Callable

import numpy as np


# ---------------------------------------------------------------------------
# Perturbation types and levels
# ---------------------------------------------------------------------------

# Perturbation levels define increasing semantic distance from the original.
# Level 0 = identity (no change)
# Level 1 = surface noise (typos, spacing, capitalization)
# Level 2 = lexical substitution (synonyms, paraphrases of words)
# Level 3 = structural change (word reorder, sentence restructure)
PERTURBATION_LEVELS = {
    "typo": 1,
    "format": 1,
    "synonym": 2,
    "reorder": 3,
}

# Small synonym map for common words — avoids NLTK/WordNet dependency
_SYNONYMS: dict[str, list[str]] = {
    "what": ["which", "what exactly"],
    "how": ["in what way", "by what means"],
    "why": ["for what reason", "what is the reason"],
    "where": ["in what place", "at what location"],
    "when": ["at what time", "on what occasion"],
    "is": ["are", "was"],
    "are": ["is", "were"],
    "big": ["large", "great", "significant"],
    "small": ["little", "tiny", "minor"],
    "good": ["beneficial", "positive", "favorable"],
    "bad": ["poor", "negative", "unfavorable"],
    "true": ["correct", "accurate", "right"],
    "false": ["incorrect", "wrong", "inaccurate"],
    "fast": ["quick", "rapid", "swift"],
    "slow": ["gradual", "unhurried", "leisurely"],
    "important": ["significant", "crucial", "key"],
    "difficult": ["challenging", "hard", "complex"],
    "easy": ["simple", "straightforward", "uncomplicated"],
    "many": ["numerous", "multiple", "several"],
    "few": ["several", "some", "a number of"],
    "first": ["primary", "initial", "earliest"],
    "last": ["final", "ultimate", "concluding"],
    "best": ["optimal", "top", "finest"],
    "worst": ["poorest", "lowest", "least effective"],
    "show": ["demonstrate", "illustrate", "reveal"],
    "use": ["utilize", "employ", "apply"],
    "make": ["create", "produce", "generate"],
    "get": ["obtain", "acquire", "receive"],
    "give": ["provide", "offer", "supply"],
    "find": ["discover", "identify", "locate"],
    "know": ["understand", "recognize", "be aware"],
    "think": ["believe", "consider", "suppose"],
    "say": ["state", "mention", "indicate"],
    "call": ["name", "refer to", "term"],
}

# Keyboard adjacency for realistic typos
_KEYBOARD_NEIGHBORS: dict[str, str] = {
    "a": "sqwz", "b": "vghn", "c": "xdfv", "d": "serfcx", "e": "wrsdf",
    "f": "drtgvc", "g": "ftyhbv", "h": "gyujnb", "i": "uojkl", "j": "huikmn",
    "k": "jiolm", "l": "kop", "m": "njk", "n": "bhjm", "o": "iklp",
    "p": "ol", "q": "wa", "r": "edft", "s": "awedxz", "t": "rfgy",
    "u": "yhij", "v": "cfgb", "w": "qase", "x": "zsdc", "y": "tghu",
    "z": "asx",
}


@dataclass
class PerturbedPrompt:
    original: str
    perturbed: str
    perturbation_type: str
    level: int  # 1=surface, 2=lexical, 3=structural


@dataclass
class ConsistencyResult:
    prompt: str
    original_answer: str
    variants: list[PerturbedPrompt]
    variant_answers: list[str]
    consistency_score: float     # fraction of variants agreeing with original answer
    # Research Idea 3: minimum perturbation level that flipped the answer (None if never flipped)
    flip_level: int | None = None
    match_fn: Callable | None = field(default=None, repr=False)


@dataclass
class AdversarialResult:
    results: list[ConsistencyResult]
    overall_consistency: float
    contradiction_rate: float     # from contradiction_probe (0.0 if not run)
    # Research Idea 3: Semantic Stability Distance — mean flip_level across questions
    # None if no flips were observed (model was fully robust on this set)
    semantic_stability_distance: float | None = None
    flip_rate: float = 0.0        # fraction of questions where any flip occurred

    def report(self) -> str:
        lines = [
            "─" * 56,
            "  Adversarial Robustness Report",
            "─" * 56,
            f"  Questions tested:       {len(self.results)}",
            f"  Overall consistency:    {self.overall_consistency:.3f}",
            f"  Flip rate:              {self.flip_rate:.3f}  "
            f"({'high' if self.flip_rate > 0.3 else 'low'} sensitivity to perturbations)",
        ]
        if self.contradiction_rate > 0:
            lines.append(f"  Contradiction rate:     {self.contradiction_rate:.3f}")
        if self.semantic_stability_distance is not None:
            lines.append(
                f"  Semantic stability dist: {self.semantic_stability_distance:.2f}  "
                f"(1=surface flip, 2=lexical, 3=structural)"
            )
        lines.append("─" * 56)

        # Most vulnerable prompts
        vulnerable = sorted(self.results, key=lambda r: r.consistency_score)[:3]
        if any(r.consistency_score < 1.0 for r in vulnerable):
            lines.append("\n  Most vulnerable prompts:")
            for r in vulnerable:
                if r.consistency_score < 1.0:
                    lines.append(f'  [{r.consistency_score:.2f}] "{r.prompt[:70]}"')
                    lines.append(f'         Original: "{r.original_answer[:60]}"')
                    for v, a in zip(r.variants, r.variant_answers):
                        if not _answers_match(a, r.original_answer):
                            lines.append(
                                f'         {v.perturbation_type}: "{a[:60]}"'
                            )
                            break

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Perturbation generators
# ---------------------------------------------------------------------------

def _inject_typo(text: str, rng: random.Random, n: int = 1) -> str:
    """Inject n realistic typos into text."""
    chars = list(text)
    indices = [i for i, c in enumerate(chars) if c.isalpha()]
    if not indices:
        return text
    for _ in range(min(n, len(indices))):
        idx = rng.choice(indices)
        c = chars[idx].lower()
        op = rng.choice(["swap", "sub", "delete", "insert"])
        if op == "swap" and idx + 1 < len(chars) and chars[idx + 1].isalpha():
            chars[idx], chars[idx + 1] = chars[idx + 1], chars[idx]
        elif op == "sub" and c in _KEYBOARD_NEIGHBORS:
            replacement = rng.choice(_KEYBOARD_NEIGHBORS[c])
            chars[idx] = replacement if chars[idx].islower() else replacement.upper()
        elif op == "delete":
            chars[idx] = ""
        elif op == "insert" and c in _KEYBOARD_NEIGHBORS:
            chars.insert(idx, rng.choice(_KEYBOARD_NEIGHBORS[c]))
    return "".join(chars)


def _format_variant(text: str, rng: random.Random) -> str:
    """Apply surface-level format changes: spacing, punctuation, capitalization."""
    variants = [
        text.upper(),
        text.lower(),
        text.rstrip("?") + ".",
        text.rstrip(".") + "?",
        "  " + text + "  ",
        text.replace(",", " ,").replace(".", " ."),
    ]
    return rng.choice(variants)


def _synonym_swap(text: str, rng: random.Random) -> str:
    """Replace one content word with a synonym from the lookup table."""
    words = text.split()
    candidates = [(i, w) for i, w in enumerate(words) if w.lower().rstrip("?,. ") in _SYNONYMS]
    if not candidates:
        return text
    idx, word = rng.choice(candidates)
    clean = word.lower().rstrip("?,. ")
    replacement = rng.choice(_SYNONYMS[clean])
    # Preserve trailing punctuation
    punct = word[len(clean):]
    words[idx] = replacement + punct
    return " ".join(words)


def _reorder_words(text: str, rng: random.Random) -> str:
    """Shuffle words within each clause (split on comma/semicolon)."""
    clauses = re.split(r"([,;])", text)
    result = []
    for clause in clauses:
        if clause in (",", ";"):
            result.append(clause)
            continue
        words = clause.split()
        if len(words) <= 2:
            result.append(clause)
            continue
        # Keep first and last word, shuffle middle
        middle = words[1:-1]
        rng.shuffle(middle)
        result.append(" ".join([words[0]] + middle + [words[-1]]))
    return "".join(result)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def perturb(
    prompts: list[str],
    types: list[str] | None = None,
    n_per_type: int = 1,
    seed: int = 42,
) -> list[list[PerturbedPrompt]]:
    """Generate perturbations for each prompt.

    Args:
        prompts: Original prompt strings.
        types: Which perturbation types to apply. Defaults to all four:
               ["typo", "format", "synonym", "reorder"].
        n_per_type: How many variants to generate per type per prompt.
        seed: Random seed for reproducibility.

    Returns:
        List of lists — outer list matches prompts, inner list contains
        PerturbedPrompt objects (one per type × n_per_type).
    """
    if types is None:
        types = ["typo", "format", "synonym", "reorder"]

    rng = random.Random(seed)
    result = []

    for prompt in prompts:
        variants: list[PerturbedPrompt] = []
        for ptype in types:
            for _ in range(n_per_type):
                if ptype == "typo":
                    perturbed = _inject_typo(prompt, rng, n=rng.randint(1, 3))
                elif ptype == "format":
                    perturbed = _format_variant(prompt, rng)
                elif ptype == "synonym":
                    perturbed = _synonym_swap(prompt, rng)
                elif ptype == "reorder":
                    perturbed = _reorder_words(prompt, rng)
                else:
                    raise ValueError(f"Unknown perturbation type: {ptype!r}")
                variants.append(PerturbedPrompt(
                    original=prompt,
                    perturbed=perturbed,
                    perturbation_type=ptype,
                    level=PERTURBATION_LEVELS.get(ptype, 1),
                ))
        result.append(variants)

    return result


def _answers_match(a: str, b: str) -> bool:
    a, b = a.strip().lower(), b.strip().lower()
    return a == b or a in b or b in a


def consistency_score(
    model_fn: Callable[[str], str],
    prompts: list[str],
    types: list[str] | None = None,
    n_per_type: int = 1,
    seed: int = 42,
    match_fn: Callable[[str, str], bool] | None = None,
) -> AdversarialResult:
    """Measure how consistently a model answers semantically equivalent prompts.

    For each prompt, generates perturbations then checks whether the model's
    answer changes. Consistency score = fraction of variants that agree with
    the original answer.

    Research Idea 3: stores flip_level per question (minimum perturbation level
    that caused an answer flip). Mean flip_level = Semantic Stability Distance.

    Args:
        model_fn: Any callable (str) -> str, including ModelAdapter instances.
        prompts: Questions to test.
        types: Perturbation types. Defaults to all four.
        n_per_type: Variants per type.
        seed: Random seed.
        match_fn: How to compare answers. Defaults to case-insensitive substring.

    Returns:
        AdversarialResult with per-question consistency scores and aggregate metrics.
    """
    if match_fn is None:
        match_fn = _answers_match

    all_variants = perturb(prompts, types=types, n_per_type=n_per_type, seed=seed)
    results: list[ConsistencyResult] = []

    for prompt, variants in zip(prompts, all_variants):
        original_answer = model_fn(prompt).strip()

        variant_answers = [model_fn(v.perturbed).strip() for v in variants]
        matches = [match_fn(va, original_answer) for va in variant_answers]
        score = float(np.mean(matches)) if matches else 1.0

        # Research Idea 3: find minimum flip level
        flip_level = None
        for variant, matched in zip(variants, matches):
            if not matched:
                if flip_level is None or variant.level < flip_level:
                    flip_level = variant.level

        results.append(ConsistencyResult(
            prompt=prompt,
            original_answer=original_answer,
            variants=variants,
            variant_answers=variant_answers,
            consistency_score=score,
            flip_level=flip_level,
        ))

    overall = float(np.mean([r.consistency_score for r in results]))
    flip_levels = [r.flip_level for r in results if r.flip_level is not None]
    flip_rate = len(flip_levels) / len(results) if results else 0.0
    ssd = float(np.mean(flip_levels)) if flip_levels else None

    return AdversarialResult(
        results=results,
        overall_consistency=overall,
        contradiction_rate=0.0,
        semantic_stability_distance=ssd,
        flip_rate=flip_rate,
    )


_CONTRADICTION_TEMPLATE = """\
Consider the following statement: "{statement}"

Is the opposite of this statement true? Answer with only Yes or No."""


def contradiction_probe(
    model_fn: Callable[[str], str],
    questions: list[str],
    answers: list[str],
) -> float:
    """Measure how often a model contradicts its own answers.

    For each (question, answer) pair, challenges the model: "Is the opposite
    of [answer] true?" A model that says Yes is contradicting itself.

    Args:
        model_fn: Any callable (str) -> str.
        questions: Original questions.
        answers: The model's answers to those questions (or ground-truth answers).

    Returns:
        Contradiction rate in [0, 1] — fraction of cases where model agreed
        with the negation of its own answer.
    """
    if len(questions) != len(answers):
        raise ValueError("questions and answers must have the same length")

    contradictions = 0
    for question, answer in zip(questions, answers):
        statement = f"The answer to '{question}' is: {answer}"
        prompt = _CONTRADICTION_TEMPLATE.format(statement=statement)
        verdict = model_fn(prompt).strip().lower()
        if verdict.startswith("yes"):
            contradictions += 1

    return contradictions / len(questions) if questions else 0.0
