"""Shared MMLU loader for all experiments."""

from __future__ import annotations
import random


SUBJECTS = [
    "high_school_geography",
    "high_school_psychology",
    "high_school_computer_science",
    "high_school_biology",
    "college_biology",
    "college_computer_science",
    "world_religions",
    "prehistory",
    "sociology",
    "moral_scenarios",
]

_IDX_TO_LETTER = {0: "A", 1: "B", 2: "C", 3: "D"}


def load(n: int = 100, seed: int = 42) -> tuple[list[str], list[str]]:
    """Return (questions, labels) from a balanced MMLU sample.

    Questions are formatted as multiple-choice with A/B/C/D options.
    Labels are the correct letter (A/B/C/D).
    """
    from datasets import load_dataset

    examples: list[dict] = []
    per_subject = max(1, n // len(SUBJECTS))
    rng = random.Random(seed)

    for subj in SUBJECTS:
        try:
            ds = load_dataset("cais/mmlu", subj, split="test", trust_remote_code=False)
            rows = list(ds)
            rng.shuffle(rows)
            examples.extend(rows[:per_subject])
        except Exception:
            continue
        if len(examples) >= n:
            break

    examples = examples[:n]

    questions, labels = [], []
    for ex in examples:
        choices = "\n".join(
            f"{_IDX_TO_LETTER[i]}) {c}" for i, c in enumerate(ex["choices"][:4])
        )
        q = (
            f"{ex['question']}\n{choices}\n"
            "Answer with only the letter A, B, C, or D."
        )
        questions.append(q)
        labels.append(_IDX_TO_LETTER[ex["answer"]])

    return questions, labels
