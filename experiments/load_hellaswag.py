"""Shared HellaSwag loader for experiments.

HellaSwag tests commonsense NLI: given a short activity context, pick which of
4 sentence continuations is most plausible. Harder than MMLU for language models
because it tests world-knowledge + narrative coherence rather than factual recall.
"""

from __future__ import annotations
import random

_IDX_TO_LETTER = {0: "A", 1: "B", 2: "C", 3: "D"}


def load(n: int = 100, seed: int = 42) -> tuple[list[str], list[str]]:
    """Return (questions, labels) from HellaSwag validation set.

    Questions formatted as A/B/C/D multiple choice.
    Labels are the correct letter (A/B/C/D).
    """
    from datasets import load_dataset

    ds = load_dataset("Rowan/hellaswag", split="validation", trust_remote_code=False)
    rows = list(ds)
    rng = random.Random(seed)
    rng.shuffle(rows)
    rows = rows[:n]

    questions, labels = [], []
    for row in rows:
        ctx = row["ctx"].strip()
        activity = row["activity_label"]
        endings = row["endings"][:4]
        correct = int(row["label"])

        # Pad if somehow fewer than 4 endings
        while len(endings) < 4:
            endings.append(endings[-1])

        choices_str = "\n".join(f"{_IDX_TO_LETTER[i]}) {e}" for i, e in enumerate(endings))
        q = (
            f"Activity: {activity}\n"
            f"Context: {ctx}\n"
            f"Which ending is most plausible?\n"
            f"{choices_str}\n"
            "Answer with only the letter A, B, C, or D."
        )
        questions.append(q)
        labels.append(_IDX_TO_LETTER[correct])

    return questions, labels
