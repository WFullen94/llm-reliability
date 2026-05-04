"""Shared TruthfulQA loader for experiments.

Uses the multiple_choice config (mc1_targets): each question has one correct
answer and several incorrect ones. We pick the correct + 3 random incorrect to
form a 4-option A/B/C/D question, matching the MMLU format.
"""

from __future__ import annotations
import random

_IDX_TO_LETTER = {0: "A", 1: "B", 2: "C", 3: "D"}


def load(n: int = 100, seed: int = 42) -> tuple[list[str], list[str]]:
    """Return (questions, labels) from TruthfulQA MC1.

    Questions formatted as A/B/C/D multiple choice.
    Labels are the correct letter (A/B/C/D).
    """
    from datasets import load_dataset

    ds = load_dataset("truthful_qa", "multiple_choice", split="validation", trust_remote_code=False)
    rows = list(ds)
    rng = random.Random(seed)
    rng.shuffle(rows)
    rows = rows[:n]

    questions, labels = [], []
    for row in rows:
        choices_all = row["mc1_targets"]["choices"]
        labels_all = row["mc1_targets"]["labels"]  # 1=correct, 0=incorrect

        # Find correct answer index
        correct_idx = next(i for i, l in enumerate(labels_all) if l == 1)
        correct_text = choices_all[correct_idx]

        # Sample 3 distractors
        distractors = [c for i, c in enumerate(choices_all) if i != correct_idx]
        rng.shuffle(distractors)
        distractors = distractors[:3]

        # Build 4-option list with correct answer at random position
        options = distractors[:3]
        correct_pos = rng.randint(0, 3)
        options.insert(correct_pos, correct_text)
        options = options[:4]

        # Pad if fewer than 4 options available
        while len(options) < 4:
            options.append(options[-1])

        choices_str = "\n".join(f"{_IDX_TO_LETTER[i]}) {c}" for i, c in enumerate(options))
        q = f"{row['question']}\n{choices_str}\nAnswer with only the letter A, B, C, or D."
        questions.append(q)
        labels.append(_IDX_TO_LETTER[correct_pos])

    return questions, labels
