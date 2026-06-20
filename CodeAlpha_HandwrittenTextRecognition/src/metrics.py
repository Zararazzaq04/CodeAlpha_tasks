"""Evaluation metrics for handwriting recognition."""

from __future__ import annotations

from typing import Iterable, List


def levenshtein_distance(reference: str, hypothesis: str) -> int:
    """Compute Levenshtein edit distance."""

    if reference == hypothesis:
        return 0
    if not reference:
        return len(hypothesis)
    if not hypothesis:
        return len(reference)

    previous = list(range(len(hypothesis) + 1))
    for i, ref_char in enumerate(reference, start=1):
        current = [i]
        for j, hyp_char in enumerate(hypothesis, start=1):
            insert_cost = current[j - 1] + 1
            delete_cost = previous[j] + 1
            replace_cost = previous[j - 1] + (ref_char != hyp_char)
            current.append(min(insert_cost, delete_cost, replace_cost))
        previous = current
    return previous[-1]


def character_error_rate(references: Iterable[str], hypotheses: Iterable[str]) -> float:
    """Character Error Rate: total character edits divided by characters."""

    total_distance = 0
    total_chars = 0
    for reference, hypothesis in zip(references, hypotheses):
        total_distance += levenshtein_distance(reference, hypothesis)
        total_chars += max(1, len(reference))
    return total_distance / max(1, total_chars)


def word_error_rate(references: Iterable[str], hypotheses: Iterable[str]) -> float:
    """Word Error Rate using word-level Levenshtein distance."""

    total_distance = 0
    total_words = 0
    for reference, hypothesis in zip(references, hypotheses):
        ref_words: List[str] = reference.split()
        hyp_words: List[str] = hypothesis.split()
        total_distance += _sequence_distance(ref_words, hyp_words)
        total_words += max(1, len(ref_words))
    return total_distance / max(1, total_words)


def exact_match_accuracy(references: Iterable[str], hypotheses: Iterable[str]) -> float:
    """Percentage of predictions that exactly match the target text."""

    pairs = list(zip(references, hypotheses))
    if not pairs:
        return 0.0
    matches = sum(reference == hypothesis for reference, hypothesis in pairs)
    return matches / len(pairs)


def _sequence_distance(reference: List[str], hypothesis: List[str]) -> int:
    """Levenshtein distance for token lists."""

    if reference == hypothesis:
        return 0

    previous = list(range(len(hypothesis) + 1))
    for i, ref_token in enumerate(reference, start=1):
        current = [i]
        for j, hyp_token in enumerate(hypothesis, start=1):
            insert_cost = current[j - 1] + 1
            delete_cost = previous[j] + 1
            replace_cost = previous[j - 1] + (ref_token != hyp_token)
            current.append(min(insert_cost, delete_cost, replace_cost))
        previous = current
    return previous[-1]

