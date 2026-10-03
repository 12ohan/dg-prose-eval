"""Objective prose-quality metrics.

Computed locally, no model calls. These are the deterministic half of the
eval; the LLM judge (judge.py) is the subjective half.

The headline metric is `rep_4`, because it is the one that most directly
tests the hypothesis motivating this whole project: that bidirectional block
attention reduces the lexical repetition that autoregressive decoding
produces structurally.

Why repetition is the right target
----------------------------------
AR decoding picks token i conditioned on tokens <i. If a word or phrase has
already appeared, that fact is in the context and raises its probability of
reappearing. There is no mechanism to notice and avoid it. This is the
repetition failure mode every AR LM shows and no amount of decoding tricks
removes.

A bidirectional canvas of 256 tokens sees the whole window at once, so it can
observe that "deliberately" already appears three times in the current block
and choose otherwise. If that benefit is real, it should show up as a lower
`rep_4` on the repetition-category prompts specifically, and should show up
MORE there than on fiction prompts.
"""

from __future__ import annotations

import re
from collections import Counter

_WORD = re.compile(r"[a-z0-9']+")


def _tokens(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def distinct_n(tokens: list[str], n: int) -> float:
    """Fraction of n-grams in `tokens` that are unique. Higher = richer."""
    if len(tokens) < n:
        return float("nan")
    grams = [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
    return len(set(grams)) / len(grams)


def repetition_rate(tokens: list[str], n: int) -> float:
    """Fraction of n-grams that are repeats. Lower = less repetitive.

    Inverse of distinct_n, kept separate because it reads more naturally
    when the claim is "this model repeats itself less".
    """
    d = distinct_n(tokens, n)
    return float("nan") if d != d else 1.0 - d


def type_token_ratio(tokens: list[str]) -> float:
    """Unique words / total words. Length-sensitive; compare only within
    matched output lengths."""
    if not tokens:
        return float("nan")
    return len(set(tokens)) / len(tokens)


def content_word_repetition(tokens: list[str], top_k: int = 20) -> float:
    """Fraction of the most common content words (length >= 4) that occur
    3+ times. Targets the specific failure of overusing a single term."""
    content = [t for t in tokens if len(t) >= 4]
    if not content:
        return float("nan")
    counts = Counter(content)
    overused = sum(1 for _, n in counts.most_common(top_k) if n >= 3)
    return overused / top_k


def sentence_length_stats(text: str) -> dict:
    """Mean and stdev of sentence lengths in words.

    Very low stdev is a known tell of degraded generation: the model settles
    into a uniform rhythm regardless of content. Informative as a
    corroborating signal, not a primary metric.
    """
    sents = [s for s in re.split(r"[.!?]+", text) if len(_tokens(s)) > 2]
    if len(sents) < 2:
        return {"sent_mean": float("nan"), "sent_stdev": float("nan"), "n_sent": len(sents)}
    lens = [len(_tokens(s)) for s in sents]
    mean = sum(lens) / len(lens)
    var = sum((x - mean) ** 2 for x in lens) / len(lens)
    return {"sent_mean": mean, "sent_stdev": var ** 0.5, "n_sent": len(sents)}


def lexical_sophistication(tokens: list[str]) -> float:
    """Mean unigram frequency proxy: fraction of tokens that appear exactly
    once in the response. Higher = less reliance on a small set of words."""
    if not tokens:
        return float("nan")
    counts = Counter(tokens)
    return sum(1 for t in tokens if counts[t] == 1) / len(tokens)


def longest_repeated_span(tokens: list[str], n: int = 4) -> int:
    """Longest run of consecutive n-grams that recur. Catches verbatim
    looping, which the aggregate rep_n can mask if it is diluted."""
    if len(tokens) < n * 2:
        return 0
    grams = [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
    counts = Counter(grams)
    if not counts:
        return 0
    return max(counts.values())


def all_metrics(text: str) -> dict:
    toks = _tokens(text)
    m = {
        "n_tokens": len(toks),
        "distinct_2": distinct_n(toks, 2),
        "distinct_3": distinct_n(toks, 3),
        "rep_4": repetition_rate(toks, 4),
        "rep_6": repetition_rate(toks, 6),
        "ttr": type_token_ratio(toks),
        "hapax_frac": lexical_sophistication(toks),
        "content_overuse": content_word_repetition(toks),
        "max_4gram": longest_repeated_span(toks, 4),
    }
    m.update(sentence_length_stats(text))
    return m


if __name__ == "__main__":
    good = ("The kelp held its shape against the current, each blade bending "
            "and returning with a rhythm that suggested patience rather than "
            "resistance. Sunlight came through in shafts that moved slowly "
            "enough to watch.")
    bad = ("The beautiful amazing wonderful kelp was very nice. The beautiful "
           "amazing kelp was very nice and good. It was the most beautiful "
           "thing. Very nice beautiful good amazing wonderful.")

    for name, txt in (("clean", good), ("repetitive", bad)):
        m = all_metrics(txt)
        print(f"--- {name} ---")
        for k in ("n_tokens", "distinct_3", "rep_4", "ttr", "hapax_frac",
                  "content_overuse", "max_4gram", "sent_stdev"):
            print(f"  {k:16s} {m[k]:.4f}")
