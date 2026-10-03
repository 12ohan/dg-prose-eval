"""Compare the two arms. Runs locally on your Mac, no TPU needed.

    python analyze.py ../out/diffusion.jsonl ../out/ar.jsonl

Reports per-category deltas with paired bootstrap confidence intervals.
The CIs matter: with ~29 prompts a raw mean difference is easy to
over-read. The published DiffusionGemma report reports no error bars at
all, which is one of its more legitimate criticisms -- we should not
reproduce that mistake in our own eval.
"""

from __future__ import annotations

import json
import math
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path

HEADLINE = ["rep_4", "rep_6", "distinct_3", "ttr", "hapax_frac",
            "content_overuse", "max_4gram", "sent_stdev", "n_tokens"]
# Lower is better for these; higher is better for the rest.
LOWER_IS_BETTER = {"rep_4", "rep_6", "content_overuse", "max_4gram"}
# Length is a confound to report, not a metric to win.
NEUTRAL = {"n_tokens"}


def load(path: str) -> dict[int, dict]:
    rows = {}
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                r = json.loads(line)
                rows[r["idx"]] = r
    return rows


def paired_bootstrap(diffs: list[float], iters: int = 20000, seed: int = 0):
    """95% CI on the mean of paired differences, resampling prompts."""
    if len(diffs) < 2:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    n = len(diffs)
    means = []
    for _ in range(iters):
        means.append(sum(diffs[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    lo = means[int(0.025 * iters)]
    hi = means[int(0.975 * iters)]
    return lo, hi


def fmt_delta(d: float) -> str:
    if d != d:
        return "   n/a  "
    star = ""
    return f"{d:+.4f}{star}"


def main() -> None:
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)

    a_rows = load(sys.argv[1])
    b_rows = load(sys.argv[2])
    a_name = next(iter(a_rows.values()))["model"] if a_rows else "A"
    b_name = next(iter(b_rows.values()))["model"] if b_rows else "B"

    shared = sorted(set(a_rows) & set(b_rows))
    print(f"A = {a_name}   B = {b_name}")
    print(f"paired prompts: {len(shared)}  "
          f"(A only: {sorted(set(a_rows) - set(b_rows))}, "
          f"B only: {sorted(set(b_rows) - set(a_rows))})\n")

    cats = defaultdict(list)
    for i in shared:
        cats[a_rows[i]["category"]].append(i)

    for scope, idxs in [("ALL", shared)] + sorted(cats.items()):
        idxs = [i for i in idxs if i in shared]
        if not idxs:
            continue
        print(f"=== {scope}  (n={len(idxs)}) ===")
        print(f"{'metric':16s} {a_name:>10s} {b_name:>10s} "
              f"{'delta':>10s}  {'95% CI':>20s}  win")
        for m in HEADLINE:
            va = [a_rows[i]["metrics"].get(m) for i in idxs]
            vb = [b_rows[i]["metrics"].get(m) for i in idxs]
            pairs = [(x, y) for x, y in zip(va, vb)
                     if x is not None and y is not None
                     and x == x and y == y]
            if len(pairs) < 2:
                continue
            xs = [p[0] for p in pairs]
            ys = [p[1] for p in pairs]
            diffs = [x - y for x, y in pairs]
            ma, mb = statistics.mean(xs), statistics.mean(ys)
            lo, hi = paired_bootstrap(diffs)

            sig = ""
            if m in NEUTRAL:
                sig = "  -"
            elif lo <= 0 <= hi:
                sig = "  ~"
            elif m in LOWER_IS_BETTER:
                sig = "  A" if ma < mb else "  B"
            else:
                sig = "  A" if ma > mb else "  B"
            print(f"{m:16s} {ma:10.4f} {mb:10.4f} {fmt_delta(ma - mb):>10s}"
                  f"  [{lo:+.4f},{hi:+.4f}]{sig}")
        print()

    print("legend: delta = A - B;  'A'/'B' = that side wins;  '~' = CI spans 0;  '-' = confound")
    print("CAUTION: ttr and hapax_frac are length-sensitive. If n_tokens")
    print("         differs a lot between arms, read those two with suspicion.")


if __name__ == "__main__":
    main()
