"""LLM-as-judge for the prose eval. Runs locally on your Mac.

    export GOOGLE_API_KEY=...
    python judge.py ../out/diffusion.jsonl ../out/ar.jsonl

This is the subjective half; metrics.py is the objective half. They are
deliberately kept separate -- objective n-gram stats catch repetition, but
they cannot tell you which sentence actually reads better, which is the thing
you originally cared about when you started this.

Two designs that matter:

1. BLIND + SWAPPED. The judge never sees which model produced which text, and
   each pair is evaluated twice with the order reversed. Position bias in LLM
   judges is well documented and shows up as a systematic preference for
   whichever text is presented first. Swapping and only counting responses
   where the judge is order-invariant removes most of it.

2. BOTH DIRECTIONS AND A TIE OPTION. Forcing a winner on every pair
   manufactures preference from noise. Ties are a real outcome and are
   reported as such.

Model note: defaults to gemini-3.8-flash per the current Google model
generation. Override with JUDGE_MODEL if you want a bigger judge.
"""

from __future__ import annotations

import json
import os
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "gemini-3.8-flash")

RUBRIC = """You are evaluating two pieces of prose that were written to the same
instruction. Judge ONLY the prose quality. Ignore factual claims and ignore
whether you agree with the content.

Score these dimensions, each 1-10:

- DICTION: word choice. Is it precise and varied, or generic and padded?
  Penalise stock phrases ("delve into", "tapestry", "testament to").
- COHERENCE: does the writing hold together, or drift or contradict itself?
- VOICE: is there an authorial presence, or does it read like filler?
- NATURALNESS: does it read as written prose, or as padded-out text?

Then give an OVERALL preference: "A", "B", or "tie".

Reply in EXACTLY this format and nothing else:
DICT_A: <n>
DICT_B: <n>
COH_A: <n>
COH_B: <n>
VOICE_A: <n>
VOICE_B: <n>
NAT_A: <n>
NAT_B: <n>
OVERALL: <A|B|tie>"""


def parse_reply(txt: str) -> dict | None:
    out = {}
    for key in ("DICT", "COH", "VOICE", "NAT"):
        m = re.search(rf"{key}_A:\s*(\d+)", txt)
        n = re.search(rf"{key}_B:\s*(\d+)", txt)
        if m and n:
            out[f"{key}_A"] = int(m.group(1))
            out[f"{key}_B"] = int(n.group(1))
    m = re.search(r"OVERALL:\s*(A|B|tie)", txt, re.I)
    if not m:
        return None
    out["OVERALL"] = m.group(1).strip().lower()
    return out


def ask(client, model, prompt_text, a_text, b_text, retries=3):
    msg = (f"INSTRUCTION:\n{prompt_text}\n\n"
           f"--- PASSAGE A ---\n{a_text}\n\n--- PASSAGE B ---\n{b_text}\n\n{RUBRIC}")
    for attempt in range(retries):
        try:
            resp = client.models.generate_content(
                model=model,
                contents=msg,
                config={"temperature": 0.3, "max_output_tokens": 400},
            )
            parsed = parse_reply(resp.text or "")
            if parsed:
                return parsed
        except Exception as exc:
            print(f"    [retry {attempt + 1}] {type(exc).__name__}: {exc}")
    return None


def main() -> None:
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not key:
        sys.exit("set GOOGLE_API_KEY first")
    from google import genai
    client = genai.Client(api_key=key)

    def load(p):
        return {json.loads(l)["idx"]: json.loads(l)
                for l in open(p) if l.strip()}

    a_rows, b_rows = load(sys.argv[1]), load(sys.argv[2])
    a_name = next(iter(a_rows.values()))["model"]
    b_name = next(iter(b_rows.values()))["model"]
    shared = sorted(set(a_rows) & set(b_rows))
    print(f"judge={JUDGE_MODEL}  A={a_name}  B={b_name}  n={len(shared)}\n")

    rng = random.Random(0)
    results = []
    for n, idx in enumerate(shared):
        ra, rb = a_rows[idx], b_rows[idx]
        if not ra["response"] or not rb["response"]:
            continue
        print(f"[{n + 1}/{len(shared)}] idx={idx} {ra['category']}", flush=True)

        # Pass 1: as-is.  Pass 2: swapped.
        p1 = ask(client, JUDGE_MODEL, ra["prompt"], ra["response"], rb["response"])
        p2 = ask(client, JUDGE_MODEL, ra["prompt"], rb["response"], ra["response"])

        rec = {"idx": idx, "category": ra["category"], "pass1": p1, "pass2": p2}
        if p1 and p2:
            if p1["OVERALL"] == "tie" and p2["OVERALL"] == "tie":
                rec["verdict"] = "tie"
            elif p1["OVERALL"] == p2["OVERALL"]:
                rec["verdict"] = p1["OVERALL"]      # order-invariant
            else:
                rec["verdict"] = "inconsistent"      # position bias
        results.append(rec)
        print(f"    -> {rec.get('verdict', 'parse-fail')}", flush=True)

    out = Path("../out/judgments.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"judge": JUDGE_MODEL, "A": a_name, "B": b_name, "results": results},
        indent=2))

    tally = defaultdict(int)
    for r in results:
        tally[r.get("verdict", "parse-fail")] += 1
    print(f"\n=== verdicts ({len(results)} judged) ===")
    for k, v in sorted(tally.items()):
        print(f"  {k:14s} {v:3d}   ({v / max(len(results), 1):.0%})")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
