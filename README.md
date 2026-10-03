# Stage 0: does bidirectional block attention help prose?

Hypothesis: DiffusionGemma's bidirectional 256-token canvas reduces the
lexical repetition that AR decoding produces structurally, and helps most on
open-ended writing where there is no verifiable answer.

Both arms use `gemma`'s own samplers; `diffusion.ChatSampler` is documented
as a drop-in replacement for `gm.text.ChatSampler`, so the decoding algorithm
is the only difference.

## Run (Kaggle TPU v5e-8, or any 128GB TPU)

    !pip install -q gemma
    !python nb/generate.py --model diffusion --out out/diffusion.jsonl
    !python nb/generate.py --model ar        --out out/ar.jsonl

Run one arm per invocation -- never both resident. ~50 GB of bf16 weights each.

## Analyze (local)

    python nb/analyze.py out/diffusion.jsonl out/ar.jsonl

Objective metrics with paired bootstrap 95% CIs, per prompt category.

## Judge (local, needs GOOGLE_API_KEY)

    python nb/judge.py out/diffusion.jsonl out/ar.jsonl

Blind pairwise judging, each pair evaluated twice with order swapped;
only order-invariant verdicts are counted.

## Files

| file | role |
|---|---|
| `nb/prompts.py` | 29 prompts, 5 categories; `repetition` is the core test |
| `nb/metrics.py` | objective n-gram / diversity stats, no model calls |
| `nb/generate.py` | one-model-per-run generation, writes JSONL |
| `nb/analyze.py` | paired comparison, bootstrap CIs |
| `nb/judge.py` | blind LLM judge, gemini-3.8-flash |

## Decision rule

If diffusion wins on `rep_4` / `content_overuse` on the `repetition`
category with a CI that excludes 0, the hypothesis holds and the two-tower
work (Stage 1) is worth doing. If it wins nowhere or loses, save the month.
