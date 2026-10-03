#!/usr/bin/env python3
"""Emit COLAB.md from the repo's real state.

Deliberately minimal: three steps, and step 2 is a single script that both
installs and verifies. There is no separate install cell and no separate
verify cell -- having those as two scripts is what produced the "which cell
do I run" confusion, and the old cell1_install.py carried a `numpy<2` pin
that cannot work on Python 3.13.

    python make_colab_md.py
"""

from __future__ import annotations

import pathlib

ROOT = pathlib.Path(__file__).parent
NB = ROOT / "nb"
REPO_URL = "https://github.com/12ohan/dg-prose-eval.git"

TITLE = f"""# DiffusionGemma prose eval - Colab

Source of truth: <https://github.com/12ohan/dg-prose-eval>

**Three steps.**

| # | Command | Weights |
|---|---|---|
| 1 | `!cd /content/dg && git pull --ff-only` | no |
| 2 | `!python /content/dg/nb/setup_and_verify.py` | no |
| 3 | Cell 3 block below | **yes** - ~50 GB device memory |

Step 2 installs *and* verifies in one interpreter. There is deliberately no
separate install cell: the old `cell1_install.py` pinned `numpy<2`, which has
no Python 3.13 wheels, so it could never work on Colab. It has been removed.

Re-run step 1 whenever I push; step 2 is idempotent.
"""

STEP1 = """## Step 1 - pull

If `/content/dg` does not exist yet, clone it first with the Step 0 block
at the bottom of this file. Otherwise:

```
!cd /content/dg && git pull --ff-only && git log --oneline -1
```
"""

STEP2 = """## Step 2 - install and verify

One script, one interpreter. Prints `python:` first, then each install step
as `ok`/`FAIL`, then the import check. Success ends with `diff ckpt:` and
`>>> ENVIRONMENT SOUND.`

```
!python /content/dg/nb/setup_and_verify.py
```
"""

STEP3 = """## Step 3 - generate

Needs >=50 GB of **device** memory (HBM/VRAM), not system RAM. JAX loads
parameters onto the accelerator.

| Runtime | Device mem | Works? |
|---|---|---|
| Colab TPU v5e-1 | 15.7 GiB | no |
| Colab CLI `--gpu A100` / `H100` | 80 GB | yes, one model |
| Kaggle TPU v5e-8 | 128 GB | yes, one model comfortably |

First run downloads ~50 GB from the public bucket (5-15 min) and JIT-compiles
for 1-2 min. Run once with `MODEL = "diffusion"`, then once with `"ar"`,
in separate sessions.
"""

STEP3_CODE = '''import os
import subprocess
import sys
from pathlib import Path

REPO = Path("/content/dg")
MODEL = "diffusion"                  # <- run once as "diffusion", then "ar"
OUT = REPO / "out"
OUT.mkdir(parents=True, exist_ok=True)
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

r = subprocess.run(
    [sys.executable, str(REPO / "nb" / "generate.py"),
     "--model", MODEL, "--out", str(OUT / f"{MODEL}.jsonl")],
    cwd=REPO, capture_output=True, text=True)
print(r.stdout[-4000:])
if r.returncode:
    print("STDERR:", r.stderr[-4000:])

path = OUT / f"{MODEL}.jsonl"
if path.exists():
    import json
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    ok = [x for x in rows if x.get("response")]
    mean = sum(x["metrics"].get("n_tokens", 0) for x in ok) / max(len(ok), 1)
    print(f"\\n{len(ok)}/{len(rows)} rows, mean {mean:.0f} tokens")
    print("sample:\\n", ok[0]["response"][:500] if ok else "(none)")
else:
    print("no output file")
'''

STEP0 = f"""## Step 0 - first-time clone only

```
import shutil, subprocess
from pathlib import Path
D = Path("/content/dg")
if D.exists():
    shutil.rmtree(D)
subprocess.run(["git", "clone", "--depth", "1", "{REPO_URL}", str(D)], check=True)
print(subprocess.run(["git", "-C", str(D), "log", "--oneline", "-1"],
                     capture_output=True, text=True).stdout)
```
"""

NOTES = """---

## What broke along the way

1. **PyPI `gemma` 4.0.1 has no `diffusion` module.** Google's own Colab
   example assumes `pip install gemma` then `from gemma import diffusion`.
   That does not work against the current PyPI release. Git `main` is 4.1.0
   and has it, so `setup_and_verify.py` installs from git.

2. **`tensorflow-cpu` was removed from PyPI**, but `gemma` and `kauldron` both
   still hard-pin it, so pip cannot resolve the install at all. Fixed with a
   three-line stub package that forwards to `tensorflow`, the real package now.

3. **kauldron uses `np.float128`**, removed in numpy 2. Pinning `numpy<2` was
   the first attempt and it fails on Python 3.13 (Colab's version): numpy<2
   has no 3.13 wheels, so pip built from source for 144s and died. Current fix
   is a `sitecustomize.py` written into site-packages that restores
   `np.float128 = np.longdouble` at interpreter startup. Portable across
   Python versions and it propagates to every subprocess.

4. **`Cannot uninstall PyJWT 2.7.0`** - apt-installed with no RECORD file, so
   pip refuses to remove it. Handled with `--break-system-packages` plus
   `--ignore-installed`.

## Verified against installed source, not docs

- `diffusion.ChatSampler` subclasses `gm.text.ChatSampler` directly, so the
  "drop-in replacement" claim is structural, not conventional.
- All 7 `diffusion.ChatSampler` kwargs used by `generate.py` are valid:
  `model`, `params`, `tokenizer`, `pad_length`, `max_out_length` inherited;
  `canvas_length`, `max_denoising_steps` are its own fields.
- All 3 `gm.text.ChatSampler` kwargs are valid: `model`, `params`, `multi_turn`.
- `gm.ckpts.load_params` accepts `restore_concurrent_gb`, `text_only` and
  `quantize`. `text_only` / `quantize` are the lever for later if both towers
  ever need to be resident at once.

## Local analysis (your Mac)

With both `out/diffusion.jsonl` and `out/ar.jsonl`:

```
python nb/analyze.py out/diffusion.jsonl out/ar.jsonl
```

Paired comparison, bootstrap 95% CIs, per prompt category. Then with
`GOOGLE_API_KEY` set:

```
python nb/judge.py out/diffusion.jsonl out/ar.jsonl
```

Blind pairwise judging, each pair judged twice with order swapped; only
order-invariant verdicts count.

## Decision rule

If diffusion wins on `rep_4` / `content_overuse` on the `repetition` category
with a CI excluding zero, the hypothesis holds and the two-tower work is worth
doing. If it wins nowhere or loses, that is a month saved.
"""


def main() -> None:
    md = "\n".join([
        TITLE,
        "---\n", STEP1, "```\n" + STEP1.split("```")[1].strip() + "\n```\n",
        "---\n", STEP2, "```\n" + STEP2.split("```")[1].strip() + "\n```\n",
        "---\n", STEP3, "```python\n" + STEP3_CODE.strip() + "\n```\n",
        "---\n", "## Step 0 - first-time clone only\n",
        "```python\n" + STEP0.split("```")[1].strip().split("\n", 1)[1].strip()
        + "\n```\n",
        NOTES,
    ])
    (ROOT / "COLAB.md").write_text(md)

    nf = md.count("```")
    assert nf % 2 == 0, f"unbalanced fences: {nf}"
    for s in ("Step 0", "Step 1", "Step 2", "Step 3"):
        assert s in md, f"missing {s}"
    print(f"wrote COLAB.md ({len(md):,} bytes, {nf // 2} blocks)")


if __name__ == "__main__":
    main()
