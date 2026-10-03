# DiffusionGemma prose eval - Colab cells

Source of truth: <https://github.com/12ohan/dg-prose-eval>

**Cell 0 is the only block you copy by hand.** It clones the repo, which
already contains `generate.py`, `prompts.py` and `metrics.py`. Everything
after that reads from the clone, so you never re-copy code. Re-run Cell 0 to
pick up changes I push.

| Cell | What | Weights |
|---|---|---|
| 0 | clone the repo | no |
| 1 | install + landmine workarounds | no |
| 2 | verify install, print devices | no |
| 3 | generate | **yes** - ~50 GB from `gs://gemma-data/` |

Cells 0-2 load **zero** weights and finish in a couple of minutes. Run those
first. If Cell 2 prints `diff ckpt`, the environment is sound and Cell 3 is
where the real cost starts.

---

## Cell 0 - clone

One line, re-runnable. It deletes any previous checkout first so you always
get a clean tree rather than a merge conflict.

Once the repo is public this is all you need. If you later make it private
again, either enable Colab's GitHub connector (Settings > Integrations) or
put a token in Colab Secrets named `GITHUB_TOKEN` and prefix the clone with
the authorization header shown below Cell 0's block.

```python
import shutil
import subprocess
from pathlib import Path

REPO_URL = "https://github.com/12ohan/dg-prose-eval.git"
DEST = Path("/content/dg")

if DEST.exists():
    shutil.rmtree(DEST)          # re-runnable: always fetch a clean tree

# plain subprocess rather than !git so there is no dependency on IPython
# magic variable interpolation -- this cell parses as ordinary Python
subprocess.run(["git", "clone", "--depth", "1", REPO_URL, str(DEST)], check=True)

head = subprocess.run(["git", "log", "--oneline", "-1"], cwd=DEST,
                      capture_output=True, text=True).stdout.strip()
print("cloned at", head)
print("files:", sorted(p.name for p in (DEST / "nb").iterdir() if p.suffix == ".py"))
```

---

## Cell 1 - install

Three landmines, each of which fails the install outright. Validated in a
clean venv; details at the bottom of this file.

```python
# DiffusionGemma prose eval -- install
# Three landmines, each of which fails the install outright:
#   1. PyPI `gemma` 4.0.1 has NO `diffusion` module -> install from git (4.1.0)
#   2. `tensorflow-cpu` removed from PyPI, still pinned by gemma+kauldron -> stub
#   3. kauldron imports np.float128, removed in numpy 2 -> pin numpy<2
import pathlib
import subprocess
import sys
import time


def sh(cmd):
    t = time.time()
    p = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    ok = p.returncode == 0
    print(f"[{'ok ' if ok else 'FAIL'}] {cmd[:72]}  ({time.time() - t:.0f}s)", flush=True)
    if not ok:
        print((p.stderr or p.stdout)[-1200:], flush=True)
    return ok


sh(f"{sys.executable} -m pip install -q tensorflow")

_d = pathlib.Path("/tmp/tfstub")
_d.mkdir(exist_ok=True)
(_d / "setup.py").write_text(
    "from setuptools import setup\n"
    "setup(name='tensorflow-cpu', version='2.21.0', install_requires=['tensorflow'])\n"
)
sh(f"{sys.executable} -m pip install -q {_d}")

sh(f'{sys.executable} -m pip install -q "numpy<2"')
sh(f'{sys.executable} -m pip install -q "git+https://github.com/google-deepmind/gemma.git"')
```

---

## Cell 2 - verify (weightless)

Confirms jax sees the right backend and that `gemma.diffusion` actually
imported. If `diff ckpt` prints, the install is sound. Seconds, no download.

```python
# Weightless verification: loads NO weights, so it costs seconds not 50 GB.
# If "diff ckpt" prints, the install is sound and cell 4 will work.
import subprocess
import sys

CHECK = r'''
import jax
print("jax     :", jax.__version__, "| backend", jax.default_backend())
devs = jax.devices()
print("devices :", len(devs), devs[0].device_kind)
try:
    gb = devs[0].memory_stats().get("bytes_limit", 0) / 2**30
    print(f"hbm     : {gb:.1f} GiB per device")
except Exception:
    pass

from gemma import gm, diffusion
print("diff ckpt:", diffusion.CheckpointPath.DIFFUSIONGEMMA_26B_A4B_IT.value)
print("ar   ckpt:", gm.ckpts.CheckpointPath.GEMMA4_26B_A4B_IT.value)

import dataclasses, inspect
for cls in (gm.text.ChatSampler, diffusion.ChatSampler):
    f = {x.name for x in dataclasses.fields(cls)}
    print(f"{cls.__module__}.{cls.__name__}: {len(f)} fields")
print("ChatSampler subclass of gm.text.ChatSampler:",
      issubclass(diffusion.ChatSampler, gm.text.ChatSampler))
'''

r = subprocess.run([sys.executable, "-c", CHECK], capture_output=True, text=True)
print(r.stdout.strip())
if r.returncode:
    print("STDERR:", r.stderr.strip()[-2000:])
if "diff ckpt" not in r.stdout:
    raise SystemExit("INSTALL VERIFICATION FAILED -- do not proceed to cell 4")
print("\nOK: environment is sound.")
```

---

## Cell 3 - generate

First run downloads ~50 GB from the public bucket - budget 5-15 min. The
first `sampler.chat` JIT-compiles for another 1-2 min, once.

Set `MODEL` to `"diffusion"` or `"ar"`. Run it twice, once each. Never both in
one session - two 26B models is ~100 GB.

```python
import os
import subprocess
import sys
from pathlib import Path

REPO = Path("/content/dg")           # where Cell 0 cloned
MODEL = "diffusion"                  # <- "diffusion" or "ar"
OUT = REPO / "out"
OUT.mkdir(parents=True, exist_ok=True)
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

print("loading ~50 GB of weights (first run downloads), then JIT compiling...")
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
    print(f"\n{len(ok)}/{len(rows)} rows, mean {mean:.0f} tokens")
    print("sample:\n", ok[0]["response"][:500] if ok else "(none)")
else:
    print("no output file")
```

```python
# private-repo variant, if you ever flip it back:
#   from google.colab import userdata
#   tok = userdata.get("GITHUB_TOKEN")
#   !git -c http.extraheader="AUTHORIZATION: basic $(printf 'x-access-token:%s' '{tok}' | base64)" \
#       clone --depth 1 https://github.com/12ohan/dg-prose-eval.git /content/dg
```

---

## Notes on the three landmines

1. **PyPI `gemma` 4.0.1 has no `diffusion` module.** Google's own Colab
   example assumes `!pip install -q gemma` then `from gemma import diffusion`
   - that does not work against the current PyPI release. Git `main` is
   4.1.0 and does have it, so install from git.

2. **`tensorflow-cpu` was removed from PyPI**, but `gemma` and `kauldron`
   both still hard-pin it, so pip cannot resolve the install at all. The fix
   is a three-line stub package that forwards to `tensorflow`, which is the
   real package now.

3. **kauldron imports `np.float128`**, removed in numpy 2. Hence `numpy<2`.

## Verified against installed source, not docs

- `diffusion.ChatSampler` subclasses `gm.text.ChatSampler` directly, so the
  "drop-in replacement" claim is structural, not conventional.
- All 7 `diffusion.ChatSampler` kwargs used by `generate.py` are valid:
  `model`, `params`, `tokenizer`, `pad_length`, `max_out_length` inherited;
  `canvas_length`, `max_denoising_steps` are its own fields.
- All 3 `gm.text.ChatSampler` kwargs are valid: `model`, `params`,
  `multi_turn`.
- `gm.ckpts.load_params` accepts `restore_concurrent_gb`, `text_only` and
  `quantize`. `text_only` / `quantize` are the lever for later if both
  towers ever need to be resident at once.

## Local analysis (your Mac, not Colab)

Once you have both `out/diffusion.jsonl` and `out/ar.jsonl`:

```
python nb/analyze.py out/diffusion.jsonl out/ar.jsonl
```

Paired comparison, bootstrap 95% CIs, broken out per prompt category. Then
with `GOOGLE_API_KEY` set:

```
python nb/judge.py out/diffusion.jsonl out/ar.jsonl
```

Blind pairwise judging, each pair judged twice with the order swapped; only
order-invariant verdicts are counted.

## Decision rule

If diffusion wins on `rep_4` / `content_overuse` on the `repetition` category
with a CI that excludes zero, the hypothesis holds and the two-tower work is
worth doing. If it wins nowhere or loses, that is worth a month saved.
