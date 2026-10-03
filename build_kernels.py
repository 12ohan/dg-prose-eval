#!/usr/bin/env python3
"""Build the two Kaggle kernel directories from the shared nb/ sources.

Two kernels rather than one: each arm loads ~50 GB of bf16 weights, and
failing to free JAX buffers between arms inside a single kernel is an easy
way to OOM on a 128 GB TPU. Separate kernels also isolate failures -- if the
diffusion arm breaks on a version conflict, the AR arm is untouched.

Usage:  python build_kernels.py
Then:   kaggle kernels push -p kernels/diffusion
        kaggle kernels push -p kernels/ar
"""

from __future__ import annotations

import json
import pathlib
import shutil

ROOT = pathlib.Path(__file__).parent
NB_SRC = ROOT / "nb"
OUT = ROOT / "kernels"

CELLS = {
    "install": """# --- install -------------------------------------------------------------
# Validated end-to-end in a clean venv 2026-10-03. Three landmines, each of
# which would have burned a whole TPU slot:
#   1. PyPI `gemma` 4.0.1 has NO `diffusion` module -> must install from git
#      (git main is 4.1.0 and does have it).
#   2. `tensorflow-cpu` was REMOVED from PyPI, but gemma and kauldron both
#      still pin it -> pip cannot resolve -> install a stub that forwards to
#      `tensorflow`.
#   3. kauldron imports np.float128, removed in numpy 2 -> pin numpy<2.
import pathlib, subprocess, sys, time

def sh(cmd):
    t = time.time()
    p = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    ok = p.returncode == 0
    print(f"[{'ok ' if ok else 'FAIL'}] {cmd[:72]}  ({{time.time()-t:.0f}}s)", flush=True)
    if not ok:
        print((p.stderr or p.stdout)[-1200:], flush=True)
    return ok

sh(f"{{sys.executable}} -m pip install -q tensorflow")
_d = pathlib.Path("/tmp/tfstub"); _d.mkdir(exist_ok=True)
(_d / "setup.py").write_text(
    "from setuptools import setup\\n"
    "setup(name='tensorflow-cpu', version='2.21.0', install_requires=['tensorflow'])\\n")
sh(f"{{sys.executable}} -m pip install -q {{_d}}")
sh(f'{{sys.executable}} -m pip install -q "numpy<2"')
sh(f'{{sys.executable}} -m pip install -q "git+https://github.com/google-deepmind/gemma.git"')

# Weightless verification: no weights downloaded, catches a broken install
# before we spend 50 GB of download and TPU time.
_r = subprocess.run([sys.executable, "-c",
    "import jax;print('jax',jax.__version__,jax.default_backend(),"
    "len(jax.devices()),jax.devices()[0].device_kind);"
    "from gemma import gm,diffusion;"
    "print('diff ckpt:',diffusion.CheckpointPath.DIFFUSIONGEMMA_26B_A4B_IT.value);"
    "print('ar   ckpt:',gm.ckpts.CheckpointPath.GEMMA4_26B_A4B_IT.value)"],
    capture_output=True, text=True)
print("VERIFY:", _r.stdout.strip() or _r.stderr.strip()[-1500:], flush=True)
if "diff ckpt" not in _r.stdout:
    raise SystemExit("install verification FAILED - do not proceed to weights")
{devices_cell}""",

    "run": """# --- generate ------------------------------------------------------------
# One model resident at a time. Weights stream from public gs://gemma-data/.
import os, sys, pathlib
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
sys.path.insert(0, "/kaggle/working/src")
!python /kaggle/working/src/generate.py --model @@MODEL@@ --out /kaggle/working/out/@@MODEL@@.jsonl
""",

    "package": """# --- package output ------------------------------------------------------
# Only the JSONL comes back; the ~50 GB of weights stay on the TPU.
import pathlib, json, shutil
src = pathlib.Path("/kaggle/working/out/@@MODEL@@.jsonl")
dst = pathlib.Path("/kaggle/working/@@MODEL@@.jsonl")
if src.exists():
    shutil.copy(src, dst)
    rows = [json.loads(l) for l in src.read_text().splitlines() if l.strip()]
    ok = [r for r in rows if r.get("response")]
    print(f"rows={len(rows)}  non_empty={len(ok)}  "
          f"mean_tokens={sum(r['metrics'].get('n_tokens',0) for r in ok)/max(len(ok),1):.0f}")
else:
    print("NO OUTPUT FILE")
print("done")
""",
}


def as_lines(text: str) -> list[str]:
    """nbformat wants source as a list of lines with trailing newlines."""
    return text.splitlines(keepends=True)


def nb(model: str) -> dict:
    devices = (
        "print('devices:', jax.devices())\n"
        "for d in jax.devices():\n"
        "    try: print('  ', d.device_kind, round("
        "d.memory_stats().get('bytes_limit',0)/2**30, 1), 'GiB')\n"
        "    except Exception: pass"
    )
    srcs = [
        CELLS["install"].replace("@@DEVICES@@", devices),
        CELLS["run"].replace("@@MODEL@@", model),
        CELLS["package"].replace("@@MODEL@@", model),
    ]
    return {
        "cells": [
            {"cell_type": "markdown",
             "metadata": {},
             "source": as_lines(
                 f"# DiffusionGemma prose eval - `{model}` arm\n"
                 "\n"
                 "Downloads weights from public `gs://gemma-data/`. "
                 "Nothing is uploaded to Kaggle.\n")},
            *[
                {"cell_type": "code", "execution_count": None,
                 "metadata": {}, "outputs": [], "source": as_lines(s)}
                for s in srcs
            ],
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python",
                           "name": "python3"},
            "accelerator": "TPU VM",
            "colab": {"name": "dg-prose-eval", "provenance": []},
            "kaggle": {"acceleratorType": "tpu", "isInternetEnabled": True},
        },
        "nbformat": 4,
        "nbformat_minor": 0,
    }


def main() -> None:
    for model in ("diffusion", "ar"):
        d = OUT / model
        if d.exists():
            shutil.rmtree(d)
        (d / "src").mkdir(parents=True)
        for f in ("prompts.py", "metrics.py", "generate.py"):
            shutil.copy(NB_SRC / f, d / "src" / f)
        (d / f"dg-prose-{model}.ipynb").write_text(json.dumps(nb(model), indent=1))
        (d / "kernel-metadata.json").write_text(json.dumps({
            # title must slugify to the id suffix or kaggle rejects/misroutes
            "id": f"m12ohan/dg-prose-{model}",
            "title": f"Dg Prose {model.capitalize()}",
            "code_file": f"dg-prose-{model}.ipynb",
            "language": "python",
            "kernel_type": "notebook",
            "is_private": True,
            "enable_gpu": False,
            "enable_tpu": True,
            "tpu_vm": False,
            "dataset_sources": [],
        }, indent=2))
        print(f"built {d}")


if __name__ == "__main__":
    main()
