"""Stage 0 generation: DiffusionGemma vs Gemma-4-AR on the prose prompt set.

Runs ONE model per invocation so we never hold two 26B models resident.
On a v5e-8 (128 GB) that is 50 GB of bf16 weights for one arm, comfortable.

Usage (from a Kaggle TPU notebook or Colab):

    !python generate.py --model diffusion --out ../out/diffusion.jsonl
    !python generate.py --model ar        --out ../out/ar.jsonl

Both arms use the same sampler interface -- `diffusion.ChatSampler` is
documented as a drop-in replacement for `gm.text.ChatSampler` -- so the only
difference between the arms is the decoding algorithm itself, which is
exactly the comparison we want.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import pathlib
import sys
import time

HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(HERE))

from metrics import all_metrics          # noqa: E402
from prompts import get_prompts, CATEGORIES  # noqa: E402


def report_devices() -> None:
    import jax
    devs = jax.devices()
    print(f"[env] jax {jax.__version__}  backend={jax.default_backend()}")
    print(f"[env] {len(devs)} device(s): {devs}")
    for d in devs:
        try:
            print(f"[env]   {d.device_kind}  "
                  f"{d.memory_stats().get('bytes_limit', 0) / 2**30:.1f} GiB")
        except Exception:
            pass


def build_diffusion_sampler():
    from gemma import gm, diffusion
    print("[load] DiffusionGemma 26B-A4B-it ...", flush=True)
    t0 = time.time()
    model = diffusion.DiffusionGemma_26B_A4B()
    params = gm.ckpts.load_params(
        diffusion.CheckpointPath.DIFFUSIONGEMMA_26B_A4B_IT,
        restore_concurrent_gb=16,
    )
    print(f"[load] done in {time.time() - t0:.0f}s", flush=True)

    sampler = diffusion.ChatSampler(
        model=model,
        params=params,
        tokenizer=gm.text.Gemma4Tokenizer(),
        canvas_length=256,
        max_denoising_steps=48,   # max; adaptive stopping lands near 12
        pad_length=1024,          # matches the reference example
        max_out_length=3072,
    )
    return sampler


def build_ar_sampler(size: str = "26B_A4B"):
    from gemma import gm
    print(f"[load] Gemma 4 {size}-it (AR) ...", flush=True)
    t0 = time.time()
    model = getattr(gm.nn, f"Gemma4_{size}")()
    params = gm.ckpts.load_params(
        getattr(gm.ckpts.CheckpointPath, f"GEMMA4_{size}_IT"),
        restore_concurrent_gb=16,
    )
    print(f"[load] done in {time.time() - t0:.0f}s", flush=True)

    return gm.text.ChatSampler(model=model, params=params, multi_turn=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["diffusion", "ar"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--category", default=None,
                    help="restrict to one prompt category")
    ap.add_argument("--limit", type=int, default=0, help="0 = all prompts")
    ap.add_argument("--warmup", action="store_true", default=True)
    ap.add_argument("--size", default="26B_A4B",
                    help="AR model size. E2B / E4B fit a 16 GB TPU and are "
                         "for pipeline validation only -- DiffusionGemma only "
                         "exists at 26B_A4B, so --model diffusion ignores this.")
    args = ap.parse_args()

    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    report_devices()

    if args.model == "diffusion":
        sampler = build_diffusion_sampler()
    else:
        sampler = build_ar_sampler(args.size)
        if args.size != "26B_A4B":
            print(f"[warn] --size {args.size} is a SMALLER model. Validates "
                  f"the pipeline, not the science. Real eval needs 26B_A4B.",
                  flush=True)

    items = get_prompts(args.category)
    if args.limit:
        items = items[:args.limit]
    print(f"[run] {len(items)} prompts, model={args.model}", flush=True)

    if args.warmup:
        print("[run] warmup (JIT compile, 1-2 min, one-time)...", flush=True)
        t0 = time.time()
        _ = sampler.chat("warmup")
        print(f"[run] warmup done in {time.time() - t0:.0f}s", flush=True)

    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w") as fh:
        for n, (idx, cat, prompt) in enumerate(items):
            t0 = time.time()
            try:
                text = sampler.chat(prompt)
                err = None
            except Exception as exc:  # keep going; partial results are useful
                text, err = "", f"{type(exc).__name__}: {exc}"
                print(f"[run] {idx} FAILED: {err}", flush=True)
                gc.collect()

            dt = time.time() - t0
            row = {
                "idx": idx,
                "model": args.model if args.model == "diffusion" else f"ar-{args.size}",
                "category": cat,
                "category_desc": CATEGORIES[cat],
                "prompt": prompt,
                "response": text,
                "seconds": round(dt, 2),
                "error": err,
            }
            row["metrics"] = all_metrics(text) if text else {}
            fh.write(json.dumps(row) + "\n")
            fh.flush()

            m = row["metrics"]
            print(f"[{n + 1}/{len(items)}] idx={idx} {cat:12s} "
                  f"{dt:6.1f}s  ntok={m.get('n_tokens', 0):4d}  "
                  f"rep4={m.get('rep_4', float('nan')):.4f}  "
                  f"ttr={m.get('ttr', float('nan')):.3f}", flush=True)

    print(f"[done] wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
