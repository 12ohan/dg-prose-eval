#!/usr/bin/env python3
"""Watch Kaggle kernels and fire a native macOS notification on completion.

Why this exists: I am a stateless API. When a kernel run finishes, nothing
can push a message to me. So you would otherwise have to poll manually and
then remember to come back and wake me. This does the polling and surfaces
the result to you directly, so waking me stays optional and you only do it
when you actually want the analysis.

    python watch_kernels.py m12ohan/dg-prose-diffusion m12ohan/dg-prose-ar

Polls every INTERVAL seconds, downloads kernel output on terminal state,
summarises it into the notification body, and appends to watch.log.

Not a heavy job: one subprocess call per interval, negligible CPU.
Safe to leave running.
"""

from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys
import time

HERE = pathlib.Path(__file__).parent
LOG = HERE / "watch.log"
INTERVAL = int(os.environ.get("WATCH_INTERVAL", "60"))
TERMINAL = {"complete", "error", "KernelWorkerStatus.COMPLETE",
            "KernelWorkerStatus.ERROR", "KernelWorkerStatus.CANCELLED"}
DOWNLOAD_DIR = HERE / "out" / "kernels"


def log(msg: str) -> None:
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with LOG.open("a") as fh:
        fh.write(line + "\n")


def notify(title: str, body: str, sound: bool = True) -> None:
    safe_b = body.replace('"', "'")[:400]
    script = (
        f'display notification "{safe_b}" with title "{title}"'
        + (' sound name "Glass"' if sound else "")
    )
    subprocess.run(["osascript", "-e", script], check=False,
                   capture_output=True)


def kaggle(*args: str) -> tuple[int, str]:
    p = subprocess.run(["kaggle", *args], capture_output=True, text=True,
                       env={**os.environ, "PYTHONUNBUFFERED": "1"})
    return p.returncode, (p.stdout + p.stderr).strip()


def status_of(ref: str) -> str | None:
    rc, out = kaggle("kernels", "status", ref)
    if rc != 0:
        return None
    # 'm12ohan/x has status "KernelWorkerStatus.RUNNING"'
    if '"' not in out:
        return None
    return out.split('"')[1].split(".")[-1].upper()


def summarize(ref: str) -> str:
    """Pull headline stats out of the downloaded JSONL for the notification."""
    name = ref.split("/")[-1]
    arm = "ar" if name.endswith("-ar") else "diffusion"
    jl = DOWNLOAD_DIR / f"{arm}.jsonl"
    if not jl.exists():
        return "output file not found"
    try:
        rows = [json.loads(l) for l in jl.read_text().splitlines() if l.strip()]
    except Exception as exc:
        return f"could not parse output: {exc}"
    ok = [r for r in rows if r.get("response")]
    errs = [r for r in rows if r.get("error")]
    if not ok:
        return f"0 usable rows of {len(rows)}; {len(errs)} errors"
    mean_tok = sum(r["metrics"].get("n_tokens", 0) for r in ok) / len(ok)
    mean_rep4 = sum(r["metrics"].get("rep_4", 0) for r in ok) / len(ok)
    secs = sum(r.get("seconds", 0) for r in ok) / len(ok)
    return (f"{len(ok)}/{len(rows)} rows  mean_tok={mean_tok:.0f}  "
            f"rep_4={mean_rep4:.4f}  {secs:.1f}s/gen"
            + (f"  ({len(errs)} errors)" if errs else ""))


def main() -> None:
    refs = sys.argv[1:]
    if not refs:
        print(__doc__)
        sys.exit(1)
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    log(f"watching: {', '.join(refs)}  (every {INTERVAL}s, log -> {LOG})")

    pending = {r: None for r in refs}
    while pending:
        for ref in list(pending):
            st = status_of(ref)
            if st is None:
                log(f"{ref}: status query failed")
                continue
            if pending[ref] is None:
                pending[ref] = st
                log(f"{ref}: {st}")
                continue
            if st != pending[ref]:
                log(f"{ref}: {pending[ref]} -> {st}")
                pending[ref] = st
            if st in {"COMPLETE", "ERROR", "CANCELLED"}:
                del pending[ref]
                if st == "COMPLETE":
                    rc, out = kaggle("kernels", "output", ref, "-p",
                                     str(DOWNLOAD_DIR))
                    log(f"{ref}: output download rc={rc} {out[:120]}")
                    body = summarize(ref)
                    notify(f"Kaggle done: {ref.split('/')[-1]}", body)
                    log(f"{ref}: {body}")
                else:
                    rc, out = kaggle("kernels", "output", ref, "-p",
                                     str(DOWNLOAD_DIR))
                    log(f"{ref}: FAILED {st}; logs rc={rc}")
                    notify(f"Kaggle {st}: {ref.split('/')[-1]}",
                           f"status={st}. Check kernel logs in the Kaggle UI.")
        if pending:
            time.sleep(INTERVAL)

    log("all watched kernels reached a terminal state")


if __name__ == "__main__":
    main()
