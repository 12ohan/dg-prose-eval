"""Install + verify in ONE interpreter, with full error output.

Cell 1 and Cell 2 previously ran as separate `python script.py` invocations.
If those resolved to different interpreters, pip would install into a
site-packages the verifier could not see -- exactly the
"ModuleNotFoundError: No module named 'gemma'" symptom. This script does
both steps under a single sys.executable and prints that path first, so a
mismatch is impossible.

Run:
    !python /content/dg/nb/setup_and_verify.py
"""

import subprocess
import sys
import time
from pathlib import Path

print(f"python   : {sys.executable}")
print(f"version  : {sys.version.split()[0]}")
print(f"platform : {sys.platform}")
print("-" * 68)


def run(label, cmd, check=True):
    t = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True)
    ok = p.returncode == 0
    print(f"[{'ok  ' if ok else 'FAIL'}] {label}  ({time.time() - t:.0f}s)")
    if not ok:
        # print generously -- truncated stderr is how failures get hidden
        print((p.stdout or "")[-1500:])
        print((p.stderr or "")[-3000:])
    if check and not ok:
        print(f"\n>>> STOPPED at: {label}")
        raise SystemExit(1)
    return ok


PY = sys.executable

# 0. prerequisites ----------------------------------------------------------
run("git present", ["git", "--version"], check=False)
run("pip present", [PY, "-m", "pip", "--version"])

# 1. tensorflow + stub for the removed tensorflow-cpu pin --------------------
run("pip install tensorflow",
    [PY, "-m", "pip", "install", "-q", "tensorflow"])
stub = Path("/tmp/tfstub")
stub.mkdir(exist_ok=True)
(stub / "setup.py").write_text(
    "from setuptools import setup\n"
    "setup(name='tensorflow-cpu', version='2.21.0', "
    "install_requires=['tensorflow'])\n"
)
run("pip install tensorflow-cpu STUB", [PY, "-m", "pip", "install", "-q", str(stub)])

# 2. numpy < 2 (kauldron imports np.float128) --------------------------------
run("pip install numpy<2", [PY, "-m", "pip", "install", "-q", "numpy<2"])

# 3. gemma from git -- PyPI 4.0.1 has no `diffusion` module -----------------
run("pip install gemma (from git)",
    [PY, "-m", "pip", "install", "-q",
     "git+https://github.com/google-deepmind/gemma.git"])

print("-" * 68)
print("what pip thinks is installed:")
p = subprocess.run([PY, "-m", "pip", "list"], capture_output=True, text=True)
for line in p.stdout.splitlines():
    if line.lower().startswith(("gemma", "jax", "flax", "kauldron", "numpy",
                                "tensorflow", "orbax", "etils")):
        print("   ", line)

print("-" * 68)
print("import check, same interpreter:")

VERIFY = r'''
import jax
print("jax     :", jax.__version__, "| backend", jax.default_backend())
d = jax.devices()
print("devices :", len(d), d[0].device_kind,
      f"{d[0].memory_stats().get('bytes_limit', 0) / 2**30:.1f} GiB")
import sys
print("gemma resolved from:", end=" ")
from gemma import gm, diffusion
import gemma
print(gemma.__file__)
print("diff ckpt:", diffusion.CheckpointPath.DIFFUSIONGEMMA_26B_A4B_IT.value)
print("ar   ckpt:", gm.ckpts.CheckpointPath.GEMMA4_26B_A4B_IT.value)
import dataclasses
print("ChatSampler fields:",
      len(dataclasses.fields(gm.text.ChatSampler)), "ar /",
      len(dataclasses.fields(diffusion.ChatSampler)), "diffusion")
'''

p = subprocess.run([PY, "-c", VERIFY], capture_output=True, text=True)
print(p.stdout.strip())
if p.returncode:
    print("IMPORT FAILED:\n" + p.stderr.strip()[-3000:])
    raise SystemExit(1)
print("\n>>> ENVIRONMENT SOUND. Cell 3 is safe to attempt.")
