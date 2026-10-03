"""Install + verify gemma on a Colab VM.

Fixes two things that broke the previous recipe:

1. `numpy<2` was pinned because kauldron references np.float128, which numpy
   2 removed. But numpy<2 has no wheels for Python 3.13 (Colab), so pip
   tried to build from source and died after ~2.5 min. Dropping the pin and
   instead patching the missing attribute at interpreter startup is portable
   across Python versions and costs nothing.

2. `Cannot uninstall PyJWT 2.7.0` -- Debian installs that via apt, leaving no
   RECORD file, so pip refuses to remove it. PEP 668's
   --break-system-packages plus --ignore-installed gets past it.

Run:
    !python /content/dg/nb/setup_and_verify.py
"""

import site
import subprocess
import sys
import sysconfig
import time
from pathlib import Path

PY = sys.executable
print(f"python   : {PY}")
print(f"version  : {sys.version.split()[0]}")
print("-" * 70)


def run(label, cmd, required=True, tail=2500):
    t = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True)
    ok = p.returncode == 0
    print(f"[{'ok  ' if ok else 'FAIL'}] {label}  ({time.time() - t:.0f}s)")
    if not ok:
        print((p.stdout or "")[-tail // 2:])
        print((p.stderr or "")[-tail:])
    if required and not ok:
        print(f"\n>>> STOPPED at: {label}")
        raise SystemExit(1)
    return ok


# --- 0. sitecustomize: restore np.float128 if numpy 2 dropped it -------------
# Written into site-packages so EVERY subprocess picks it up automatically,
# including the `python -c` verification below. This replaces the numpy pin.
try:
    sp = Path(sysconfig.get_paths()["purelib"])
except Exception:
    sp = Path(site.getsitepackages()[0])

shim = sp / "sitecustomize.py"
existing = shim.read_text() if shim.exists() else ""
marker = "# gemma-numpy-float128-shim"
if marker not in existing:
    shim.write_text(
        existing
        + f'\n\n{marker}\n'
        + "import numpy as _np\n"
        + "if not hasattr(_np, 'float128'):\n"
        + "    _np.float128 = _np.longdouble\n"
    )
    print(f"[ok  ] wrote numpy float128 shim -> {shim}")
else:
    print(f"[ok  ] numpy float128 shim already present")

run("shim works", [PY, "-c",
                   "import numpy as np; assert hasattr(np,'float128'); "
                   "print('np.float128 =', np.float128)"])

# --- 1. tensorflow + stub for the removed tensorflow-cpu pin ------------------
run("pip install tensorflow", [PY, "-m", "pip", "install", "-q",
                               "--break-system-packages", "tensorflow"])
stub = Path("/tmp/tfstub")
stub.mkdir(exist_ok=True)
(stub / "setup.py").write_text(
    "from setuptools import setup\n"
    "setup(name='tensorflow-cpu', version='2.21.0', "
    "install_requires=['tensorflow'])\n")
run("pip install tensorflow-cpu STUB",
    [PY, "-m", "pip", "install", "-q", "--break-system-packages", str(stub)])

# --- 2. PyJWT is apt-installed with no RECORD; pip must not try to remove it --
run("pip install PyJWT (overwrite apt copy)",
    [PY, "-m", "pip", "install", "-q", "--break-system-packages",
     "--ignore-installed", "PyJWT"])

# --- 3. gemma from git: PyPI 4.0.1 has no `diffusion` module -----------------
run("pip install gemma (from git)",
    [PY, "-m", "pip", "install", "-q", "--break-system-packages",
     "--ignore-installed", "PyJWT",
     "git+https://github.com/google-deepmind/gemma.git"])

print("-" * 70)
print("relevant installed packages:")
p = subprocess.run([PY, "-m", "pip", "list"], capture_output=True, text=True)
for line in p.stdout.splitlines():
    if line.lower().startswith(("gemma", "jax", "flax", "kauldron", "numpy",
                                "tensorflow", "orbax", "etils", "pyjwt")):
        print("   ", line)

print("-" * 70)
VERIFY = r'''
import jax, gemma
print("jax      :", jax.__version__, "| backend", jax.default_backend())
d = jax.devices()
print("devices  :", len(d), d[0].device_kind,
      f"{d[0].memory_stats().get('bytes_limit', 0)/2**30:.1f} GiB")
print("gemma    :", gemma.__file__)
from gemma import gm, diffusion
print("diff ckpt:", diffusion.CheckpointPath.DIFFUSIONGEMMA_26B_A4B_IT.value)
print("ar   ckpt:", gm.ckpts.CheckpointPath.GEMMA4_26B_A4B_IT.value)
import dataclasses
print("ChatSampler fields:", len(dataclasses.fields(gm.text.ChatSampler)),
      "ar /", len(dataclasses.fields(diffusion.ChatSampler)), "diffusion")
print("diffusion.ChatSampler subclasses gm.text.ChatSampler:",
      issubclass(diffusion.ChatSampler, gm.text.ChatSampler))
'''

p = subprocess.run([PY, "-c", VERIFY], capture_output=True, text=True)
print(p.stdout.strip())
if p.returncode:
    print("IMPORT FAILED:\n" + p.stderr.strip()[-3000:])
    raise SystemExit(1)
print("\n>>> ENVIRONMENT SOUND.")
