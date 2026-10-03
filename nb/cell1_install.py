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
