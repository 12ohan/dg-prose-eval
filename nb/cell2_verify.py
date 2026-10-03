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
