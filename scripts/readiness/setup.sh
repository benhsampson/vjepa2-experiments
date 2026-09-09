#!/usr/bin/env bash
# Run from any directory. Reuse the image's CUDA stack without modifying it.
set -euo pipefail
cd "$(dirname "$0")/../.."
export UV_CACHE_DIR="$PWD/.cache/uv"
export TMPDIR="$PWD/.cache/tmp"
mkdir -p "$TMPDIR" artifacts/readiness/environment
BASE_PYTHON="${BASE_PYTHON:-/usr/local/bin/python}"
"$BASE_PYTHON" -m pip list --format=json > artifacts/readiness/environment/base-packages.json
"$BASE_PYTHON" -m pip freeze > artifacts/readiness/environment/base-freeze.txt
"$BASE_PYTHON" - <<'PY'
import importlib.metadata as m
from pathlib import Path
names = {'torch', 'torchvision', 'torchaudio', 'triton', 'numpy'}
pins = sorted(f'{d.metadata["Name"]}=={d.version}' for d in m.distributions()
              if d.metadata['Name'].lower() in names or d.metadata['Name'].startswith('nvidia-'))
Path('artifacts/readiness/environment/image-constraints.txt').write_text('\n'.join(pins) + '\n')
# PyPI publishes the matching release metadata without the CUDA local-version suffix.
# PEP 440 ==2.8.0 accepts installed 2.8.0+cu128; the installation is never replaced.
base = Path('artifacts/readiness/environment/base-freeze.txt').read_text()
Path('artifacts/readiness/environment/resolve-constraints.txt').write_text(base.replace('+cu128', ''))
PY
uv venv --python "$BASE_PYTHON" --system-site-packages --allow-existing .venv
if [[ -f docs/readiness/requirements-overlay.lock ]]; then
    uv pip install --python .venv/bin/python --no-deps -r docs/readiness/requirements-overlay.lock
else
    uv pip compile requirements.txt --python .venv/bin/python \
        -c artifacts/readiness/environment/resolve-constraints.txt \
        --no-annotate --no-header --quiet -o artifacts/readiness/environment/requirements-resolved.txt
    "$BASE_PYTHON" - <<'PY'
import importlib.metadata as m
from pathlib import Path
from packaging.requirements import Requirement
resolved = Path('artifacts/readiness/environment/requirements-resolved.txt').read_text()
overlay = []
for line in resolved.splitlines():
    if not line or line.startswith('#'):
        continue
    req = Requirement(line)
    try:
        satisfied = m.version(req.name) in req.specifier
    except m.PackageNotFoundError:
        satisfied = False
    if not satisfied:
        overlay.append(line)
Path('artifacts/readiness/environment/requirements-overlay.txt').write_text('\n'.join(overlay)+'\n')
PY
    uv pip install --python .venv/bin/python --no-deps \
        -r artifacts/readiness/environment/requirements-overlay.txt
fi
.venv/bin/python scripts/readiness/repair_decord_metadata.py
# uv 0.9.0's check/freeze ignore inherited Debian dist-packages. Inspect the
# effective runtime through Python/pip instead, including system-site-packages.
.venv/bin/python -m pip check
.venv/bin/python -m pip freeze > artifacts/readiness/environment/effective-freeze.txt
.venv/bin/python - <<'PY'
import importlib, json, sys
import importlib.metadata as metadata
from pathlib import Path
from packaging.requirements import Requirement
import torch, torchvision
lock = Path('docs/readiness/requirements-runtime.lock')
if lock.exists():
    for line in lock.read_text().splitlines():
        if not line or line.startswith('#'):
            continue
        req = Requirement(line)
        assert metadata.version(req.name) in req.specifier, (
            f'{req.name} differs from the reference environment; use the isolated rebuild in the runbook')
for name in ('tensorboard', 'wandb', 'iopath', 'yaml', 'numpy', 'cv2', 'submitit',
             'braceexpand', 'webdataset', 'timm', 'transformers', 'peft', 'decord',
             'pandas', 'einops', 'beartype', 'psutil', 'h5py', 'fire', 'box',
             'skimage', 'ftfy', 'jupyter'):
    print(f'Checking import: {name}', flush=True)
    importlib.import_module(name)
assert torch.cuda.is_available()
assert (torch.ones(2, device='cuda') + 1).tolist() == [2, 2]
result = dict(python=sys.version, torch=torch.__version__, torchvision=torchvision.__version__,
              cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(),
              capability=torch.cuda.get_device_capability(), bf16=torch.cuda.is_bf16_supported(),
              vram_bytes=torch.cuda.get_device_properties(0).total_memory,
              torch_path=torch.__file__, imports='passed', cuda_computation='passed')
Path('artifacts/readiness/environment/validation.json').write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(result, indent=2))
PY
