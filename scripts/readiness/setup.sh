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
PY
uv venv --python "$BASE_PYTHON" --system-site-packages --allow-existing .venv
if [[ -f docs/readiness/requirements-overlay.lock ]]; then
    uv pip install --python .venv/bin/python -r docs/readiness/requirements-overlay.lock \
        -c artifacts/readiness/environment/image-constraints.txt
else
    uv pip install --python .venv/bin/python -r requirements.txt \
        -c artifacts/readiness/environment/image-constraints.txt
fi
uv pip check --python .venv/bin/python
uv pip freeze --python .venv/bin/python > artifacts/readiness/environment/effective-freeze.txt
.venv/bin/python - <<'PY'
import importlib, json, sys
import torch, torchvision
for name in ('tensorboard', 'wandb', 'iopath', 'yaml', 'numpy', 'cv2', 'submitit',
             'braceexpand', 'webdataset', 'timm', 'transformers', 'peft', 'decord',
             'pandas', 'einops', 'beartype', 'psutil', 'h5py', 'fire', 'box',
             'skimage', 'ftfy', 'jupyter'):
    importlib.import_module(name)
assert torch.cuda.is_available()
assert (torch.ones(2, device='cuda') + 1).tolist() == [2, 2]
result = dict(python=sys.version, torch=torch.__version__, torchvision=torchvision.__version__,
              cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(),
              capability=torch.cuda.get_device_capability(), bf16=torch.cuda.is_bf16_supported(),
              vram_bytes=torch.cuda.get_device_properties(0).total_memory,
              torch_path=torch.__file__, imports='passed', cuda_computation='passed')
from pathlib import Path
Path('artifacts/readiness/environment/validation.json').write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(result, indent=2))
PY
