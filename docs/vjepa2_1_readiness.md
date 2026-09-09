# V-JEPA 2.1 readiness runbook

This process validates Meta's original V-JEPA 2.1 ViT-G/16 at 384 pixels
(approximately 2B parameters) in this repository's native PyTorch modules. It covers
installation, strict encoder/predictor loading, 16 real training-dataset videos,
two-sample smoke testing, feature extraction, and one full video-encoder backward pass.
The final section records measured outcomes and links to the evidence.

## 1. Machine and storage

Initial source commit: `8bce8c9`. Record the current commit on each reproduction.

| Component | Observed image / machine |
| --- | --- |
| GPU | NVIDIA RTX PRO 4500 Blackwell, 32,623 MiB reported by nvidia-smi |
| Compute capability | 12.0 (`sm_120`); BF16 supported |
| NVIDIA driver | 580.159.04; nvidia-smi reports CUDA 13.0 driver capability |
| PyTorch runtime | 2.8.0+cu128, CUDA runtime 12.8; already supports sm_120 |
| torchvision / torchaudio | 0.23.0+cu128 / 2.8.0+cu128 |
| Host CPU / RAM | AMD EPYC 7443P, 48 logical CPUs, 251 GiB RAM |
| Container limits | 10.2 CPU cores of quota; 61,999,996,928 bytes RAM (57.7 GiB) |
| Python / uv | `/usr/local/bin/python`, Python 3.12.3; uv 0.9.0 |
| Workspace | `/workspace/vjepa2`, persistent network filesystem |
| Root filesystem | 30 GB total, about 28 GB initially free |

The initial 24 GB VRAM estimate was incorrect; the user chose validation using the
actual 32 GB GPU. CUDA arithmetic passed before installation. The driver's CUDA
capability and PyTorch's CUDA runtime version need not be identical.

Allow at least 40 GB of workspace storage. The checkpoint alone is
**30,238,058,912 bytes** (30.24 GB / 28.16 GiB). Keep it off the root filesystem.
The 16 videos total **21,217,438 bytes**, spanning **16 classes**. Environments,
caches, weights, videos, and generated features are ignored by Git under `.venv/`,
`.cache/`, and `artifacts/`.

```bash
cd /workspace/vjepa2
git rev-parse HEAD
nvidia-smi
free -h
df -h . /tmp
```

## 2. Install the exact dependencies, reusing the image

```bash
cd /workspace/vjepa2
bash scripts/readiness/setup.sh
```

The script creates a uv environment with `--system-site-packages` and installs the
60 pinned additions in [requirements-overlay.lock](readiness/requirements-overlay.lock).
It verifies the effective dependency versions against
[requirements-runtime.lock](readiness/requirements-runtime.lock), checks dependency
consistency, imports all direct runtime dependencies, and performs CUDA arithmetic.
It does not modify the base image or reinstall PyTorch, torchvision, torchaudio,
NumPy, or CUDA libraries. No separate CUDA toolkit, flash-attn build, or xformers
package is required: the native model uses PyTorch SDPA despite its factory name.

Evidence goes to `artifacts/readiness/environment/`: original image packages,
image constraints, effective versions, and `validation.json`. Image-specific GPU
stack pins are also in [requirements-image.lock](readiness/requirements-image.lock).
First-time imports can be slow on the network filesystem; setup prints each import.

Packaging findings incorporated into the script:

- uv 0.9.0's resolver/checker does not account for the inherited Debian
  `dist-packages` on this image. Initial resolution tried to find
  `torch==2.8.0+cu128` instead of reusing the installed build. Resolution now uses
  public release metadata (`==2.8.0`, which accepts `2.8.0+cu128` under PEP 440),
  subtracts image packages already satisfying the requirements, then installs only
  the overlay with `--no-deps`. Final checking uses Python/pip and actual imports.
  Access to the PyTorch wheel metadata host also failed DNS resolution in the first
  attempt; the final overlay installation only needs PyPI.
- The decord 0.6.0 Linux wheel filename declares `py3-none-manylinux2010_x86_64`,
  but its embedded `WHEEL` file incorrectly declares `cp36-cp36m-manylinux2010_x86_64`.
  Setup corrects only that installation metadata and its RECORD hash, retaining
  `WHEEL.original`. Decoder code and binaries are unchanged. Real-video decoding
  is separately verified on Python 3.12.

### Alternative: rebuild without the supplied image packages

On another Linux x86-64 machine with Python 3.12 and an NVIDIA driver supporting
CUDA 12.8, this installs a separate matching runtime, including PyTorch/CUDA.
It needs more storage and downloads. The measured run uses the inherited image;
this portability recipe is not a claim of testing on another machine.

```bash
cd /workspace/vjepa2
export UV_CACHE_DIR="$PWD/.cache/uv"
export TMPDIR="$PWD/.cache/tmp"
mkdir -p "$TMPDIR"
uv venv --python python3.12 --seed .venv-isolated
uv pip install --python .venv-isolated/bin/python -r docs/readiness/requirements-runtime.lock
.venv-isolated/bin/python scripts/readiness/repair_decord_metadata.py
.venv-isolated/bin/python -m pip check
```

Use `.venv-isolated/bin/python` instead of `.venv/bin/python` below. For checkpoint
installation, set `BASE_PYTHON="$PWD/.venv-isolated/bin/python"` when invoking the shell helper.

## 3. Download the original model weights

The user selected Meta's original checkpoint after learning that the available
Hugging Face ViT-G 2.1 upload was a community conversion. No Hugging Face token,
remote model code, or tensor conversion is needed.

```bash
cd /workspace/vjepa2
bash scripts/readiness/download_checkpoint.sh
```

Source: [Meta's V-JEPA 2.1 ViT-G/16 checkpoint](https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitG_384.pt).
Destination: `artifacts/readiness/checkpoints/vjepa2_1_vitG_384.pt`.

The helper downloads four concurrent 512 MiB HTTP ranges, validates their response
boundaries, retries transient errors, and records completed ranges in
`.pt.download.json`. Rerun the same command after interruption. Do not delete the
state while a partial file exists or run two downloaders concurrently. A pre-existing
contiguous curl partial file can be reused when no range state exists. The final
file must match the byte count and recorded full SHA-256.

Initially curl sustained about 24 MiB/s; the range downloader reused its first
6.01 GB instead of restarting. The downloader prints progress and the full checksum. The verified SHA-256 is
`7aae1a3c7a31d258af9c985388b5d2f20587469380f2e11b54d7876ac8cfe58a`,
also recorded in [checkpoint.sha256](readiness/checkpoint.sha256).

## 4. Download the exact 16-video subset

```bash
cd /workspace/vjepa2
.venv/bin/python scripts/readiness/dataset.py
```

The script uses [CVDF's Kinetics-400 training distribution](https://github.com/cvdfoundation/kinetics-dataset)
and the checked-in [sample manifest](readiness/kinetics16.json). It streams only
the necessary beginning of `train/part_0.tar.gz`, without retaining the whole
1.63 GB archive, and writes the clips and training annotations under
`artifacts/readiness/data/`.

The initial selection was the first 16 decodable annotated training clips with
at least 64 frames and at most two clips per class. All 16 came from different
classes; none failed decoding. The manifest contains original YouTube IDs, source
timestamps, labels, archive member names, sizes, and SHA-256s. Reruns replay those
exact entries and verify existing local files before reusing them. `paths.csv`
uses the native format: absolute path and integer label separated by a space,
without a header. Once prepared, the validation commands use local files only.

Kinetics is included in Table 1 of the
[V-JEPA 2.1 paper](https://arxiv.org/html/2603.14482v1) and the repository's
pretraining configurations. This is a sample of a documented pretraining dataset;
the release does not identify exact per-example membership in its 733K-video
Kinetics training mixture.

## 5. Strict loading and two-sample smoke test

```bash
cd /workspace/vjepa2
.venv/bin/python scripts/readiness/validate.py --mode smoke
```

This constructs `vjepa2_1_vit_gigantic_384(pretrained=False)` on CPU and strictly
loads both `target_encoder` and `predictor` from the checkpoint. Only native prefix
cleanup is applied. Missing, unexpected, or mismatched tensors fail the run. The
predictor is then released and the encoder moved to CUDA. The checkpoint is loaded
with `mmap=True, weights_only=True`, using eight CPU threads. It contains both encoder
copies, predictor, optimizer, scaler, and training metadata, explaining its much
larger disk size than the inference model.

Explicit loading bypasses the existing
`VJEPA_BASE_URL = "http://localhost:8300"` in `src/hub/backbones.py`. Its public
URL is commented out upstream; no local server or model-source change is needed.
For a loading-only diagnostic, use `--mode load`.

The smoke test processes the first two manifest clips and repeats the first
forward pass to check numerical consistency.

## 6. Run all 16 clips

```bash
cd /workspace/vjepa2
.venv/bin/python scripts/readiness/validate.py --mode inference
```

The deterministic recipe is:

- Native `VideoDataset.loadvideo_decord`, RGB, 16 frames, step 4, one temporal
  segment, no random clip sampling. The validator calls the decoder directly so
  failed samples cannot be silently replaced by the dataset's retry logic.
- Native frame indices: `[0, 4, 8, 12, 17, 21, 25, 29, 34, 38, 42, 46, 51, 55, 59, 63]`.
  The source uses clipped `linspace`; `range(0, 64, 4)` selects different frames.
- Center view from the native three-view evaluation transform: resize short side
  to 384, crop 384×384, scale pixels to [0, 1], normalize with ImageNet mean/std.
  The generic single-view helper resizes to `384 * 256 / 224`, so it is not used.
- Input `[1, 3, 16, 384, 384]`; expected features `[1, 4608, 1664]`.
- FP32 parameters, BF16 autocast, SDPA, batch size one, inference mode.

`artifacts/readiness/results/{smoke,inference}/` contains strict-load reports,
configuration and source commit, per-clip numerical statistics, synchronized
forward latency, CUDA peak allocated/reserved memory, and a summary. `pooled.pt`
holds one 1664-dimensional embedding per clip with filenames. `dense_00.pt` and
`dense_01.pt` retain the first two full token tensors in FP32 on CPU.

Features must be finite and nonconstant within and across videos. The repeated
sample must satisfy `rtol=1e-4, atol=1e-5`. This checks pipeline behavior and does
not measure classification accuracy. The one-segment/one-view workload is smaller
than the paper's multi-view benchmark; no classification head is loaded.

## 7. Check full video-encoder gradients

```bash
cd /workspace/vjepa2
.venv/bin/python scripts/readiness/validate.py --mode backward
```

This runs a separate forward/backward pass on the first clip with all encoder
parameters trainable, FP32 parameters/gradients, BF16 autocast, and native activation
checkpointing. A seeded fixed projection maps pooled features to 16 values, and
MSE against a fixed random target supplies a diagnostic gradient. No optimizer is
created and no weights are updated.

The report checks finite gradients throughout the participating encoder and
nonzero norms at the video tokenizer, first attention block, last attention block,
and final normalization. Image-only parameters and intermediate normalizations
unused by this last-layer video diagnostic are explicitly listed as expected to
have no gradient. The predictor is strictly loaded but is not trained here.

Evidence is in `artifacts/readiness/results/backward/backward.json`: loss,
forward/backward time, GPU peaks, participating parameter count, and gradient
checks. This establishes backward compatibility, not convergence or the memory
needed for Adam states, an EMA teacher, or self-supervised predictor training.

## 8. Repeat the repository and style checks

These optional developer tools are separate from the runtime overlay. Installing
with `--no-deps` preserves the inherited runtime versions; their full additional
dependency set is pinned in the developer lock.

```bash
cd /workspace/vjepa2
UV_CACHE_DIR="$PWD/.cache/uv" uv pip install --python .venv/bin/python --no-deps -r docs/readiness/requirements-dev.lock
.venv/bin/python -m pip check
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 .venv/bin/python -m pytest tests -q
.venv/bin/python -m isort scripts/readiness --check
.venv/bin/python -m black scripts/readiness --check
.venv/bin/python -m flake8 --config .flake8 scripts/readiness
sha256sum --check docs/readiness/checkpoint.sha256
```

The repository suite passed **26 tests in 63.76 seconds**, including its CUDA tests.
The 19 warnings concern deprecated upstream timm/PyTorch interfaces. isort, black,
flake8, shell syntax checks, and `git diff --check` passed for this work.

## Results

All requested validation stages passed on 2026-09-09 using the supplied image.
The native model source was unchanged. Commands executed from source commit
`661a3304d364cd3d4972cb5c4ea3ce1c5fa7dcad`; later commits add documentation, evidence,
and script formatting/import ordering. An AST comparison confirmed the validation
logic is unchanged after formatting.

| Check | Measured result |
| --- | --- |
| Final locked setup rerun | Exit 0; dependency consistency, exact versions, imports, CUDA all passed |
| Strict encoder load | 1,845,216,768 parameters; 590 tensors; no missing/unexpected keys |
| Strict predictor load | 59,433,472 parameters; 308 tensors; no missing/unexpected keys |
| Total released modules | 1,904,650,240 parameters (advertised as 2B) |
| Dataset | 16 clips, 16 classes, 21.2 MB; fresh manifest replay matched every SHA-256 |
| Inference | 16/16 passed; dense shape `[1, 4608, 1664]`; pooled shape `[16, 1664]` |
| Repeat consistency | Maximum absolute feature difference 0.0 in both inference runs |
| GPU forward time | 0.266 s/clip including first-call warmup; 0.246 s/clip excluding it |
| Inference GPU peak | 10.66 GiB allocated; 11.12 GiB reserved |
| Full encoder backward | Passed; 1,843,925,504 parameters received gradients |
| Gradient diagnostics | No unexpected missing, nonfinite, or zero-gradient tensors |
| Backward GPU peak | 14.23 GiB allocated; 16.10 GiB reserved |
| Forward + backward time | 1.392 s; diagnostic loss 2.214416 |
| Container OOM events | Zero |
| Repository test suite | 26 passed, 19 deprecation warnings |
| New-script style checks | isort, black, and flake8 passed |

Timing covers synchronized GPU forwards or forward/backward, excluding CPU model
construction, checkpoint loading, video decoding, and file writes. CUDA memory
measurements cover those same GPU workloads, including resident model parameters.
Reserved memory is allocator reservation, not an additional allocation to add to
allocated memory. This was a single-GPU, batch-one, 16-frame experiment.

Machine-readable evidence is committed in [results.json](readiness/results.json),
[environment.json](readiness/environment.json), and
[dataset-validation.json](readiness/dataset-validation.json). Large tensors, weights,
and videos remain local and can be regenerated with the numbered commands above.
Saved dense tensors were independently reloaded and checked against pooled embeddings;
all shapes, finiteness, filenames, and ordering matched exactly. A second full-file
SHA-256 check validates the retained checkpoint.

This establishes a working native-code, real-data inference and gradient pipeline
for subsequent fine-tuning experiments. It does not establish representation quality
on a target task, optimizer/EMA memory requirements, or training convergence. The
next experiment can select a task, objective, and trainable parameters using these
measured memory requirements as its baseline.
