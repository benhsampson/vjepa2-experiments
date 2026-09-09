# V-JEPA 2.1 readiness runbook

Work in progress: commands and measured results will be finalized after GPU validation.

The target is Meta's original V-JEPA 2.1 ViT-G/16 at 384 pixels (approximately 2B parameters),
strictly loaded into this repository's native encoder and predictor. The validation uses
16 Kinetics-400 training clips, starting with two, followed by one full video-encoder
backward pass without an optimizer step.

## Machine and decisions

- NVIDIA RTX PRO 4500 Blackwell, 32 GB VRAM (the initial 24 GB estimate was incorrect).
- AMD EPYC 7443P, 48 logical CPUs; 251 GiB system RAM.
- Python 3.12.3, uv 0.9.0; image PyTorch 2.8.0+cu128 and torchvision 0.23.0+cu128.
- CUDA computation passed before installation; reuse the image through a uv virtual
  environment with `--system-site-packages`.
- Use the official Meta download, as agreed after finding only a community conversion
  of this model on Hugging Face.
- Keep all large artifacts and caches under `/workspace/vjepa2`: the checkpoint is
  30,238,058,912 bytes, larger than the available space on the container root filesystem.

## Initial setup

From the repository root:

```bash
bash scripts/readiness/setup.sh
```

The setup records base packages, pins the existing torch/CUDA stack, installs missing
runtime requirements into `.venv`, checks dependencies/imports, and verifies CUDA.
Generated environments and artifacts are excluded from Git.

## Sources and known pitfalls

- [Official source and model list](https://github.com/facebookresearch/vjepa2).
- [Official checkpoint](https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitG_384.pt).
- [V-JEPA 2.1 paper](https://arxiv.org/html/2603.14482v1), Table 1 includes Kinetics.
- [CVDF Kinetics videos and annotations](https://github.com/cvdfoundation/kinetics-dataset).
- `src/hub/backbones.py` points pretrained downloads at localhost. Construct with
  `pretrained=False` and explicitly load the downloaded checkpoint.
- The native 2.1 Kinetics evaluation uses 16 frames, step 4, and 384-pixel inputs.
  Its three-view transform resizes the short side to 384. The generic one-view helper
  resizes to `384 * 256 / 224`, so we select the center view of the native three-view transform.
