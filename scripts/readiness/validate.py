"""Strict native checkpoint loading, real-video inference, and an encoder backward check."""
import argparse
import gc
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint

from evals.video_classification_frozen.utils import make_transforms
from src.datasets.video_dataset import VideoDataset
from src.hub.backbones import _clean_backbone_key, vjepa2_1_vit_gigantic_384


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def memory():
    return dict(peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                peak_reserved_bytes=torch.cuda.max_memory_reserved())


def load_encoder(checkpoint, output):
    start = time.perf_counter()
    print("Constructing native ViT-G encoder and predictor on CPU", flush=True)
    encoder, predictor = vjepa2_1_vit_gigantic_384(pretrained=False)
    print("Reading official checkpoint with mmap", flush=True)
    state = torch.load(checkpoint, map_location="cpu", mmap=True, weights_only=True)
    report = dict(checkpoint=str(checkpoint.relative_to(ROOT)), checkpoint_keys=list(state),
                  factory="src.hub.backbones.vjepa2_1_vit_gigantic_384", strict=True)
    for name, key, model in (("encoder", "target_encoder", encoder), ("predictor", "predictor", predictor)):
        weights = _clean_backbone_key(dict(state[key]))
        result = model.load_state_dict(weights, strict=True)
        assert not result.missing_keys and not result.unexpected_keys
        report[name] = dict(parameters=sum(p.numel() for p in model.parameters()),
                            state_tensors=len(weights), missing_keys=result.missing_keys,
                            unexpected_keys=result.unexpected_keys)
        print(f"Strict {name} load passed: {report[name]}", flush=True)
        del weights
    del state, predictor
    gc.collect()
    report["seconds"] = time.perf_counter() - start
    save_json(output / "loading.json", report)
    return encoder


def load_clip(dataset, clip):
    # Call directly to avoid VideoDataset.__getitem__'s random replacement of failed samples.
    frames, indices = dataset.loadvideo_decord(str(ROOT / "artifacts/readiness/data" / clip["filename"]), 16)
    if len(frames) == 0:
        raise RuntimeError(f"Native decoder failed for {clip['filename']}")
    assert frames.shape == (16, clip["height"], clip["width"], 3)
    expected = np.clip(np.linspace(0, 64, num=16), 0, 63).astype(np.int64)
    assert np.array_equal(indices[0], expected), indices
    views = dataset.transform(frames)
    tensor = views[1].unsqueeze(0).contiguous()
    assert tensor.shape == (1, 3, 16, 384, 384)
    assert torch.isfinite(tensor).all()
    return tensor, indices[0].tolist()


def infer(encoder, dataset, clips, output):
    encoder.eval()
    rows, pooled = [], []
    for i, clip in enumerate(clips):
        cpu_input, indices = load_clip(dataset, clip)
        x = cpu_input.cuda()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            features = encoder(x)
        torch.cuda.synchronize()
        seconds = time.perf_counter() - start
        usage = memory()
        assert features.shape == (1, 4608, 1664), features.shape
        features = features.float().cpu()
        assert torch.isfinite(features).all()
        std = features.std().item()
        token_variance = features.var(dim=1).mean().item()
        assert std > 1e-6 and token_variance > 1e-8
        embedding = features.mean(dim=1).squeeze(0)
        pooled.append(embedding)
        if i < 2:
            torch.save(features, output / f"dense_{i:02d}.pt")
        row = dict(filename=clip["filename"], label=clip["label"], input_shape=list(x.shape),
                   frame_indices=indices, feature_shape=list(features.shape),
                   mean=features.mean().item(), std=std, token_variance=token_variance,
                   inference_seconds=seconds, **usage)
        if i == 0:
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                repeat = encoder(x).float().cpu()
            delta = (repeat - features).abs().max().item()
            torch.testing.assert_close(repeat, features, rtol=1e-4, atol=1e-5)
            row["repeat_max_abs_difference"] = delta
            row["repeat_tolerance"] = dict(rtol=1e-4, atol=1e-5)
        rows.append(row)
        save_json(output / "inference.json", rows)
        print(f"Inference {i+1}/{len(clips)}: {seconds:.2f}s, "
              f"peak {usage['peak_allocated_bytes']/2**30:.2f} GiB", flush=True)
        del x, features
    embeddings = torch.stack(pooled)
    assert embeddings.shape == (len(clips), 1664)
    assert embeddings.var(dim=0).mean().item() > 1e-8
    torch.save(dict(filenames=[r["filename"] for r in clips], embeddings=embeddings), output / "pooled.pt")
    save_json(output / "inference_summary.json", dict(samples=len(rows), passed=True,
              embedding_shape=list(embeddings.shape), between_clip_variance=embeddings.var(dim=0).mean().item(),
              max_peak_allocated_bytes=max(r['peak_allocated_bytes'] for r in rows),
              max_peak_reserved_bytes=max(r['peak_reserved_bytes'] for r in rows),
              mean_inference_seconds=sum(r['inference_seconds'] for r in rows)/len(rows)))


def backward(encoder, dataset, clip, output):
    encoder.train()
    encoder.use_activation_checkpointing = True
    encoder.zero_grad(set_to_none=True)
    gc.collect()
    torch.cuda.empty_cache()
    x, _ = load_clip(dataset, clip)
    x = x.cuda()
    generator = torch.Generator(device="cuda").manual_seed(20260909)
    projection = torch.randn(1664, 16, generator=generator, device="cuda") / 1664**0.5
    target = torch.randn(1, 16, generator=generator, device="cuda")
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        features = encoder(x)
        projected = features.float().mean(dim=1) @ projection
        loss = F.mse_loss(projected.float(), target)
    assert torch.isfinite(loss)
    loss.backward()
    torch.cuda.synchronize()
    seconds = time.perf_counter() - start
    usage = memory()
    # This diagnostic uses the video tokenizer and last-layer features only.
    unused_prefixes = ("patch_embed_img.", "img_mod_embed", "norms_block.0.",
                       "norms_block.1.", "norms_block.2.")
    missing, nonfinite, zero, unused, norms = [], [], [], [], {}
    for name, parameter in encoder.named_parameters():
        if parameter.grad is None:
            (unused if name.startswith(unused_prefixes) else missing).append(name)
            continue
        if not torch.isfinite(parameter.grad).all():
            nonfinite.append(name)
        norm = parameter.grad.norm().item()
        if norm == 0:
            zero.append(name)
        if name in ("patch_embed.proj.weight", "blocks.0.attn.qkv.weight",
                    "blocks.47.attn.qkv.weight", "norms_block.3.weight"):
            norms[name] = norm
    report = dict(passed=not missing and not nonfinite and all(v > 0 for v in norms.values()) and len(norms) == 4,
                  loss=loss.item(), objective="Seeded fixed projection of pooled features, MSE to fixed random target",
                  optimizer_step=False, activation_checkpointing=True, trainable_parameters=sum(p.numel() for p in encoder.parameters()),
                  gradient_parameters=sum(p.numel() for p in encoder.parameters() if p.grad is not None),
                  missing_gradients=missing, nonfinite_gradients=nonfinite,
                  zero_gradient_tensors=zero, expected_unused=unused, gradient_norms=norms,
                  forward_backward_seconds=seconds, **usage)
    save_json(output / "backward.json", report)
    print(json.dumps(report, indent=2), flush=True)
    assert report["passed"], "Backward validation failed; see backward.json"
    encoder.zero_grad(set_to_none=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("load", "smoke", "inference", "backward"), required=True)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "artifacts/readiness/checkpoints/vjepa2_1_vitG_384.pt")
    args = parser.parse_args()
    torch.set_num_threads(8)
    torch.manual_seed(20260909)
    np.random.seed(20260909)
    output = ROOT / "artifacts/readiness/results" / args.mode
    output.mkdir(parents=True, exist_ok=True)
    save_json(output / "run.json", dict(mode=args.mode, seed=20260909, torch=torch.__version__,
              cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(),
              commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
              parameter_dtype="float32", autocast_dtype="bfloat16", attention="native SDPA",
              frames=16, frame_step=4, resolution=384, batch_size=1,
              temporal_segments=1, spatial_view="center of native three-view transform"))
    encoder = load_encoder(args.checkpoint.resolve(), output)
    if args.mode == "load":
        return
    encoder = encoder.cuda()
    manifest = json.loads((ROOT / "docs/readiness/kinetics16.json").read_text())
    clips = manifest["clips"]
    transform = make_transforms(training=False, crop_size=384, num_views_per_clip=3)
    dataset = VideoDataset(str(ROOT / "artifacts/readiness/data/paths.csv"), frames_per_clip=16,
                           frame_step=4, num_clips=1, random_clip_sampling=False,
                           filter_short_videos=True, transform=transform)
    if args.mode == "backward":
        backward(encoder, dataset, clips[0], output)
    else:
        infer(encoder, dataset, clips[:2] if args.mode == "smoke" else clips, output)


if __name__ == "__main__":
    main()
