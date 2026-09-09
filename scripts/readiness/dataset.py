"""Stream a tiny Kinetics-400 train subset, or replay its checked-in manifest."""

import argparse
import csv
import hashlib
import io
import json
import tarfile
from collections import Counter
from pathlib import Path

import decord
import numpy as np
import requests

ROOT = Path(__file__).resolve().parents[2]
BASE = "https://s3.amazonaws.com/kinetics/400/"


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def inspect_video(path):
    vr = decord.VideoReader(str(path), num_threads=2)
    if len(vr) < 64:
        raise ValueError(f"Too short: {len(vr)} frames")
    indices = np.clip(np.linspace(0, 64, num=16), 0, 63).astype(np.int64)
    frames = vr.get_batch(indices).asnumpy()
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"Unexpected decoded shape: {frames.shape}")
    return dict(frames=len(vr), fps=float(vr.get_avg_fps()), height=int(frames.shape[1]), width=int(frames.shape[2]))


def download_subset(output, manifest_path, count):
    output.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    response = session.get(BASE + "annotations/train.csv", timeout=60)
    response.raise_for_status()
    annotation_bytes = response.content
    (output / "train.csv").write_bytes(annotation_bytes)
    annotations = {}
    for row in csv.DictReader(io.StringIO(response.text)):
        name = f"{row['youtube_id']}_{int(row['time_start']):06d}_{int(row['time_end']):06d}.mp4"
        annotations[name] = row
    replay = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    rows, rejected, classes = [], [], Counter()
    if replay:
        if count != len(replay["clips"]):
            raise ValueError("Requested count differs from saved manifest")
        if hashlib.sha256(annotation_bytes).hexdigest() != replay["annotations_sha256"]:
            raise ValueError("Source annotations changed; investigate before replacing the manifest")
        wanted = {c["member"]: c for c in replay["clips"]}
        shards = list(dict.fromkeys(c["archive"] for c in replay["clips"]))
        for clip in replay["clips"]:
            local = output / clip["filename"]
            if local.exists() and sha256(local) == clip["sha256"]:
                inspect_video(local)
                rows.append(clip)
                wanted.pop(clip["member"])
    else:
        response = session.get(BASE + "train/k400_train_path.txt", timeout=60)
        response.raise_for_status()
        shards = response.text.splitlines()
        wanted = None
    for url in shards:
        if len(rows) == count:
            break
        url = url.replace("http:", "https:")
        print(f"Streaming {url}", flush=True)
        with session.get(url, stream=True, timeout=(30, 120)) as response:
            response.raise_for_status()
            with tarfile.open(fileobj=response.raw, mode="r|gz") as archive:
                for member in archive:
                    if not member.isfile() or not member.name.endswith(".mp4"):
                        continue
                    if wanted is not None and member.name not in wanted:
                        continue
                    name = Path(member.name).name
                    annotation = annotations.get(name)
                    if annotation is None or annotation["split"] != "train":
                        rejected.append(dict(member=member.name, reason="No matching train annotation"))
                        continue
                    if wanted is None and classes[annotation["label"]] >= 2:
                        continue
                    # Write only regular video members, using basenames rather than archive paths.
                    destination = output / name
                    partial = destination.with_suffix(".mp4.part")
                    stream = archive.extractfile(member)
                    with partial.open("wb") as out:
                        while chunk := stream.read(1024 * 1024):
                            out.write(chunk)
                    try:
                        info = inspect_video(partial)
                    except Exception as exc:
                        rejected.append(dict(member=member.name, reason=str(exc)))
                        partial.unlink()
                        if replay:
                            raise
                        continue
                    digest = sha256(partial)
                    if wanted is not None and digest != wanted[member.name]["sha256"]:
                        raise ValueError(f"Checksum mismatch for {name}")
                    partial.replace(destination)
                    clip = dict(
                        filename=name,
                        archive=url,
                        member=member.name,
                        sha256=digest,
                        bytes=destination.stat().st_size,
                        **annotation,
                        **info,
                    )
                    if replay:
                        clip = wanted.pop(member.name)
                    rows.append(clip)
                    classes[annotation["label"]] += 1
                    print(f"{len(rows)}/{count}: {name}: {annotation['label']}", flush=True)
                    if len(rows) == count:
                        break
    if len(rows) != count:
        raise RuntimeError(f"Only obtained {len(rows)} of {count} clips")
    if replay:
        rows = replay["clips"]
        manifest = replay
    else:
        manifest = dict(
            dataset="Kinetics-400",
            split="train",
            count=count,
            selection="First decodable annotated clips in archive order; <=2 per class; >=64 frames",
            annotations_url=BASE + "annotations/train.csv",
            annotations_sha256=hashlib.sha256(annotation_bytes).hexdigest(),
            clips=rows,
            rejected=rejected,
        )
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    # Native VideoDataset format: absolute path then integer label, no header.
    labels = {label: i for i, label in enumerate(sorted({r["label"] for r in annotations.values()}))}
    (output / "paths.csv").write_text("".join(f"{output / r['filename']} {labels[r['label']]}\n" for r in rows))
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Verified {len(rows)} clips, {len(set(r['label'] for r in rows))} classes", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/readiness/data")
    parser.add_argument("--manifest", type=Path, default=ROOT / "docs/readiness/kinetics16.json")
    parser.add_argument("--count", type=int, default=16)
    args = parser.parse_args()
    download_subset(args.output.resolve(), args.manifest.resolve(), args.count)
