"""Resume the official checkpoint with four independent HTTP range requests."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import time

import requests

ROOT = Path(__file__).resolve().parents[2]
URL = "https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitG_384.pt"
SIZE = 30238058912
DEST = ROOT / "artifacts/readiness/checkpoints/vjepa2_1_vitG_384.pt"
PART = DEST.with_suffix(".pt.part")
STATE = DEST.with_suffix(".pt.download.json")


def write_state(state):
    temporary = STATE.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(state, indent=2) + "\n")
    temporary.replace(STATE)


def fetch_range(start, end):
    for attempt in range(5):
        try:
            with requests.get(URL, headers={"Range": f"bytes={start}-{end}"},
                              stream=True, timeout=(30, 120)) as response:
                response.raise_for_status()
                expected = f"bytes {start}-{end}/{SIZE}"
                if response.status_code != 206 or response.headers.get("Content-Range") != expected:
                    raise RuntimeError(f"Server did not honor range: {response.status_code}, {response.headers}")
                written = 0
                with PART.open("r+b", buffering=0) as output:
                    output.seek(start)
                    for block in response.iter_content(1024 * 1024):
                        if written + len(block) > end - start + 1:
                            raise RuntimeError("Range response exceeds requested length")
                        view = memoryview(block)
                        while view:
                            n = output.write(view)
                            if not n:
                                raise OSError("Checkpoint write made no progress")
                            view = view[n:]
                        written += len(block)
                    os.fsync(output.fileno())
                if written != end - start + 1:
                    raise RuntimeError(f"Truncated range: {written} bytes")
            return str(start)
        except (requests.RequestException, OSError, RuntimeError):
            if attempt == 4:
                raise
            time.sleep(min(2**attempt, 10))


def main():
    DEST.parent.mkdir(parents=True, exist_ok=True)
    if not DEST.exists():
        if STATE.exists():
            state = json.loads(STATE.read_text())
            if not PART.exists():
                raise RuntimeError("Download state exists without its partial file")
        else:
            # A pre-existing curl partial file is a contiguous, reusable prefix.
            prefix = PART.stat().st_size if PART.exists() else 0
            if prefix > SIZE:
                raise RuntimeError("Oversized partial checkpoint")
            state = dict(url=URL, size=SIZE, prefix_bytes=prefix, chunk_bytes=512 * 1024**2, completed=[])
            PART.touch(exist_ok=True)
            write_state(state)
        assert state["url"] == URL and state["size"] == SIZE
        ranges = [(start, min(start + state["chunk_bytes"], SIZE) - 1)
                  for start in range(state["prefix_bytes"], SIZE, state["chunk_bytes"])]
        pending = [(start, end) for start, end in ranges if str(start) not in state["completed"]]
        print(f"Reusing {state['prefix_bytes']/1e9:.2f} GB prefix; {len(pending)} ranges remain", flush=True)
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(fetch_range, start, end) for start, end in pending]
            for future in as_completed(futures):
                state["completed"].append(future.result())
                write_state(state)
                print(f"Completed {len(state['completed'])}/{len(ranges)} ranges", flush=True)
        assert PART.stat().st_size == SIZE
        PART.replace(DEST)
    assert DEST.stat().st_size == SIZE
    print("Computing full checkpoint SHA-256", flush=True)
    with DEST.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    reference = ROOT / "docs/readiness/checkpoint.sha256"
    if reference.exists():
        expected = reference.read_text().split()[0]
        if digest != expected:
            raise RuntimeError(f"Checkpoint SHA-256 mismatch: {digest} != {expected}")
    checksum = f"{digest}  {DEST.relative_to(ROOT)}\n"
    (DEST.parent / "checkpoint.sha256").write_text(checksum)
    print(checksum, flush=True)


if __name__ == "__main__":
    main()
