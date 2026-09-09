"""Correct decord 0.6.0's embedded tag to match its published Linux wheel filename.

The PyPI wheel is named py3-none-manylinux2010_x86_64 but embeds a cp36 tag.
Only installation metadata is repaired; decoder binaries and Python code are unchanged.
"""
import base64
import csv
import hashlib
import importlib.metadata
import io
from pathlib import Path
import sys


distribution = importlib.metadata.distribution("decord")
assert distribution.version == "0.6.0"
metadata = Path(distribution._path)
assert metadata.is_relative_to(Path(sys.prefix)), "Never modify the base image"
wheel = metadata / "WHEEL"
old = wheel.read_text()
incorrect = "Tag: cp36-cp36m-manylinux2010_x86_64"
correct = "Tag: py3-none-manylinux2010_x86_64"
if incorrect in old:
    (metadata / "WHEEL.original").write_text(old)
    new = old.replace(incorrect, correct)
    wheel.write_text(new)
    record = metadata / "RECORD"
    rows = list(csv.reader(io.StringIO(record.read_text())))
    digest = base64.urlsafe_b64encode(hashlib.sha256(new.encode()).digest()).decode().rstrip("=")
    for row in rows:
        if row[0] == f"{metadata.name}/WHEEL":
            row[1:] = [f"sha256={digest}", str(len(new.encode()))]
    with record.open("w", newline="") as stream:
        csv.writer(stream).writerows(rows)
    print("Corrected decord's embedded wheel tag to its published py3-none Linux tag")
else:
    assert correct in old, f"Unexpected decord wheel metadata: {old}"
    print("decord wheel metadata already correct")
