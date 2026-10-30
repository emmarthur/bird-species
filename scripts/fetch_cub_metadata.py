#!/usr/bin/env python3
"""Download CUB-200-2011 text metadata (no bird images) for Ch. 2 training."""

from __future__ import annotations

import urllib.request
from pathlib import Path

BASE = (
    "https://raw.githubusercontent.com/IBM/cdfsl-benchmark/master/"
    "filelists/CUB/CUB_200_2011"
)
FILES = (
    "classes.txt",
    "image_class_labels.txt",
    "attributes/image_attribute_labels.txt",
)


def main() -> None:
    root = Path(__file__).resolve().parent.parent / "data" / "CUB_200_2011"
    for rel in FILES:
        dest = root / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        url = f"{BASE}/{rel}"
        print(f"Downloading {url} -> {dest}")
        urllib.request.urlretrieve(url, dest)
        print(f"  {dest.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()
