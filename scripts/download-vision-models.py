"""Explicit downloader for local vision GGUF packs (CPU-only). No silent network."""
from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "sidecars"))

from sight.vlm import PACKS, models_root  # noqa: E402


def _fetch(url: str, dest: Path) -> None:
    print(f"downloading {url} -> {dest}")
    urllib.request.urlretrieve(url, dest)  # noqa: S310 - fixed https hosts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pack", choices=["lite", "quality", "all"], default="lite")
    args = parser.parse_args()
    packs = ["lite", "quality"] if args.pack == "all" else [args.pack]
    root = models_root()
    root.mkdir(parents=True, exist_ok=True)
    for name in packs:
        spec = PACKS[name]
        for key in ("text", "mmproj"):
            dest = root / spec[key]
            if dest.exists() and dest.stat().st_size > 10_000_000:
                print(f"skip {dest} (exists)")
                continue
            _fetch(f"https://huggingface.co/{spec['repo']}/resolve/main/{spec[key]}", dest)
            head = dest.read_bytes()[:4]
            if head != b"GGUF":
                raise SystemExit(f"bad file {dest}: magic {head!r} (wrong repo layout? fix PACKS)")
    print("done")


if __name__ == "__main__":
    main()
