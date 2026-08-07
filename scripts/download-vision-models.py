"""Explicit downloader for local vision GGUF packs (CPU-only). No silent network."""
from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "sidecars"))

from sight.vlm import PACKS, models_root  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _fetch(url: str, dest: Path, expected_sha256: str) -> None:
    if dest.is_file() and _sha256(dest) == expected_sha256:
        print(f"skip {dest} (verified)")
        return
    partial = dest.with_suffix(dest.suffix + ".part")
    partial.unlink(missing_ok=True)
    print(f"downloading {url} -> {dest}")
    try:
        urllib.request.urlretrieve(url, partial)  # noqa: S310 - fixed https hosts
        if partial.read_bytes()[:4] != b"GGUF":
            raise RuntimeError(f"bad GGUF magic for {dest}")
        actual = _sha256(partial)
        if actual != expected_sha256:
            raise RuntimeError(
                f"SHA-256 mismatch for {dest.name}: expected {expected_sha256}, got {actual}"
            )
        partial.replace(dest)
    except Exception:
        partial.unlink(missing_ok=True)
        dest.unlink(missing_ok=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pack", choices=["lite", "quality", "quality_large", "all"], default="lite"
    )
    args = parser.parse_args()
    packs = list(PACKS) if args.pack == "all" else [args.pack]
    root = models_root()
    root.mkdir(parents=True, exist_ok=True)
    for name in packs:
        spec = PACKS[name]
        for key in ("text", "mmproj"):
            dest = root / spec[key]
            expected = spec.get("sha256", {}).get(key)
            if not expected:
                raise SystemExit(f"missing verified SHA-256 for {name}.{key}")
            _fetch(
                f"https://huggingface.co/{spec['repo']}/resolve/main/{spec[key]}",
                dest,
                expected,
            )
    print("done")


if __name__ == "__main__":
    main()
