"""Content-addressed dump cache with TTL."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any


CACHE_TTL_SECONDS = 7 * 24 * 3600




# --------------------------------------------------------------------------
# Dump cache: content-addressed, deterministic, TTL-bounded
# --------------------------------------------------------------------------

_last_cache_cleanup: float = 0.0


# Bump when the dump schema changes so stale dumps (e.g. without gaps,
# design QA or section style) are not served from cache.
# qa9: see returns a visual context document (Task 7, Vision 2.0).
# scene1: content-addressed artifacts and no-store-safe analysis (Sens 1.3).
# scene5: reconstruction-safe controls and OCR-counted font metrics (Sens 1.3.6).
# scene10: compact numeric badges use a local multi-font digit atlas.
# scene15: measured inline typography, protected control chrome, and inpainted
# live-text backgrounds must not reuse pre-1.3.7 visual dumps.
CACHE_SCHEMA_VERSION = "scene18"
# document19: textured canvases use a protected alpha-masked artwork layer.
# document29: reconstruction contracts carry the 1.3.7 DOM/raster policy.
DOCUMENT_CACHE_SCHEMA_VERSION = "document44"




def cache_root() -> Path:
    """Cache directory for analysis dumps (overridable via SENS_CACHE_DIR)."""
    if root := os.environ.get("SENS_CACHE_DIR"):
        return Path(root) / "sight"
    if local := os.environ.get("LOCALAPPDATA"):
        return Path(local) / "Sens" / "cache" / "sight"
    return Path.home() / ".cache" / "sens" / "sight"




def cache_key(image_path: str, region: dict[str, int] | None) -> str:
    digest = hashlib.sha256()
    with open(image_path, "rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    region_key = "full"
    if region is not None:
        region_key = "{x}x{y}x{w}x{h}".format(**region)
    return f"{CACHE_SCHEMA_VERSION}-{digest.hexdigest()[:32]}-{region_key}.json"


def document_cache_key(image_path: str, options: dict[str, Any]) -> str:
    """Return a content-addressed key for a completed visual document.

    Scene dumps and completed reconstruction contracts have different
    invalidation boundaries. Keeping a separate schema lets document-only
    changes avoid throwing away the expensive deterministic image analysis.
    """
    source_key = cache_key(image_path, None)
    source_digest = source_key.split("-", 2)[1]
    encoded = json.dumps(
        options,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    options_digest = hashlib.sha256(encoded).hexdigest()[:20]
    return f"{DOCUMENT_CACHE_SCHEMA_VERSION}-{source_digest}-{options_digest}.json"




def read_cache(key: str) -> dict[str, Any] | None:
    path = cache_root() / key
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if time.time() - payload.get("ts", 0) > CACHE_TTL_SECONDS:
        return None
    return payload.get("dump")




def write_cache(key: str, dump: dict[str, Any]) -> None:
    directory = cache_root()
    try:
        directory.mkdir(parents=True, exist_ok=True)
        payload = {"ts": time.time(), "dump": dump}
        fd, tmp_path = tempfile.mkstemp(dir=directory, suffix=".tmp")
        os.close(fd)
        Path(tmp_path).write_text(
            json.dumps(payload, ensure_ascii=False), encoding="utf-8"
        )
        os.replace(tmp_path, directory / key)
    except OSError:
        # Cache must never break analysis; failures degrade to a miss.
        return
    cleanup_cache(directory)




def cleanup_cache(directory: Path, interval: float = 3600.0) -> None:
    """Remove expired entries at most once per `interval` seconds."""
    global _last_cache_cleanup
    now = time.time()
    if now - _last_cache_cleanup < interval:
        return
    _last_cache_cleanup = now
    try:
        for path in directory.glob("*.json"):
            if now - path.stat().st_mtime > CACHE_TTL_SECONDS:
                path.unlink(missing_ok=True)
    except OSError:
        return
