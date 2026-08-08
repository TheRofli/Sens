from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from .settings import AppSettings, default_data_dir


@dataclass(slots=True)
class RuntimeState:
    running: bool
    model_state: str
    device: str
    backend: str
    last_error: str
    updated_at: str


def write_runtime_state(
    model_state: str,
    settings: AppSettings,
    running: bool = True,
    last_error: str = "",
    path: Path | None = None,
) -> None:
    target = path or default_data_dir() / "runtime_state.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    state = RuntimeState(
        running=running,
        model_state=model_state,
        device=settings.device,
        backend=settings.backend,
        last_error=last_error,
        updated_at=datetime.now(timezone.utc).isoformat(),
    )
    target.write_text(
        json.dumps(asdict(state), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
