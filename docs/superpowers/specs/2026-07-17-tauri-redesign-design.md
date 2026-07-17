# Tauri UI Redesign — Design Spec

**Date:** 2026-07-17
**Status:** Approved (brainstormed)
**Scope:** Redesign the Speech GUI around Tauri as the single interface; remove tkinter.

## Context

Speech currently ships two parallel UIs:

1. A **tkinter window** (`speech_app/window.py`, ~1000 lines) opened from the tray
   via "Open Speech". It hardcodes "Parakeet" in ~16 places (labels, the
   `_refresh()` status-guessing logic that pattern-matches the substring
   `"parakeet loading"`, the "Parakeet modes" card, "Download Parakeet", etc.).
   This hardcode breaks when the Whisper model is active: the status logic
   misreads the status text and the UI shows stale/incorrect state.
2. A **Tauri shell** (`tauri/`) that was rewritten in the multi-model phase to a
   live HTTP-API client. A release `.exe` already exists (built 2026-06-29,
   pre-multi-model changes); the toolchain (Rust via `~/.cargo`, node via
   `C:\Program Files\nodejs`, `tauri-cli` in `node_modules`) is installed.

The user's complaints, in order:
- The tray and window still show "Parakeet" everywhere even after Whisper was
  wired in.
- The interface is cluttered and inconvenient.

## Decision (brainstorm outcomes)

- **Approach:** Tauri becomes the *only* GUI. tkinter (`window.py`) is deleted.
  If Tauri is not built, only tray + CLI remain (accepted trade-off).
- **Window layout (choice B):** sidebar + main area, 4 sections:
  **Статус / Модели / Настройки / История**.
- **Model cards (choice Y):** full detail cards (name, repo id, description,
  install size/status, actions) — not compact tiles.
- **Settings (choice Q):** accordion of 3 collapsible sections, one open at a
  time: **Качество и очистка / Движок и вывод / Горячие клавиши**.
- **Tray:** keep the full menu, but strip every hardcoded "Parakeet" string and
  drive labels from the active model's label.
- **Install progress:** spinner + "Устанавливается…" with the button disabled;
  no percentage. The "installed" state arrives via the 2 s status poll.

## What is removed

- `speech_app/window.py` (the tkinter `SpeechWindow` and all its rendering).
- From `speech_app/app.py`: the `SpeechWindow` instance, the
  `_show_primary_window` tkinter-vs-tauri branch (now Tauri-only), and any
  window-only refresh hooks. The `WindowApp`-style helper methods that the API
  and tray also use (`get_settings_values`, `save_settings_values`,
  `history_rows`, `copy_history_entry`, `current_model_label`,
  `available_models`, `set_model`, etc.) **stay** — they are not tkinter-bound.
- Hardcoded "Parakeet" strings in `tray.py` (Load/Unload/Loading items) and
  `visuals.py` (AppUserModelID string) — replaced with dynamic model labels.
- Tests that import or mock `SpeechWindow`/tkinter (`test_app_state.py`'s window
  assertions, `test_overlay.py` if window-coupled). These are updated or removed.

## What stays unchanged

- The ASR pipeline: `engines/`, `engine_manager.py`, `vad.py`, `textpost.py`.
- `tray.py` structure (full menu) — only label hardcode is fixed.
- `overlay.py` (recording/transcribing indicator) — not redesigned here.
- `api.py` (the HTTP-API server) — no new endpoints needed.
- `audio.py`, `hotkeys.py`, `system.py`, `output.py`, `history.py`,
  `runtime_state.py`, `portable.py`, `model_status.py`.
- CLI commands (`speech model install/list`, etc.).
- `settings.json` / `history.jsonl` formats.

## Architecture

```
Tray icon (pystray)
  └─ "Open Speech" → SystemActions.open_tauri_ui()
                      └─ launches tauri/src-tauri/target/release/speech-tauri.exe
                          (or `npm run tauri:dev` in dev)

Tauri window (React + Rust)
  └─ HTTP client → http://127.0.0.1:<port>  (port from data/api.port)
      GET  /api/status
      GET  /api/models
      GET/POST /api/settings
      POST /api/model, /api/model/load, /api/model/unload, /api/model/install
      GET  /api/history
      POST /api/history/copy, /api/action/copy_last

Python core (SpeechApp)
  └─ SpeechAPIServer (api.py) serves the above; UI thread-safe via post_ui_sync
```

No new IPC; the existing HTTP-API already exposes everything the redesigned UI
needs.

## UI: Tauri window

### Shell (sidebar + main)

- Window: 1120×760, minWidth 760, minHeight 560 (unchanged).
- **Sidebar** (left, ~230px): brand "Speech" + tagline "локальная диктовка";
  4 nav items (Статус default, Модели, Настройки, История); a "model ticket" at
  the bottom showing the active model label + load state (live, from
  `/api/status`).
- **Main area**: content of the active section. Active section persists via
  `window.location.hash` (`#status`, `#models`, `#settings`, `#history`).
- Status poll every 2 s drives the sidebar ticket and the Статус hero.

### Section: Статус (default)

- **Hero:** eyebrow "Push-to-talk"; headline = `Loading` | `Ready` | `Stopped` |
  `Off` (derived from `/api/status` `model_state` + `engine_enabled`); subline
  with the hotkey + platform hint.
- **Latest transcript:** text from `/api/history?limit=1`; "Копировать" button →
  `/api/action/copy_last`. Empty state: "Нет транскрипта. Удержи хоткей и скажи
  что-нибудь."
- **Two mini-metrics:** Устройство (CPU/CUDA), Хоткей. Informational only.
- **Explicitly absent:** RAM/CPU/Threads, the "modes" card, install commands,
  duplicate Refresh buttons.

### Section: Модели (cards, choice Y)

One full-detail card per preset from `/api/models`:

- **Header:** preset label + badge.
  - Badge text: `<size> установлен` when installed; `не установлен` otherwise;
    `загружена` when active & loaded.
- **Subheader:** `model_id` · engine (`Transformers` / `faster-whisper`).
- **Description:** `preset.description` (1–2 lines).
- **Install progress:** shown only while installing — a spinner + the line
  "Скачивание + конверсия CT2 (~5 мин, 5-8 ГБ)". No percentage. Button disabled.
- **Actions:**
  - `[Установить]` — hidden/disabled when already installed; calls
    `/api/model/install {key}`.
  - `[Выбрать активной]` — calls `/api/model {key}`; disabled on the active
    preset (which instead shows `[Активна ✓]`).
  - `[Выгрузить]` — shown when the active model is loaded; calls
    `/api/model/unload`.

### Section: Настройки (accordion, choice Q)

Three collapsible sections, one open at a time. **Quality** open by default.

1. **Качество и очистка** (open by default):
   - Постобработка текста (checkbox) → `postprocess_text`
   - VAD чувствительность (number) → `vad_sensitivity`
   - Beam size (number) → `beam_size`
   - Temperature (number) → `temperature`
   - Repetition penalty (number) → `repetition_penalty`
   - No-repeat ngram size (number, advanced) → `no_repeat_ngram_size`
   - Compression ratio threshold (number, advanced, Whisper) → `compression_ratio_threshold`
   - Log-prob threshold (number, advanced, Whisper) → `log_prob_threshold`

   The three "advanced" fields render below a "Подробнее" toggle inside the
   section so the everyday controls stay uncluttered; defaults are sane
   (`0` / `2.4` / `-1.0`) and most users never touch them.
2. **Движок и вывод:**
   - Движок включён (checkbox) → `engine_enabled`
   - Предзагрузка при запуске (checkbox) → `preload_model`
   - Вставлять в активное поле (checkbox) → `paste_to_active_input`
   - Копировать в буфер (checkbox) → `copy_to_clipboard`
   - Устройство (select: cpu/cuda/auto) → `device`
   - Backend (select: auto/transformers/nemo) → `backend`; disabled with hint
     "только для Parakeet" when the active model is Whisper.
3. **Горячие клавиши:**
   - Хоткей (text input) → `hotkey`

**Save behaviour:** no "Save" button. Every control change fires
`POST /api/settings` immediately (checkboxes/selects on change; number inputs
on blur/Enter).

### Section: История

- Header: count + search input.
- Two-pane: list (preview text + relative time) on the left, full preview on the
  right. Selection updates the preview; "Копировать" → `/api/history/copy {id}`.
- Search filters via `/api/history?q=`.

## Tray (full menu, de-hardcoded)

```
Open Speech              → open_tauri_ui()
Open History             → open_tauri_ui() at #history
Copy Last Transcript
─────────────
Engine On  ✓             → toggle_engine
Load <model label>       → load_model_background()  (was "Load Parakeet")
Loading…                 → visible only while loading
Unload <model label>     → unload_model()           (was "Unload Parakeet")
─────────────
Model  ▸                 → radio: presets (already works)
Device ▸                 → CPU / CUDA / Auto
Backend ▸                → Auto / Transformers / NeMo
─────────────
Quit
```

Every label that said "Parakeet" now uses `current_model_label()`. The Model
submenu was fixed in the previous iteration and stays.

## Open Speech path

`SystemActions.open_tauri_ui()` already prefers the release exe, then
`npm run tauri:dev`. Since tkinter is gone, this is the *only* "Open Speech"
path. If neither the exe nor npm is available, the tray posts a notification
"Sobерите Tauri: см. README" instead of falling back to a window.

## Build / verification

- `tauri/`: `npm install` (if needed), `npm run tauri:build` produces
  `tauri/src-tauri/target/release/speech-tauri.exe`. The existing toolchain
  (`~/.cargo/bin/cargo`, node, `node_modules/.bin/tauri`) supports this.
- Dev loop: `npm run tauri:dev` (hot reload) against a running `speech` core.
- Python tests: `python -m unittest discover -s tests` — must remain green after
  tkinter removal (window-coupled tests updated/removed).

## Risks

- **No GUI if Tauri breaks:** with tkinter gone, a broken/unbuilt Tauri leaves
  only tray + CLI. Accepted under Approach 1.
- **First build after redesign:** the existing release exe is pre-multi-model;
  the user must rebuild once for the new UI to take effect. Covered by the build
  instructions.
- **Label drift:** any future code that re-introduces a hardcoded "Parakeet"
  string in a label regresses the bug. Mitigated by driving all labels from
  `current_model_label()` and the presets registry.

## Out of scope (explicit)

- Redesigning the recording overlay.
- New CLI commands.
- README rewrite (a minor touch-up of the window-sections list may follow, but
  is not part of this spec).
- The ASR pipeline, engines, VAD, post-processing.
