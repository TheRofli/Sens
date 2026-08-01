# 01 — Архитектура

## 1. Общая картина

Speech — это **один Python-процесс** (`python -m speech_app run`), который:

- живёт в системном трее (pystray),
- слушает глобальные горячие клавиши (pynput + Win32 filter),
- захватывает микрофон (sounddevice),
- транскрибирует через выбранный ASR-движок,
- публикует результат (вставка в активное окно + клипборд + история),
- поднимает **локальный HTTP API** на эфемерном порту `127.0.0.1:<port>`
  (порт записывается в `data/api.port`),
- запускает **Tauri-окно** как отдельный процесс (GUI), который ходит в этот
  API.

```
┌───────────────────────────── Python-ядро (1 процесс) ─────────────────────────────┐
│                                                                                     │
│  tkinter root (withdraw)                                                            │
│   ├── ui_queue ──► _pump_ui_queue (каждые 30мс) ──► всё UI-состояние               │
│   ├── VoiceOverlay (topmost, transparent)                                           │
│   │                                                                                 │
│  TrayController (pystray, отдельный поток) ──► SpeechApp-методы (через post_ui)     │
│  GlobalHotkeyListener (pynput, отдельный поток) ──► on_start/on_stop ──► post_ui    │
│  AudioRecorder (sounddevice callback, поток аудио) ──► chunks + level_callback      │
│  EngineManager (движки Parakeet/Whisper/GigaAM; загрузка в background-потоке)       │
│  SpeechAPIServer (ThreadingHTTPServer, daemon-поток) ──► post_ui_sync для мутаций   │
│  TranscriptHistory (JSONL, data/history.jsonl)                                      │
│                                                                                     │
└─────────────────────────────────────────────────────────────────────────────────────┘
                          │  data/api.port
                          ▼
┌───────────────────────────── Tauri (отдельный процесс) ────────────────────────────┐
│  Rust (src-tauri) — HTTP-клиент к API, команды для React                            │
│  React (main.tsx) — Статус / Модели / Настройки / История (поллинг 2с)              │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

## 2. Жизненный цикл диктовки (самый важный сценарий)

```
1. Пользователь зажимает Ctrl+Win
   └─ pynput on_press → HotkeyState.press → on_start()
       └─ post_ui(_begin_recording)      [поток pynput → UI-поток]
           └─ recorder.start()  + overlay.show_recording()

2. Говорит — sounddevice callback пишет float32 chunks в self._chunks,
   level_callback обновляет overlay-волну (через post_ui).

3. Пользователь отпускает Ctrl+Win
   └─ on_release → HotkeyState.release → on_stop()
       └─ post_ui(_finish_recording)     [UI-поток]
           └─ samples = recorder.stop()
           └─ transcribing = True
           └─ Thread(_transcribe_worker).start()   [фоновый поток]

4. _transcribe_worker (фоновый поток):
   a. trim_silence(samples, vad_sensitivity)   — обрезать тишину по краям
      → пусто = "No speech detected" (публикуем пустую строку)
   b. engine.transcribe(trimmed, sr, settings) — модель
   c. text = postprocess(raw)                  — чистка (если postprocess_text)
   d. post_ui(_publish_transcript)             — вернуть на UI-поток
      → overlay.hide(), через 140мс:
        publisher.publish(text, settings):
          - history.add(text)
          - copy_to_clipboard(text)      [если copy_to_clipboard]
          - paste_into_active_input()    [если paste_to_active_input]
            → SystemActions: вспомнить активное окно до диктовки,
              вернуть фокус, Ctrl+V, клик в точку курсора
   e. ВСЕГДА post_ui(transcribing = False) — в finally/по завершении,
      чтобы overlay никогда не завис
```

**Критично:** каждый шаг перехода между потоками использует `post_ui`
(асинхронная очередь) или `post_ui_sync` (синхронная с Event). Прямой доступ
к tkinter из не-UI потоков запрещён. Подробности — в `04-THREADING.md`.

## 3. Состояния модели

Модель (активный движок) имеет состояния, которые видит весь мир:

```
unloaded ──load_model_background──► loading ──_model_load_succeeded──► loaded
    ▲                                    │
    └────────────unload_model────────────┘

error: _model_load_failed(exc) — last_error + tray-уведомление
```

Флаги: `self.model_loading` (bool, защищён UI-потоком), `self.engine.is_loaded`.

- `load_model_background()` — потокобезопасный вход: если уже loaded/loading —
  no-op; иначе `model_loading=True`, спавн `_load_model_worker` (daemon).
- `_load_model_worker` — в фоне `engine.load(settings)`; результат — через
  `post_ui(_model_load_succeeded | _model_load_failed)`.
- Смена модели (`set_model`) сохраняет settings и вызывает `unload_model()`;
  повторная загрузка — lazy (при preload или при следующей диктовке).

## 4. Компоненты и их обязанности

### `SpeechApp` (`app.py`)
Оркестратор всего. Владеет: root, ui_queue, settings, history, system,
publisher, engine, overlay, tray, recorder, hotkey_listener, api_server.
Все публичные методы, дёргаемые из других потоков, — потокобезопасны
(сами делают `post_ui`).

### `EngineManager` (`engine_manager.py`)
Держит ровно один загруженный движок. `resolve_engine(settings)` → kind
(`"parakeet" | "whisper" | "gigaam"`), `make_engine(kind)` → экземпляр.
`load()`/`transcribe()` переключают движок, если модель в настройках
изменилась (unload старого → load нового).

### `models.py` — реестр
```python
MODELS: dict[str, ModelPreset] = {
    "parakeet": ModelPreset(key, label, engine="parakeet",
                            model_id="nvidia/parakeet-tdt-0.6b-v3",
                            family="parakeet-tdt", description=...),
    "whisper-ru": ModelPreset(..., engine="whisper",
                              model_id="coriollon/whisper-large-v3-turbo-russian-codeswitch",
                              family="whisper", ...),
    "gigaam": ModelPreset(..., engine="gigaam",
                          model_id="ai-sage/GigaAM-v3",
                          family="gigaam", ...),
}
```
Пресет → `available_presets()` (порядок показа), `get_preset(key)`,
`resolve_engine(settings)`, `resolve_model_id(settings)`.
**Единый источник правды** для трея, API, установки, статуса.

### `SpeechAPIServer` (`api.py`)
`ThreadingHTTPServer` на `127.0.0.1:0`. Порт пишется в `data/api.port`.
GET — чтение снапшотов (безопасно). POST — мутации через `post_ui_sync`
(выполняются на UI-потоке с ожиданием результата). См. `03-API.md`.

### `TrayController` (`tray.py`)
pystray `Icon("Speech", ...)` + `run_detached()`. Меню:
- Open Speech (default), Open History, Copy Last Transcript
- Engine On (checked)
- Load/Unload/Loading <модель> (dynamic labels!)
- Device (CPU/GPU/Auto, radio), Backend (Auto/Transformers/NeMo, radio)
- Model (динамическое подменю из `available_models()`)
- Quit

**Ловушка pystray:** label-колбэк — 1 аргумент `(item)`, action-колбэк — 2
аргумента `(icon, item)`, checked — 1 аргумент `(item)`. Несоответствие
арности молча убивает tray-поток. См. `08-DECISIONS.md` (коммит 0bb4c1d).

### `GlobalHotkeyListener` (`hotkeys.py`)
pynput `keyboard.Listener` с `win32_event_filter`. Специфика Win:
- Ctrl+Win не приходит как обычный press, если Win уже системно обработан —
  фильтр ловит WM_KEYDOWN/WM_KEYUP для VK 0x5B/0x5C и сам эмулирует press/release
- инъекции (сгенерированные события) игнорируются
- `suppress=True` гасит системное меню Пуск при зажатии
- `ignore_releases_for` — защита от преждевременного on_stop (микровыбросы)

### `AudioRecorder` (`audio.py`)
`sounddevice.InputStream(16k, mono, float32)`. Callback аппендит копии чанков
под lock. `stop()` → `np.concatenate(chunks)` float32 [-1, 1].

### `VoiceOverlay` (`overlay.py`)
tkinter `Toplevel`: `overrideredirect(True)`, `-topmost`, `-alpha 0.86`,
`-transparentcolor`, no-activate (WS_EX_NOACTIVATE|WS_EX_TOOLWINDOW).
Режимы: recording (волна уровня), transcribing (6 точек), notice (текст с
таймаутом). Перерисовка — `root.after(28, _draw)`.

### `SystemActions` (`system.py`)
Win32-обёртки: clipboard (pyperclip → tkinter fallback), вставка Ctrl+V
(keybd_event), remember/restore активного окна (AttachThreadInput +
SetForegroundWindow + клик в точку курсора), запуск Tauri-процесса.

### `TranscriptHistory` (`history.py`)
JSONL в `data/history.jsonl`, UUID id, ISO-дата, cap = `history_limit` (100).
`add()` пишет свежую запись первой. `list()` читает файл и обрезает.

### `TranscriptPublisher` (`output.py`)
`publish(text, settings)`: strip → если пусто → None; иначе history.add +
clipboard + paste (по настройкам). Возвращает entry или None.

### `vad.py` / `textpost.py`
Чистые утилиты, см. `06-PIPELINE.md`.

## 5. Runtime-файлы на диске

| Файл | Что это |
|------|---------|
| `data/settings.json` | все настройки (см. `speech_app/settings.py`) |
| `data/history.jsonl` | история транскриптов |
| `data/api.port` | порт HTTP API (пишет ядро, читает Tauri) |
| `data/runtime_state.json` | снапшот состояния для внешних инструментов (running, model_state, device, backend, last_error) |
| `data/speech.lock` | lock одного экземпляра |

## 6. Где живут данные моделей

- `models/huggingface/` — HF-кэш (Parakeet; hub/ внутри)
- `models/whisper/<key>/` — конвертированные CTranslate2 веса + INSTALLED.json
- `models/gigaam/<key>/` — патченая копия GigaAM (модель + modeling_gigaam.py с патчем)

Пути считаются в `speech_app/engines/paths.py` с учётом `SPEECH_HOME` /
`HF_HOME` / portable-режима. См. `05-LAUNCHERS-CLI.md`.
