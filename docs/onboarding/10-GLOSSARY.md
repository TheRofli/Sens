# 10 — Глоссарий и раскладка файлов

## Термины

| Термин | Значение |
|--------|----------|
| **Пресет (preset)** | запись в `speech_app/models.py::MODELS` — описание одной модели (key, label, engine, model_id, family, description) |
| **Движок (engine)** | класс, реализующий `SpeechEngine` (load/unload/transcribe) — `engines/parakeet.py`, `engines/whisper.py`, `engines/gigaam.py` |
| **kind** | строка-идентификатор движка: `"parakeet"`, `"whisper"`, `"gigaam"` |
| **EngineManager** | владелец активного движка; переключает при смене модели |
| **VAD** | voice activity detection — здесь RMS-обрезка тишины по краям (`vad.py`) |
| **textpost** | постобработка текста (`textpost.py`): пробелы, галлюцинации, капитализация |
| **post_ui / post_ui_sync** | переход на UI-поток: асинхронно через очередь / синхронно с Event |
| **UI-поток** | tkinter mainloop-поток; единственный, кто трогает GUI |
| **overlay** | tkinter-индикатор поверх экрана (волна уровня / точки / уведомление) |
| **data/api.port** | файл с портом локального HTTP API |
| **runtime_state.json** | снапшот состояния для внешних наблюдателей |
| **CT2 / CTranslate2** | рантайм faster-whisper (INT8) |
| **e2e (GigaAM)** | end-to-end вариант модели: пунктуация + нормализация на выходе |

## Раскладка файлов Speech (актуально на 5e6dd3d)

```text
speech_app/
├── __init__.py
├── __main__.py            # python -m speech_app
├── api.py                 # HTTP API (ThreadingHTTPServer)
├── app.py                 # SpeechApp — оркестратор + CLI main()
├── audio.py               # AudioRecorder (sounddevice)
├── engine_manager.py      # выбор/переключение движка
├── history.py             # TranscriptHistory (JSONL)
├── hotkeys.py             # GlobalHotkeyListener (pynput + Win32)
├── model_status.py        # статус установки моделей
├── models.py              # РЕЕСТР пресетов
├── output.py              # TranscriptPublisher
├── overlay.py             # VoiceOverlay (tkinter, topmost)
├── parakeet_engine.py     # legacy shim (импорты старого кода)
├── portable.py            # portable-окружение (env vars)
├── resources.py           # монитор ресурсов
├── runtime_state.py       # write_runtime_state
├── settings.py            # AppSettings + SettingsStore
├── single_instance.py     # lock
├── system.py              # Win32: clipboard, paste, focus, spawn Tauri
├── textpost.py            # постобработка
├── tray.py                # TrayController (pystray)
├── vad.py                 # trim_silence (RMS)
├── visuals.py             # DPI, AppUserModelID
└── engines/
    ├── __init__.py
    ├── base.py            # SpeechEngine protocol, EngineUnavailable, LoadedEngine
    ├── gigaam.py          # GigaAMEngine (trust_remote_code, temp WAV)
    ├── install.py         # установка моделей + GigaAM patch
    ├── parakeet.py        # ParakeetEngine (transformers/NeMo)
    ├── paths.py           # пути моделей (models_root, whisper/gigaam dirs)
    └── whisper.py         # WhisperEngine (faster-whisper CT2 INT8)
```

```text
tests/                     # 109 unittest: test_models, test_engine_manager,
                           # test_api, test_install, test_model_status, test_vad,
                           # test_textpost, test_hotkeys, test_tray, test_overlay,
                           # test_system, test_settings, test_portable,
                           # test_single_instance, test_cli, test_launcher,
                           # test_packaging, test_resources, test_app_state,
                           # test_core
```

```text
tauri/
├── src/main.tsx           # React UI (Layout B: Статус/Модели/Настройки/История)
├── src/styles.css
├── index.html
├── package.json           # react 19, vite 6, @tauri-apps/api
└── src-tauri/
    ├── tauri.conf.json    # devUrl 1420, frontendDist ../dist
    ├── Cargo.toml
    └── src/main.rs        # Rust: HTTP-клиент, speech_root(), команды
```

```text
корень:
├── speech.ps1 / speech.sh # лаунчеры
├── bin/speech, speech.cmd # точки входа PATH
├── bootstrap.ps1/.sh      # установка с GitHub
├── requirements*.txt      # base / -parakeet / -whisper / -gigaam
├── README.md              # пользовательский README
├── docs/onboarding/       # ЭТА документация
├── data/                  # runtime (не в git)
├── models/                # веса (не в git)
├── .venv/                 # Python venv (не в git)
└── tauri/                 # GUI (dist/, node_modules/, target/ — не в git)
```

## Настройки (AppSettings) — полный список

| Поле | Тип | Дефолт | Смысл |
|------|-----|--------|-------|
| `model` | str | `"parakeet"` | ключ пресета активной модели |
| `model_id` | str | `"nvidia/parakeet-tdt-0.6b-v3"` | legacy/производное (для parakeet-бэкенда) |
| `backend` | str | `"auto"` | parakeet: auto/transformers/nemo |
| `device` | str | `"cpu"` | cpu/gpu/auto |
| `hotkey` | str | `"ctrl+win"` | горячая клавиша |
| `engine_enabled` | bool | `True` | вкл/выкл движок |
| `copy_to_clipboard` | bool | `True` | копировать в буфер |
| `paste_to_active_input` | bool | `True` | вставлять в активное поле |
| `suppress_hotkey` | bool | `False` | гасить системный Win |
| `preload_model` | bool | `True` | грузить модель при старте |
| `sample_rate` | int | `16000` | частота захвата |
| `vad_sensitivity` | float | `0.02` | порог VAD |
| `history_limit` | int | `100` | размер истории |
| `beam_size` | int | `5` | beam-поиск |
| `temperature` | float | `0.0` | температура |
| `repetition_penalty` | float | `1.0` | штраф повторов |
| `no_repeat_ngram_size` | int | `0` | запрет повтора n-грамм |
| `compression_ratio_threshold` | float | `2.4` | анти-галлюц. Whisper |
| `log_prob_threshold` | float | `-1.0` | анти-галлюц. Whisper |
| `postprocess_text` | bool | `True` | textpost вкл/выкл |
