# 05 — Лаунчеры, CLI, portable-окружение

## 1. Точки входа

```
bin/speech        (bash-скрипт)      → speech.sh "$@"
bin/speech.cmd    (cmd-обёртка)      → powershell -File speech.ps1 %*
speech.ps1        (PowerShell)       → установка/запуск/команды (Windows)
speech.sh         (bash)             → установка/запуск/команды (macOS/Linux)
bootstrap.ps1/.sh (GitHub-установка)
```

Пользователь запускает `speech` — в PATH лежит `bin/` (добавляется
установщиком).

## 2. speech.ps1 — что делает

- `Install-Speech` — создаёт `.venv`, ставит `requirements*.txt` (все четыре:
  requirements, -parakeet, -whisper, -gigaam), добавляет `bin` в user PATH.
- `Run-Speech` — основной запуск: экспортирует portable-окружение и
  вызывает `pythonw -m speech_app run` (без окна консоли) или
  `python -m speech_app run` для foreground.
- Команды транслируются в `python -m speech_app <args>`:
  `status`, `stop`, `restart`, `open`, `diagnose`, `model install <key>`,
  `model list`, `parakeet install` (legacy alias), `foreground`.

## 3. CLI (`python -m speech_app`)

| Команда | Действие |
|---------|----------|
| `run` | запуск ядра (трей + API + hotkeys) |
| `run --show-window` | то же + сразу открыть Tauri-окно |
| `diagnose` (или `--diagnose`) | проверка зависимостей (torch/transformers/nemo/...) |
| `model list` | статус установки всех моделей |
| `model install [key]` | установить модель (key из реестра; по умолчанию активная) |
| `parakeet install` | legacy alias для `model install parakeet` |

Не-командный запуск (`speech` без аргументов): берёт `SingleInstanceLock`
(`data/speech.lock`), создаёт `SpeechApp`, `app.run()`. Если уже запущено —
печатает «Speech is already running.»

## 4. Portable-окружение

`speech_app/portable.py::build_portable_env(root)` — единая точка задания
путей для установки «всё в одной папке»:

```python
{
  "SPEECH_HOME":          root,
  "SPEECH_DATA_DIR":      root/"data",
  "HF_HOME":              root/"models"/"huggingface",
  "HF_HUB_CACHE":         root/"models"/"huggingface"/"hub",
  "TRANSFORMERS_CACHE":   root/"models"/"huggingface"/"transformers",
  "TORCH_HOME":           root/"models"/"torch",
  "XDG_CACHE_HOME":       root/"cache",
}
```

`app.main()` ставит их через `os.environ.setdefault` (не перезаписывает
существующие) до импорта тяжёлых библиотек. Лаунчеры делают то же самое
(export SPEECH_HOME/HF_HOME).

**Следствие:** без этих переменных код ищет данные в `%APPDATA%\Speech`
(`default_data_dir()`), модели — в `%APPDATA%\Speech\models`. С ними — в
корне проекта. Поэтому тесты и CLI часто гоняют с
`SPEECH_HOME=<root>`, чтобы не пачкать APPDATA.

## 5. Раскладка на диске

```text
D:\Speech\
├── data/
│   ├── settings.json        # AppSettings (JSON, camelCase-free)
│   ├── history.jsonl        # транскрипты (JSONL)
│   ├── api.port             # порт HTTP API
│   ├── runtime_state.json   # снапшот для внешних читателей
│   └── speech.lock          # single-instance lock
├── models/
│   ├── huggingface/         # HF-кэш (Parakeet; hub/, transformers/, torch/)
│   ├── whisper/<key>/       # конвертированные CT2-веса + INSTALLED.json
│   └── gigaam/<key>/        # патченая копия GigaAM (модель + modeling_gigaam.py)
├── cache/                   # XDG cache
├── tmp/                     # temp (аудио и т.п.)
└── .venv/                   # виртуальное окружение (не в git)
```

## 6. Установка моделей

```powershell
speech model install parakeet    # snapshot_download в HF-кэш
speech model install whisper-ru  # конверсия CT2 INT8 (нужны RAM/время)
speech model install gigaam      # скачивание + патч под transformers 5.x
speech model list                # статус всех
```

Механизм — в `speech_app/engines/install.py` (см. `02-ENGINES.md` §6).

## 7. Зависимости

| Файл | Что ставит |
|------|-----------|
| `requirements.txt` | базовые (PySide? нет — tkinter/…, pystray, pynput, sounddevice, pyperclip, numpy…) |
| `requirements-parakeet.txt` | torch, accelerate, librosa, safetensors, sentencepiece, huggingface-hub, transformers (git) |
| `requirements-whisper.txt` | faster-whisper, ctranslate2 |
| `requirements-gigaam.txt` | soundfile, hydra-core, omegaconf, torchaudio, sentencepiece |

**Важно:** transformers ставится из git (`git+https://github.com/huggingface/transformers.git`) — это 5.x-dev. Все движки заточены под неё (см. патч GigaAM в `02-ENGINES.md`). Не понижай до стабильной 4.x без проверки всех трёх движков.

## 8. Тесты

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s .\tests
```

109 тестов: models, engine_manager, api (мок app), install (патч-функции),
model_status, vad, textpost, hotkeys, overlay, tray (арность pystray),
settings, portable, single_instance, system, resources, launcher-скрипты,
cli. Тесты не трогают реальные модели (моки/tempdir).
