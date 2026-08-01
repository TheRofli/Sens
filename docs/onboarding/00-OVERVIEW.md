# 00 — Overview: что такое Speech

## Зачем это приложение

Speech — локальный **push-to-talk диктант**. Пользователь зажимает горячую
клавишу, говорит (можно по-русски, по-английски, вперемешку — язык
определяется автоматически), отпускает — и готовый текст:

1. вставляется в активное поле ввода (Ctrl+V в то окно, которое было в
   фокусе до диктовки),
2. копируется в буфер обмена,
3. сохраняется в локальную историю (последние 100 записей, поиск по тексту).

Всё считается **локально, на CPU**. Никаких облачных ASR-сервисов.

## Ключевые свойства

| Свойство | Значение |
|----------|----------|
| Платформа | Windows 11 (основная), macOS (source-install) |
| Язык распознавания | авто-детект; RU + EN код-свитчинг в одном предложении |
| Оборудование | CPU (GPU не требуется и по умолчанию не используется) |
| Модели | Parakeet TDT 0.6B v3, Whisper large-v3-turbo RU-codeswitch, GigaAM v3 e2e |
| UI | Tauri 2 (React 19 + Rust) — единственное окно; трей pystray — полное меню |
| Хранение | всё локально в `data/`, `models/` — в репозиторий не попадает |

## Стек

- **Python 3.11** — ядро:
  - `tkinter` — скрытый корневой окно (`root.withdraw()`) + overlay-индикатор
  - `pystray` — системный трей (полное меню: открыть окно, копировать,
    модель, устройство, бэкенд, выйти)
  - `pynput` — глобальные горячие клавиши (low-level Win32 filter)
  - `sounddevice` — захват микрофона (16 кГц, mono, float32)
  - `transformers` / `faster-whisper` (CTranslate2) / hydra+torchaudio — ASR
  - `http.server` (stdlib) — локальный HTTP API для Tauri
- **Tauri 2** (`tauri/`) — GUI-окно: React 19 + TypeScript + Vite, Rust-бэкенд
  (`src-tauri/`), общение с ядром через HTTP `127.0.0.1:<port>` (порт в
  `data/api.port`)

## Как это ощущается пользователем

1. `speech` в терминале → появляется иконка в трее.
2. Зажимаешь `Ctrl+Win` → в центре экрана появляется overlay с волной уровня.
3. Говоришь (русский + английский как угодно), отпускаешь.
4. Overlay показывает «Transcribing…», затем текст вставляется в то поле,
   где был курсор, и в трее мелькает «Inserted».
5. История и настройки — в Tauri-окне (секции Статус / Модели / Настройки /
   История).

## Что есть в репозитории (топ-уровень)

```text
D:\Speech\
├── speech_app/            # Python-ядро (весь код)
│   ├── app.py             # SpeechApp: оркестрация всего
│   ├── api.py             # локальный HTTP API
│   ├── engines/           # ASR-движки: base, parakeet, whisper, gigaam, install, paths
│   ├── engine_manager.py  # выбор/загрузка активного движка
│   ├── models.py          # реестр пресетов моделей (единый источник правды)
│   ├── model_status.py    # статус установки моделей
│   ├── audio.py           # AudioRecorder (sounddevice)
│   ├── hotkeys.py         # глобальные хоткеи (pynput + Win32 filter)
│   ├── overlay.py         # tkinter overlay-индикатор
│   ├── output.py          # публикация транскрипта (клипборд + вставка + история)
│   ├── history.py         # история (JSONL, UUID, cap 100)
│   ├── vad.py             # RMS-обрезка тишины по краям
│   ├── textpost.py        # детерминированная чистка текста
│   ├── system.py          # Win32-действия: вставка, фокус, клипборд, запуск Tauri
│   ├── settings.py        # AppSettings + SettingsStore (JSON)
│   ├── runtime_state.py   # data/runtime_state.json для внешних читателей
│   ├── portable.py        # portable-окружение (SPEECH_HOME и др.)
│   ├── single_instance.py # lock на один экземпляр
│   ├── resources.py       # монитор ресурсов процесса
│   └── visuals.py         # DPI-awareness, AppUserModelID
├── tauri/                 # Tauri-оболочка (src/ — React, src-tauri/ — Rust)
├── tests/                 # 109 юнит-тестов (unittest)
├── speech.ps1 / speech.sh # лаунчеры (install, run, команды)
├── bin/                   # speech / speech.cmd — точки входа
├── bootstrap.ps1 / .sh    # установка с GitHub
├── requirements*.txt      # зависимости по движкам
├── data/                  # runtime: settings.json, history.jsonl, api.port
└── models/                # веса моделей (в git не попадают)
```

## Запуск за 5 минут

```powershell
# 1. Проверить, что venv готов
.\.venv\Scripts\python.exe -m speech_app --diagnose

# 2. Установить модель (если ещё нет)
speech model list                 # посмотреть статус
speech model install gigaam       # например, GigaAM

# 3. Запустить ядро
speech                            # tray + API + hotkeys
speech foreground                 # то же, но с логами в терминале (для отладки)
```

## Ключевые файлы для быстрого входа

| Файл | Строк | Зачем читать первым |
|------|-------|---------------------|
| `speech_app/app.py` | ~670 | оркестратор: все потоки и переходы состояний |
| `speech_app/models.py` | ~100 | реестр моделей — точка расширения |
| `speech_app/engine_manager.py` | ~85 | выбор движка |
| `speech_app/api.py` | ~265 | контракт с Tauri |
| `speech_app/settings.py` | ~80 | все настройки (data model) |
| `speech_app/tray.py` | ~230 | трей-меню (pystray arity-ловушки!) |
| `speech_app/hotkeys.py` | ~180 | Win32-специфика хоткеев |
| `tauri/src/main.tsx` | ~900 | весь React UI |
| `tauri/src-tauri/src/main.rs` | ~400 | Rust-команды + discovery пути |
