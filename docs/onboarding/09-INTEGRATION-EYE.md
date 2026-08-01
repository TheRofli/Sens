# 09 — Интеграция Speech в объединённый проект (Eye)

> **Цель этого документа:** дать агенту (сол) полный контекст для переноса
> Speech внутрь будущего объединённого приложения, где **Eye будет ядром**,
> а Speech — модулем (голос/диктовка). Проект получит новое имя; внутри
> будут и Eye, и Speech.

## 1. Исходное положение двух проектов

| | **Speech** | **Eye** |
|---|------------|---------|
| Назначение | локальная диктовка (push-to-talk ASR) | визуальное восприятие для агентов (vision + MCP) |
| Язык/стек | Python 3.11 (ядро) + Tauri 2 (GUI) | Node.js ≥20 (E SM-модули) + Tauri 2 (overlay) |
| Точка входа | `python -m speech_app run` | `node vision.mjs`, `node mcp.mjs` |
| GUI | Tauri-окно (Статус/Модели/Настройки/История) | Tauri-overlay (трей/настройки, прозрачный) |
| Агентный интерфейс | локальный HTTP API (`127.0.0.1:<port>`, `data/api.port`) | stdio MCP-сервер |
| Данные | `data/` (settings.json, history.jsonl, api.port, runtime_state.json), `models/` | `.artifacts/`, `.omx/`, config.json |
| Модели | Parakeet 0.6B, Whisper large-v3-turbo RU-codeswitch, GigaAM v3 e2e | vision-провайдер (MiMo token plan; настраивается) |

**Ключевое сходство:** оба используют Tauri 2 для UI. **Ключевое отличие:**
Speech = Python-процесс с HTTP-контрактом; Eye = Node-процесс с
MCP-контрактом. Их интеграция — это **процессная интеграция** (два рантайма
в одном приложении), а не слияние кода.

## 2. Целевая архитектура (рекомендация)

```
┌──────────────────────── Новое приложение (имя TBD) ────────────────────────┐
│                                                                             │
│  ┌──────────────┐   spawn / lifecycle   ┌──────────────────┐                │
│  │  Eye core    │ ◄────────────────────►│  Speech core     │                │
│  │  (Node.js)   │                       │  (Python 3.11)   │                │
│  │  vision+MCP  │                       │  dictation+ASR   │                │
│  └──────┬───────┘                       └────────┬─────────┘                │
│         │ MCP stdio / HTTP                       │ HTTP (data/api.port)     │
│         ▼                                        ▼                           │
│  ┌───────────────────────────────────────────────────────────┐               │
│  │  Tauri shell (единое окно или набор окон)                │               │
│  │  React: вкладки/секции Eye (perception) + Speech         │               │
│  │  (dictation, models, settings, history)                  │               │
│  └───────────────────────────────────────────────────────────┘               │
│                                                                             │
│  Лаунчер: единая команда (например `app` или `speech`/`eye` как алиасы)     │
│  Один каталог данных: data/ (speech) + .artifacts/, .omx/ (eye)             │
└─────────────────────────────────────────────────────────────────────────────┘
```

Рекомендация: **не переписывать Speech в Node** — его ядро (ASR, Win32,
потоки) зрелое и протестировано (109 тестов). Вместо этого:

- Python-ядро Speech остаётся **отдельным процессом** (или как минимум
  отдельным модулем с собственным venv);
- единый **оркестратор-лаунчер** (можно Node, чтобы переиспользовать
  инфраструктуру Eye) управляет жизненным циклом обоих ядер;
- Tauri-шелл расширяется: секции Eye + секции Speech (или общий сайдбар с
  двумя разделами).

## 3. Контракты, которые надо сохранить (не ломать!)

### 3.1 HTTP API Speech (полная спецификация — `03-API.md`)

- `GET /api/status` — состояние ядра и модели
- `GET /api/settings`, `POST /api/settings` — настройки (merge, sync)
- `GET /api/models` — пресеты с установочным статусом
- `POST /api/model`, `/api/model/load`, `/api/model/unload`, `/api/model/install`
- `GET /api/history?limit&q`, `POST /api/history/copy`, `POST /api/action/copy_last`

Поля — **snake_case**; discovery — `data/api.port` + `data/runtime_state.json`.
Это идеальный шов для интеграции: **Eye-агент сможет управлять Speech через
HTTP** (например, просить переключить модель, читать историю) вообще без
знания Python.

### 3.2 MCP-сервер Eye (stdio)

`node mcp.mjs` — инструменты `eye_describe`, `eye_read`, `eye_locate`,
`eye_inspect`, `eye_compare`, `eye_artifact_get`. В объединённом приложении
этот сервер можно оставить как есть; при желании добавить MCP-инструмент
`speech_transcribe` (прокси к HTTP API Speech) — тогда агенты смогут
диктовать через Eye-стек.

### 3.3 Portable-окружение Speech

`build_portable_env(root)` задаёт SPEECH_HOME / SPEECH_DATA_DIR / HF_HOME /
TRANSFORMERS_CACHE / TORCH_HOME / XDG_CACHE_HOME. При переносе Speech в новое
приложение эти переменные должны указывать на каталоги нового проекта
(см. §5 «Монтирование»).

## 4. Что переносить как есть, а что переделывать

### Переносить без изменений (ядро)

- `speech_app/` целиком (Python-модуль; единственная зависимость от
  расположения — `Path(__file__).resolve().parents[1]` для корня проекта —
  переживёт перенос, если Speech лежит подпапкой).
- `engines/`, `models.py`, `engine_manager.py`, `vad.py`, `textpost.py`,
  `history.py`, `settings.py`, `api.py`, `tray.py`, `hotkeys.py`,
  `system.py`, `overlay.py`, `audio.py` — без изменений.
- `tests/` — 109 тестов (unittest) должны остаться зелёными.

### Адаптировать

| Что | Что менять |
|-----|-----------|
| Лаунчеры `speech.ps1` / `speech.sh` / `bin/` | заменить на единый лаунчер нового приложения (или оставить как внутренние, но не публичные) |
| `bootstrap.ps1/.sh` | заменить установкой нового приложения |
| Tauri `tauri/` (React UI Speech) | **не переносить как отдельное окно** — перенести секции (Статус/Модели/Настройки/История) в общий Tauri-шелл с Eye |
| `SystemActions.open_tauri_ui` | указывать на exe нового приложения |
| CLI (`python -m speech_app ...`) | оставить как внутренний интерфейс; публичный CLI — у нового приложения |

### Переписать (новая работа)

- Единый оркестратор: старт/стоп Python-ядра и Node-ядра, порядок
  инициализации (Speech стартует после Eye или параллельно), завершение
  (graceful shutdown обоих).
- Общий Tauri-шелл: сайдбар с разделами Eye и Speech; интеграция стилей.
- Установщик: один venv (Python) + node_modules; скачивание моделей Speech
  + конфигурация vision-провайдера Eye.

## 5. Монтирование (пути и данные) — конкретный план

Новое приложение, например, `C:\...\<NewName>\`:

```text
<NewName>/
├── eye/                  # текущий Eye (Node) — перенести целиком
│   ├── vision.mjs, mcp.mjs, src/, overlay/, skills/
│   └── config.json       # vision-провайдер (MiMo token plan)
├── speech/               # текущий Speech (Python) — перенести целиком
│   ├── speech_app/
│   ├── models/           # веса ASR
│   ├── requirements*.txt
│   └── tests/
├── data/                 # общие данные:
│   ├── speech/           #   settings.json, history.jsonl, api.port,
│   │                     #   runtime_state.json (бывшее speech/data)
│   ├── eye/              #   .artifacts/, .omx/ (бывшее eye/.artifacts, .omx)
│   └── shared/           #   (на будущее: общие настройки, логи)
├── tauri/                # общий Tauri-шелл (React: Eye+Speech секции)
├── app.ps1 / app.sh      # единый лаунчер
└── .venv/                # Python venv (Speech)
```

При старте Speech-ядра задать:

```
SPEECH_HOME      = <NewName>/speech
SPEECH_DATA_DIR  = <NewName>/data/speech
HF_HOME          = <NewName>/speech/models/huggingface
```

Eye-конфиг: `EYE_HOME_DIR` указывает на `<NewName>/data/eye`.

## 6. Порядок интеграции (чек-лист для сол)

### Фаза A — «Оба ядра работают из одного лаунчера»
1. Скопировать Eye и Speech в новый каталог (см. §5).
2. Написать оркестратор: запуск `node mcp.mjs` / `node vision.mjs`
   (Eye) и `python -m speech_app run` (Speech) с правильными env.
3. Проверить: HTTP API Speech отвечает (`/api/status`), Eye MCP-сервер
   отвечает (`eye test`).
4. Прогнать тесты Speech: `unittest discover -s speech/tests` (109 зелёных).

### Фаза B — Общий Tauri-шелл
5. Создать Tauri-проект (React 19 + Rust, по образцу обоих).
6. Перенести секции Speech (Статус/Модели/Настройки/История) из
   `speech/tauri/src/main.tsx`; добавить секции Eye (perception/overlay).
7. Rust-команды: чтение `data/speech/api.port` (discovery), HTTP-клиент к
   API Speech; команды Eye — через MCP или прямой spawn.
8. Проверить визуально: скриншот окна, переключение моделей Speech,
   «Готово»-статус, история.

### Фаза C — Установщик и CLI
9. Единый `app.ps1`/`app.sh`: создать `.venv`, поставить requirements
   (Speech: base/parakeet/whisper/gigaam; Eye: `npm install`).
10. CLI-команды: `app run`, `app status`, `app model list`,
    `app model install <key>`, `app eye test`, `app diagnose`.

### Фаза D — (опционально) MCP-мост
11. Добавить в MCP-сервер Eye инструмент `speech_transcribe`/`speech_status`
    (HTTP-прокси). Тогда агент сможет диктовать и читать историю Speech.

## 7. Риски и как их обходить

| Риск | Митигация |
|------|-----------|
| **Два рантайма** (Node + Python) в одном приложении | не сливать код; процессы изолированы, контракт — HTTP/MCP; лаунчер управляет жизненным циклом |
| **Версии зависимостей**: Speech требует transformers 5.x (git), torch CPU | держать отдельный venv; не обновлять «всё сразу» |
| **Speech полагается на путь корня** (`parents[1]`) | проверить после переноса; при необходимости заменить на env `SPEECH_HOME` (уже поддерживается) |
| **Tauri-сборка Speech** (embedded dist, `npm run tauri:build`) | в общем шелле использовать тот же паттерн: `frontendDist: ../dist`, сборка через tauri CLI, никогда `cargo build` напрямую (см. `07-TAURI.md` §4) |
| **Порты/коллизии** | API Speech — эфемерный порт (0) — коллизий нет; MCP — stdio |
| **Один экземпляр** | сохранить SingleInstanceLock Speech; добавить аналогичный для нового приложения |
| **Секреты** | key MiMo (Eye config.json) не должен попадать в git/логи (уже отмечено в HANDOFF.md Eye) |

## 8. Что спросить у владельца перед началом

1. Имя нового приложения (заменит «Eye» и «Speech»).
2. Единое окно с вкладками или несколько окон (как сейчас overlay у Eye)?
3. Нужен ли MCP-инструмент `speech_transcribe` в фазе D или пока не надо.
4. Кто будет поддерживать установку моделей Speech из общего UI (сейчас это
   трей + `speech model install`).

## 9. Полезные ссылки внутри Speech

- `docs/onboarding/00-OVERVIEW.md` — обзор
- `docs/onboarding/01-ARCHITECTURE.md` — архитектура
- `docs/onboarding/03-API.md` — HTTP-контракт (шов интеграции)
- `docs/onboarding/07-TAURI.md` — как собирать/встраивать UI
- `docs/onboarding/08-DECISIONS.md` — почему так, и грабли
