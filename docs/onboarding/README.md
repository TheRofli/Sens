# Speech — Onboarding Pack

> **Для кого:** агент, который берёт Speech в работу (в перспективе — как
> модуль внутри объединённого проекта с Eye).
>
> **Как читать:** начни с `00-OVERVIEW.md`, затем `01-ARCHITECTURE.md`.
> Остальные файлы — справочники по конкретным подсистемам. Всё проверено по
> состоянию репозитория на коммит `5e6dd3d` (GigaAM добавлен).

## Оглавление

| Файл | Что внутри |
|------|-----------|
| [00-OVERVIEW.md](00-OVERVIEW.md) | Что такое Speech, сценарий использования, стек, запуск за 5 минут |
| [01-ARCHITECTURE.md](01-ARCHITECTURE.md) | Полная архитектура: компоненты, потоки данных, модель потоков, жизненный цикл |
| [02-ENGINES.md](02-ENGINES.md) | Система ASR-движков: Parakeet / Whisper / GigaAM, как добавить 4-ю модель |
| [03-API.md](03-API.md) | Локальный HTTP API (для Tauri и любых локальных клиентов) — полная спецификация |
| [04-THREADING.md](04-THREADING.md) | Многопоточность и потокобезопасность: tkinter UI-поток, `post_ui`, `post_ui_sync`, ловушки |
| [05-LAUNCHERS-CLI.md](05-LAUNCHERS-CLI.md) | Лаунчеры (speech.ps1 / speech.sh / bin), CLI-команды, portable-окружение, установка моделей |
| [06-PIPELINE.md](06-PIPELINE.md) | Пайплайн диктовки: VAD → ASR → textpost → публикация, настройки качества |
| [07-TAURI.md](07-TAURI.md) | Tauri-оболочка: React UI, Rust-команды, discovery пути к ядру, сборка |
| [08-DECISIONS.md](08-DECISIONS.md) | История решений и «почему именно так» — с граблями, на которые мы наступили |
| [09-INTEGRATION-EYE.md](09-INTEGRATION-EYE.md) | **План интеграции Speech в объединённый проект (Eye)** — контракты и точки входа |
| [10-GLOSSARY.md](10-GLOSSARY.md) | Термины, сокращения, layout файлов на диске |

## Пара слов от владельца

Speech — локальное push-to-talk приложение для диктовки. Удерживаешь хоткей,
говоришь, отпускаешь — текст вставляется в активное поле, копируется в
буфер и сохраняется в локальную историю. Всё распознавание — на машине,
никакого облака.

В проекте **три ASR-модели** (переключаются из трея/UI): Parakeet (быстрая),
Whisper RU-codeswitch (точная для смешанного RU+EN), GigaAM v3 (лучший
русский, с пунктуацией). Движок — Python + tkinter (скрытый root + overlay),
трей — pystray, горячие клавиши — pynput, аудио — sounddevice, GUI-окно —
Tauri 2 (React 19 + Rust) поверх локального HTTP API.

В перспективе Speech станет модулем большого приложения вместе с Eye
(визуальное восприятие для агентов). См. `09-INTEGRATION-EYE.md`.

## Быстрый старт (для агента)

```powershell
# Показать статус установки всех моделей
speech model list

# Установить конкретную модель
speech model install gigaam

# Запустить ядро (трей + API + hotkey-слушатель)
speech

# Прогнать тесты
.\.venv\Scripts\python.exe -m unittest discover -s .\tests
```

Полезное: `speech --diagnose` — проверка зависимостей; `data/api.port` —
порт локального API, который читает Tauri.
