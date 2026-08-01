# 03 — Локальный HTTP API

## 1. Общее

- Сервер: `speech_app/api.py`, stdlib `http.server.ThreadingHTTPServer`.
- Адрес: `127.0.0.1:<ephemeral_port>` — порт выбирает ОС при старте и
  записывается в `data/api.port` (текстовый файл, одна строка — число).
- Клиент (Tauri) читает порт из файла → `http://127.0.0.1:<port>`.
- Все ответы — JSON, `Content-Type: application/json; charset=utf-8`,
  `Cache-Control: no-store`.
- POST-запросы: `Content-Type: application/json`, тело — JSON-объект.
- Ошибки: 400 (invalid json / missing field), 404 (нет такого пути),
  500 (исключение внутри, текст в `{"error": ...}`), 202 (принято в работу).

**Правило потокобезопасности:** GET-ы читают immutable-снапшоты напрямую;
все мутации (POST) выполняются на UI-потоке через
`app.post_ui_sync(callback, timeout=5.0)` — иначе tkinter упадёт.

## 2. Эндпоинты

### GET /api/status
Снапшот состояния ядра:
```json
{
  "running": true,
  "engine_enabled": true,
  "model_state": "loaded",            // "unloaded" | "loading" | "loaded"
  "model": "gigaam",                  // preset key
  "model_label": "GigaAM v3 (русский, точная)",
  "model_loaded": true,
  "model_loading": false,
  "model_installed": true,
  "model_size_label": "428.4 MB",     // "Not installed" если не установлена
  "transcribing": false,
  "device": "cpu",
  "backend": "auto",
  "status_text": "Engine on | GigaAM v3 (русский, точная) loaded | cpu"
}
```

### GET /api/settings
Текущие настройки (subset из `AppSettings`):
```json
{
  "model": "gigaam",
  "engine_enabled": true,
  "copy_to_clipboard": true,
  "paste_to_active_input": true,
  "preload_model": true,
  "device": "cpu",
  "backend": "auto",
  "hotkey": "ctrl+win",
  "beam_size": 5,
  "temperature": 0.0,
  "repetition_penalty": 1.0,
  "no_repeat_ngram_size": 0,
  "vad_sensitivity": 0.02,
  "postprocess_text": true
}
```

### POST /api/settings
Merge-обновление: тело = любые из ключей выше. Неизвестные ключи
игнорируются. Применяется синхронно на UI-потоке.
```json
// request
{"vad_sensitivity": 0.03, "model": "whisper-ru"}
// response 200
{"ok": true, "settings": { ...полный снимок после применения... }}
```
Побочные эффекты: смена `hotkey` → перезапуск слушателя; смена `model` →
unload текущей модели (перезагрузка lazy).

### GET /api/models
Все пресеты с установочным статусом:
```json
[
  {"key": "parakeet", "label": "Parakeet (быстрая)", "engine": "parakeet",
   "model_id": "nvidia/parakeet-tdt-0.6b-v3",
   "description": "600M, мультиязычная, быстрая на CPU.",
   "installed": true, "size_label": "2.34 GB", "active": false},
  {"key": "whisper-ru", "...": "..."},
  {"key": "gigaam", "...": "..."}
]
```

### POST /api/model
```json
{"key": "gigaam"} → 200 {"ok": true, "model": "gigaam"}
```

### POST /api/model/load
Загрузить активную модель в фоне (`load_model_background`). Ответ сразу,
статус наблюдать через /api/status. → `{"ok": true}`

### POST /api/model/unload
Выгрузить. → `{"ok": true}`

### POST /api/model/install
```json
{"key": "gigaam"}  // key опционален — по умолчанию активная модель
```
Запускает установку в фоновом потоке, отвечает 202 сразу:
`{"ok": true, "key": "gigaam", "message": "installing"}`.
Прогресс: `model_installed` в /api/status флипается после записи маркера.

### GET /api/history
```text
/api/history?limit=80&q=текст
```
```json
[{"id": "366c3030-...", "text": "То есть получается в приложении..."}]
```
`limit` — максимум записей (default 80), `q` — подстрока поиска (case-insensitive).

### POST /api/history/copy
```json
{"id": "366c3030-..."} → 200 {"ok": true}
```
Копирует текст записи в клипборд.

### POST /api/action/copy_last
Копирует последний транскрипт. → `{"ok": true}`

## 3. Формат имён полей — КРИТИЧНО

Python-ядро отдаёт **snake_case** (`engine_enabled`, `model_state`,
`size_label`, `compression_ratio_threshold`...).

Rust-структуры используют
`#[serde(rename_all(serialize = "camelCase", deserialize = "snake_case"))]` —
то есть **Rust десериализует snake_case из Python и сериализует camelCase
в React** (`engineEnabled`, `modelState`, `sizeLabel`...).

`SettingsPayload` — зеркально: десериализация camelCase (от JS),
сериализация snake_case (в Python).

**Если добавишь поле в API — обнови все три стороны**: Python
(`app.get_settings_values` / `_status`), Rust (`StatusSnapshot` /
`SettingsPayload` / `ModelInfo` / `HistoryItem`) и React (`StatusSnapshot`
интерфейс в `main.tsx`). Пропуск одной стороны = «missing field» ошибка.

## 4. Кто клиенты

- **Tauri** (`tauri/src-tauri/src/main.rs`): `app_snapshot`, `get_settings`,
  `save_settings`, `get_models`, `select_model`, `load_model`, `unload_model`,
  `install_model`, `recent_history`, `copy_history_item`, `copy_last` —
  тонкие обёртки над HTTP.
- Любой локальный инструмент может читать/писать тот же API (это удобно для
  интеграции: например, Eye-агент сможет управлять Speech через HTTP без
  знания Python).

## 5. Discovery для внешних клиентов

```text
data/api.port   → порт
data/runtime_state.json → {"running": true, "model_state": "...", "device": "...",
                            "backend": "...", "last_error": "", "updated_at": "..."}
```
`runtime_state.json` — для внешних наблюдателей, которым не нужно дёргать
HTTP (например, скрипты/агенты проверяют «Speech запущен?»).
