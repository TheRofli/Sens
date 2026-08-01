# 07 — Tauri-оболочка

## 1. Роль

Tauri — **единственное GUI-окно** Speech (tkinter-окно удалено; остаётся
только скрытый root + overlay). Это отдельный процесс, который общается с
Python-ядром через локальный HTTP API (см. `03-API.md`).

```
tauri/
├── src/main.tsx        # весь React UI (~900 строк, Layout B)
├── src/styles.css      # стили
├── package.json        # vite + react 19 + @tauri-apps/api
├── index.html
└── src-tauri/
    ├── Cargo.toml      # serde, serde_json, reqwest (blocking), tauri 2
    ├── tauri.conf.json # frontendDist: "../dist", devUrl: 127.0.0.1:1420
    └── src/main.rs     # Rust: HTTP-клиент + Tauri-команды (~400 строк)
```

## 2. React UI (`tauri/src/main.tsx`)

Layout B: сайдбар + 4 секции.

- **Статус**: статус движка (Готово/Остановлено/Загрузка), активная модель,
  последний транскрипт, метрики (устройство, хоткей).
- **Модели**: карточки всех пресетов из `/api/models` — label, описание,
  размер, installed-бейдж, кнопки Load/Unload/Install, переключение активной.
- **Настройки**: аккордеон; toggles (engine_enabled, copy/paste, preload),
  поля качества (beam_size, temperature, repetition_penalty, vad_sensitivity,
  postprocess_text) — автосохранение через `POST /api/settings`.
- **История**: список `/api/history`, поиск по `q=`, кнопка copy на запись.

Механика:
- поллинг `/api/status` каждые 2 с (`refresh()` в `App`),
- `toCamelSettings(values)` — snake→camel перед отправкой в Rust,
- `offlineSnapshot` — фолбэк, если ядро недоступно (UI показывает
  «Speech core offline»), чтобы окно не выглядело сломанным без ядра.

## 3. Rust (`tauri/src-tauri/src/main.rs`)

### Discovery пути к ядру — КРИТИЧНО

```rust
fn speech_root() -> PathBuf {
    // 1. current_exe → 4 родителя вверх (release/target/src-tauri/tauri)
    //    → проверка: <root>/data/api.port или <root>/speech_app существует
    // 2. fallback: env!("CARGO_MANIFEST_DIR") → 2 родителя
    // 3. fallback: D:\Speech
}
```

Раньше использовался только `env!("CARGO_MANIFEST_DIR")` (compile-time путь)
— на другой машине/установке exe не находил ядро, и весь UI падал в
offline. Теперь путь резолвится относительно exe (коммит 41c5bc2).

### API-мост

```rust
fn api_base() -> Option<String>   // читает data/api.port → "http://127.0.0.1:<port>"
fn api_get<T>(path) / api_post<T>(path, body)  // reqwest::blocking, timeout 8s
```

Команды (все — тонкие обёртки над HTTP):
`app_snapshot`, `get_settings`, `save_settings`, `get_models`, `select_model`,
`load_model`, `unload_model`, `install_model`, `recent_history`,
`copy_history_item`, `copy_last`, плюс legacy `speech_status`,
`speech_diagnose`, `speech_restart`, `speech_stop` (шелят в `bin/speech.cmd`).

### Сериализация — КРИТИЧНО (три стороны)

Python отдаёт **snake_case**; React ждёт **camelCase**. Rust-структуры:

```rust
#[derive(Serialize, Deserialize, Debug)]
#[serde(rename_all(serialize = "camelCase", deserialize = "snake_case"))]
struct StatusSnapshot { running: bool, engine_enabled: bool, /* ... */ }
```

`SettingsPayload` — зеркально:
`#[serde(rename_all(serialize = "snake_case", deserialize = "camelCase"))]`.

Три поля на `StatusSnapshot` (`historyCount`, `speechRoot`, `modelSnapshot`)
— client-side дополнения, отсутствуют в ответе ядра: у них `#[serde(default)]`.

**Симптом поломки:** «error decoding response body: missing field
engineEnabled» в логах ядра (или «всё остановлено» в UI). Это значит, что
case-маппинг сломан в одной из сторон.

### Оффлайн-фолбэк

`offline_snapshot()` — если ядро не запущено, `app_snapshot` возвращает
фолбэк, и UI показывает «Speech core offline» вместо пустого экрана.

## 4. Сборка — ТОЛЬКО через `npm run tauri:build`

```powershell
cd tauri
npm install
npm run tauri:build
```

**Критично:** `cargo build --release` напрямую **НЕ встраивает** React-фронтенд
в бинарник — exe тогда грузит `devUrl` (http://127.0.0.1:1420) и показывает
«Hmmm... can't reach this page / ERR_CONNECTION_REFUSED». Встраивание
`dist/` в exe делает `tauri build` (через `npm run tauri:build`):
перед компиляцией он гоняет `npm run build` (tsc + vite), а
`frontendDist: "../dist"` встраивается в бинарник.

Сборка даёт:
- `src-tauri/target/release/speech-tauri.exe` — exe с embedded dist
- `src-tauri/target/release/bundle/msi/Speech_0.1.0_x64_en-US.msi`
- `src-tauri/target/release/bundle/nsis/Speech_0.1.0_x64-setup.exe`

**Запуск ядра → окно:** `SpeechApp._show_primary_window` →
`SystemActions.open_tauri_ui(speech_root)` — находит release exe и запускает
его; если exe нет — fallback-уведомление «Build Tauri».

## 5. Известные грабли (история)

| Симптом | Причина | Фикс (коммит) |
|---------|---------|----------------|
| UI показывает «всё остановлено», модели не видны, история пуста | `speech_root()` на compile-time `CARGO_MANIFEST_DIR` → Rust не находил `data/api.port` → все команды падали → React уходил в offlineSnapshot | `speech_root()` через `std::env::current_exe()` + parent×4 (41c5bc2) |
| `ERR_CONNECTION_REFUSED` на 127.0.0.1:1420 в окне | сборка через `cargo build` напрямую → exe грузит devUrl вместо embedded dist | собирать только через `npm run tauri:build` |
| `missing field engineEnabled` (декод ошибки) | `#[serde(rename_all = "camelCase")]` на всей структуре — Python отдаёт snake_case | `rename_all(serialize="camelCase", deserialize="snake_case")` (41c5bc2) |

## 6. Проверка после изменений

1. `npm run tauri:build` — должна пройти целиком (tsc → vite → cargo → msi/nsis).
2. Запустить ядро: `speech`.
3. Запустить exe из `target/release/speech-tauri.exe` (НЕ dev-режим).
4. Проверить: статус «Готово», модели видны, переключение модели работает,
   история видна, настройки сохраняются.
5. Скриншот окна для визуальной проверки (при возможности).
