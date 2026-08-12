# Sens 1.4.0 Touch — совместимость: снимок схем 1.3.8 (baseline)

> Зафиксировано на слайсе 0 (2026-08-13). Touch не должен ломать эти
> контракты; схемы аддитивные. Источник: `crates/sens-mcp/src/main.rs` и
> `crates/sens-protocol/src/lib.rs` (коммит 7601775).

## Неизменяемые элементы (трогать нельзя)

- Протокол: MCP stdio, JSON-RPC 2.0, крate `rmcp` 0.16.0 (workspace).
- Envelope брокера: `InvokeRequest` / `InvokeResult` (request_id,
  capability_id, operation, status, data, artifacts, provenance, usage,
  warnings, error, elapsed_ms).
- Реестр capabilities: `default_capabilities()` — теперь 3 записи
  (sight, hearing, touch); существующие записи не менялись, только
  добавлена touch (state `Unavailable` до регистрации координатора).
- Состояния: `AppState`, `CapabilityState`, `ConnectionState`, `JobState`,
  `Permission` — без изменений.

## Схемы слуха (операции hearing, не менять)

### `sens_hear` (operation "hear")

```jsonc
{
  "audioPath": "string (required)",
  "language": "string | null",
  "model": "string | null",
  "frames": "uint | null",
  "at": "number[] | null",
  "every": "number | null",
  "timeoutMs": "uint | null (default 180000)",
  "saveToHistory": "bool (default false)"
}
```

Правила: без clipboard/paste/history; сегменты с таймстампами; remote-модель
только по ключу пользователя (OpenRouter).

### `sens_watch` (operation "watch")

```jsonc
{
  "videoPath": "string (required)",
  "prompt": "string | null",
  "model": "string | null"
}
```

### `sens_fetch` (operation "fetch")

```jsonc
{ "url": "string (required)" }
```

## Схемы зрения, критичные для реюза в Touch

### URL-политика web_fetch (переиспользуется Touch)

- Только http(s); запрет loopback/private/link-local/multicast/reserved на
  начальной навигации, редиректах и субресурсах; лимиты размера (2 МБ) и
  времени (30 c); `data:`/`blob:`/`about:` — локальные браузерные ресурсы.
- Фикстура категорий: `tests/touch/fixtures/private_urls.json`.
- Отдельная trust boundary: endpoint провайдера из конфига пользователя НЕ
  подпадает под SSRF-политику (решение v1.1).

## Проверки совместимости на слайсе 0

- `cargo test -p sens-protocol` — 8 тестов (2 старых + 6 новых touch).
- `tests/touch/test_contracts.py` — 12 тестов (stdlib, Python 3.11).
- Старые схемы слуха не модифицированы; diff репозитория на слайсе 0
  затрагивает только: sens-protocol (новый модуль touch + манифест),
  sidecars/touch/ (роли), tests/touch/ (фикстуры), docs/touch/ (документы).
