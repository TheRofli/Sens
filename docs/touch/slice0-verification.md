# Sens 1.4.0 Touch — верификация фактов слайса 0

> Зафиксировано 2026-08-13. Все факты проверены по первоисточникам;
> даты и версии пиннингуются для воспроизводимого QA.

## MCP Tasks extension — ПОДТВЕРЖДЕНО

- Официальное расширение: `modelcontextprotocol/ext-tasks` (spec),
  объявлено в спецификации MCP 2026-07-28.
- Механика: клиент объявляет `io.modelcontextprotocol/tasks` в `_meta`
  запроса; сервер рекламирует в `server/discover`; длинные операции
  возвращают `CreateTaskResult` (taskId, status, ttlMs, pollIntervalMs);
  поллинг через `tasks/get`; mid-flight input через `tasks/update`
  (статус `input_required`); отмена — `tasks/cancel` (кооперативная);
  опциональные `notifications/tasks`.
- **Consent-флоу Touch ложится на `input_required`** нативно.
- Важно: Task можно возвращать ТОЛЬКО клиенту, объявившему поддержку.
  → фолбэк `sens_touch_status`/`sens_touch_cancel` обязателен.
- **Client matrix (`modelcontextprotocol.io/extensions/client-matrix`)
  Tasks пока НЕ отслеживает** (отслеживаются только MCP Apps/OAuth/
  Enterprise) — поддержка в хостах ещё не устоялась; на слайде 1
  проверить rmcp 0.16: поддержку `server/discover` и extension
  negotiation; при отсутствии — нативный Tasks в sens-mcp реализуем
  поверх (задача слайса 1).

## Модели (пинним конкретные, не latest)

| Канал | Slug | Цена (за 1M токенов) | Контекст | Статус |
|---|---|---|---|---|
| OpenRouter | `deepseek/deepseek-v4-flash-0731` | $0.08 in / $0.252 out | 1M | с 2026-07-31 (ПОДТВЕРЖДЕНО openrouter.ai) |
| DeepSeek direct | `deepseek-v4-flash` | $0.14 in (cache miss) / $0.28 out | 1M | актуальное имя; legacy `deepseek-chat`/`deepseek-reasoner` деприкейтед 2026-07-24 (api-docs.deepseek.com) |
| DeepSeek direct | `deepseek-v4-pro` | $0.435 in / $0.87 out | 1M | актуальное имя |

- 0731-бета поддерживает нативный Responses API и адаптирована под
  Codex; для Touch используем OpenAI-compatible `/chat/completions`
  (tool calling) — поддерживается.
- Цены нужны для pessimistic cost-оценки: конфиг `provider.price_per_1k_*`
  (опционально; по умолчанию — усреднённая оценка).

## Поисковые провайдеры

- Tavily — POST `https://api.tavily.com/search` `{api_key, query}` →
  `{results: [{title, url, content}], answer}`. Форма совпадает с
  `tests/touch/fixtures/mock_search.py`.
- SerpAPI / Brave — адаптеры конфигурируются отдельно; без ключа поиска
  `web_search` честно недоступен, `web_fetch` работает (решение v1.1).

## Среда выполнения (проверено на машине автора)

- Python для тестов: `C:\Users\kanal\AppData\Local\Programs\Python\
  Python311\python.exe` (stdlib достаточно для фикстур слайса 0).
- Packaged python Sens: `%LOCALAPPDATA%\Sens\runtime\python\python.exe`.
- Git Bash: `cargo` не на PATH — `export PATH="$HOME/.cargo/bin:$PATH"`.

## Открытые пункты на слайс 1

1. rmcp 0.16: поддержка `server/discover` / extension negotiation;
   иначе — собственная реализация Tasks-методов поверх rmcp.
2. Точные дефолты token-бюджетов (50k/6k/24k/6k/2.5k) — прогон на
   реальном провайдере.
3. Подтвердить формат usage OpenRouter (поле `usage` в chat.completion).
