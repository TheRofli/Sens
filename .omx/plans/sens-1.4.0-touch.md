# Sens 1.4.0 — Touch: осязание (делегирование дешёвым воркер-моделям)

> Обновлено 2026-08-13: **слайсы 0–4 реализованы, слайс 5 (dev-гейт) пройден**
> на коммите с полным прогоном: fmt, clippy `-D warnings` (workspace excl.
> sens-desktop — экологический лок runtime\python в build script), 63
> Rust-теста, 17 python-тестов. Полный дизайн: `docs/touch/touch-1.4.0-plan.md`.

## Статус реализации (2026-08-13)

- **Слайс 0** ✅ контракты v1.1, роли EN, фикстуры, верификация фактов.
- **Слайс 1** ✅ `TouchExecutor` в sens-broker: конфиг `touch` из config.json,
  job store (capacity 2, очередь FIFO, TTL, noStore, лимит хранимых),
  кумулятивные token-бюджеты, pessimistic spend-оценка + consent
  (`awaiting_consent` → `sens_touch_status(job_id, consent:true)`), дневной
  лимит, **provider proxy** (ключ только в брокере, HTTPS делает брокер,
  Debug маскирует ключ), 7 MCP-тулов в sens-mcp (fallback-путь:
  sens_touch/_parallel/_opinions/_verify/_status/_cancel/_check).
- **Слайс 2** ✅ `sidecars/touch/touch-worker.py` (agent loop, роли из
  roles/*.md, tool calling через broker, JSON-финальный ответ, честный
  partial при limit); broker-executed тулы read/glob/grep (scope check,
  canonicalization, лимиты, traversal/symlink), evidence-рецепты в момент
  чтения, верификатор claims (двухосная семантика, downgrade verified →
  inferred при невалидных рецептах).
- **Слайс 3** ✅ web_fetch (https-only + private-range/loopback rejection,
  2 МБ/30 c, рецепт с sha256), web_search (Tavily/SerpAPI/Brave, без ключа
  честно disabled), opinions (изоляция, дефолтные перспективы по ролям,
  count/N), verify, check (чистые предикаты без LLM и трат).
- **Слайс 4** ✅ coder: песочница брокера (копирование scope + зависимых
  файлов), write только в песочницу, **unified diff генерирует брокер**
  (similar), рабочее дерево primary не трогается, tests_required без
  исполнения (run_tests НЕ в v1).
- **Слайс 5 (dev)** ✅ интеграционный гейт: E2E брокер → touch-worker.py →
  mock-провайдер (полный цикл с рецептами и verified-claims), check-
  предикаты (verified/refuted/unverifiable), coder-песочница (diff + дерево
  primary нетронуто), opinions (явные и дефолтные перспективы), тесты
  «ключ не в логах/IPC/телах запросов», «auth на каждом вызове».
- **MCP Tasks**: rmcp 0.16 поддерживает Tasks-протокол нативно
  (CreateTaskResult/tasks:get/cancel в ServerResult, TaskManager). Job store
  Tasks-ready (durable job_id, статусы, TTL). Нативный мост через
  OperationProcessor rmcp — отдельный пункт релизной упаковки (не блокирует
  dev-гейт; fallback sens_touch_status/cancel работает всегда).
- **Публичная установка 1.4.0**: НЕ выпущена. Dev-линия не входит в
  установщики до релизной упаковки (версии, tauri build, sidecar-деплой,
  подписанный релиз по отмашке).

## Outcome

Sens получает четвёртое чувство — **Осязание (Touch)**: текстовый primary-агент
может делегировать самодостаточную работу дешёвым воркер-моделям с ролями,
бюджетами, изоляцией контекста и evidence-рецептами. Провайдер — любой
OpenAI-совместимый API по ключу пользователя (OpenRouter/DeepSeek). Существующие
Sight, Hearing и MCP-тулы остаются совместимыми.

Слоган (после ревью): **Sens doesn't trust worker evidence blindly. It verifies
the evidence beneath model conclusions.**

## Ключевые решения v1.1

- **Двухосная верификация**: `claim_status` (inferred — всегда для
  семантических выводов; verified — только для машинных предикатов:
  file_exists/line_contains/pattern_count/url_contains_quote/hash_matches) и
  `evidence_status` (verified/refuted/unverifiable).
- **Evidence-рецепты**: брокер выдаёт рецепт (evidence_id, sha256,
  observed_at) в момент реального чтения/докачки; воркер ссылается только на
  рецепты; race исключён; ссылка на несуществующий рецепт отклоняется.
- **Provider transport broker-owned**: ключ только в памяти брокера, HTTPS к
  провайдеру делает брокер (provider proxy); воркер не имеет ключа, сети и
  прямого FS-доступа.
- **Worker requests, Broker permits and executes**: все привилегированные
  тулы (read/glob/grep/web_fetch/web_search/sandbox write) исполняет брокер;
  воркер — agent loop, выбор тулов, синтез.
- **Coder v1 = patch producer**: песочница брокера (scope + минимальные
  зависимые файлы), unified diff генерирует брокер; **run_tests НЕ в v1**
  (allowlist не является OS-песочницей; настоящий isolation — отдельная
  будущая capability). Патч применяет только primary.
- **Async через MCP Tasks extension** (ext-tasks, 2026-07-28): CreateTaskResult
  + tasks/get/update/cancel; consent — через input_required elicitation;
  фолбэк для хостов без Tasks — sens_touch_status/sens_touch_cancel
  (consent через status(job_id, consent:true) / await status `awaiting_consent`).
- **Кумулятивные token-бюджеты** (не только output): max_total_input_tokens
  50k, max_total_output_tokens 6k, max_context_tokens 24k,
  max_tool_result_tokens 6k, max_single_tool_result_tokens 2.5k; usage —
  суммарно по всем model calls.
- **Role prompts — английские** (canonical, provider-neutral); objective —
  на языке пользователя.
- **Дефолтные перспективы opinions заморожены** (researcher/critic/reviewer
  по 3 шт.; если primary передал perspectives — только они).
- **sens_touch_check** — 7-й тул: чистые предикаты без LLM и трат.
- Имена моделей: OpenRouter `deepseek/deepseek-v4-flash-0731` (пиним slug),
  DeepSeek direct `deepseek-v4-flash` (legacy deepseek-chat не использовать).
- Provider endpoint — пользовательская trust boundary, SSRF-политика к нему
  НЕ применяется (в отличие от model-controlled web_fetch).
- UI-предупреждение: «Remote Touch providers may receive portions of files
  within the scope you delegate to workers».

## Не-цели

- Исполнение команд/тестов в v1 (run_tests убран); настоящий OS sandbox —
  1.4.x/1.5.
- Применение патчей к рабочему дереву автоматически.
- max_depth > 1, мультипровайдерный роутинг, adaptive router.
- Восстановление job'ов после рестарта брокера.
- Секреты/credentials/сеть у воркеров — никогда.
- Голос/медиа (1.5.0), интеграция с UMELO (отдельный трек).

## Слайсы (TDD, гейты в docs/touch/touch-1.4.0-plan.md §20–21)

- **Слайс 0** — контракты v1.1 (TaskPacket, WorkerResult с двумя осями,
  рецепты, 7 тулов, роли EN, манифесты), проверка MCP Tasks spec + client
  matrix + актуальные slug'и моделей, фикстуры (грязный репо, injection,
  несуществующий рецепт, мок провайдера, private-range, изменение файла
  между шагами).
- **Слайс 1** — TouchCoordinator: key holder, job store (capacity 2, TTL,
  noStore), scheduler, кумулятивные бюджеты, consent; **provider proxy**;
  **MCP Tasks мост** + фолбэк; IPC (model_request/tool_request/рецепты).
- **Слайс 2** — воркер-brain (loop, feature-detect tool calling) +
  broker-executed тулы (read/glob/grep, scope check, canonicalization) +
  evidence-рецепты + claim verifier (предикаты).
- **Слайс 3** — web_fetch/web_search в брокере, opinions (изоляция,
  дефолтные перспективы, синтез), sens_touch_verify, sens_touch_check.
- **Слайс 4** — coder: песочница (scope + зависимости), write-тул, diff
  брокером, tests_required без исполнения.
- **Слайс 5** — интеграционный гейт 1.4.0: E2E с реальным ключом,
  explorer → opinions → coder → verify → check; бюджеты/отмена/consent
  (оба пути); ключ не в логах/IPC; обновить docs/current-state.md.

## Обязательные гейты

cargo fmt/clippy/test workspace; pytest воркера; тесты: ключ не в
stdout/stderr/logs/activity/IPC/WorkerResult; воркер без сети; ссылка на
несуществующий рецепт отклоняется; cumulative tokens; partial при лимитах;
consent (Tasks и фолбэк); семантический claim не может стать verified;
claims без рецептов — unverifiable.
