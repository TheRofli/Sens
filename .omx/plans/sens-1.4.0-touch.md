# Sens 1.4.0 — Touch: осязание (делегирование дешёвым воркер-моделям)

> Обновлено 2026-08-13 (v1.1): интегрированы все 15 пунктов внешнего ревью
> (GPT-5.6 SOL). Полный дизайн-документ: `docs/touch/touch-1.4.0-plan.md`.

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
