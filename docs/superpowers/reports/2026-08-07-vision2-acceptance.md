# Sens Vision 2.0 — отчёт приёмки

Дата: 2026-08-07
План: `docs/superpowers/plans/2026-08-06-sens-vision2.md` (11 тасков)
Спека: `docs/superpowers/specs/2026-08-06-sens-vision2-design.md`
Статус: **выполнено; приёмка пройдена с двумя зафиксированными открытыми вопросами качества моделей** (по плану — не блокируют релиз)

## Что сделано (коммиты)

| Коммит | Содержание |
|---|---|
| `15473f2` | поправки pre-flight в план |
| `340e690` | иерархия секций/элементов, SoM, screen summary (+596, как есть) |
| `942ac30` | рефакторинг sight-worker.py (2411 строк) в пакет `sight/` (8 модулей), entrypoint-шим |
| `b6973ae` | ascii-карта + pytest-инфраструктура |
| `2ff11f4` | детекторы кругового/вертикального декоративного текста |
| `66ec905` | DTCG design tokens |
| `135521b` | сборщик документа + markdown-рендер |
| `2a0c921` | lazy llama-cpp VLM-хост (lite/quality) + даунлоадер |
| `c78f85a` | see→документ; операции zoom/ask/element/warm; cache qa9 |
| `508c58d` | Rust: манифест sens-protocol, 6 MCP-инструментов, валидация, warm при старте |
| `cfd0ea1` | url-capture: CSS-анимации + frame-diff motion |
| `4540a2d` | QA-рефрейм facts + legacy issues; cache qa10 |

## Критерии приёмки (спека §10)

### 1. Круговой текст Mono X7 — ⚠️ частичное (открытый вопрос, не блокирует)

`detect_circular` на синтетике работает (тест: 12 боксов по окружности → группа с радиусом).
На фикстуре `…-02.png` RapidOCR отдаёт круговую надпись как **широкие горизонтальные
боксы-строки** (`L4e4b'K8, WEb5+uN-20` и пр., ~600px шириной), а не глиф-боксы по
окружности → геометрическому детектору нечего фитить, `circular: []`. Это ограничение
гранулярности OCR на данной фикстуре, не баг детектора. `quality=true` детектор не
починит (меняется только VLM-модель). Варианты на будущее: детектор «мусорных»
текстовых кластеров (низкий confidence + не-словарные токены) как кандидатов на
VLM-транскрипцию, либо SoM-подсказка региона вручную через `sens_ask`.

Вертикальный декоративный текст при этом детектирован (`ids [3,10,16,18,23]` — лента
"Illustration, Creative Coding, Web Experiments, Art, Design"), транскрипция lite-моделью
галлюцинирует (500M — слабый предел; quality-пак moondream2-F16 ~3.7 ГБ не скачивался —
opt-in, см. ниже).

### 2. Dot-волна Hyperstudio — ⚠️ частичное

Детерминированно регион пойман как `texture`-блок `[0,428,2552,1208]` (source=measured).
VLM-caption региона — «stylized abstract cityscape / geometric shapes»: семантика есть,
но без явных «dot/matrix/wave»-дескрипторов (слабость SmolVLM2-500M). Не просто `texture`
в документе: графика получает подпись, а не голый тип.

### 3. Вайб-дескрипторы — ✅

- Mono X7: «bold, geometric aesthetic, reminiscent of **Brutalism or Retro** design» ✓
- Постер Summer Drive: «**Brutalism or Retro** design… nostalgia for a bygone era» ✓
- Hyperstudio: «bold, **industrial** aesthetic… **grays, blacks**» (dark-minimal) ✓

### 4. «MONO X7», не «MONIOXT» — ⚠️ зафиксировано фактическое состояние

OCR в `legacy.ocr` остаётся «MONIOXT»/«MONO X?» (детерминированный слой честен).
Документ не содержит ложного «исправления»: circular/проблемные зоны без уверенной
транскрипции остаются как есть. Связное «MONO X7» в тексте присутствует в OCR-строке
«Mono X7 is a canvas for the digital age» и «Mono X7: …» (body-текст).

### 5. Измерения без оценок — ✅

`see` возвращает `doc.measurements` (16 фактов на Mono X7), секция называется
«измерения (факты, НЕ оценки)», слова «issues» в документе нет; `legacy.design` хранит
старую форму `issues` для совместимости; `verification` не потерял конфликты
(text_overflow/card_alignment) после рефрейма.

### 6. sens_motion — ✅

- dope.security: 6 keyframes, 117 CSS-правил анимаций, 96 frame-diff событий, шрифты
  (Whyte Inktrap), фон `rgb(9,9,9)`;
- caldera.xyz: 1 keyframe (`__framer-loading-spin`), 11 правил, 7 live-анимаций,
  32 frame-diff события.

### 7. e2e «воспроизведение» (DeepSeek) — материалы готовы, прогон на стороне потребителя

- `docs/superpowers/reports/e2e/vision-prompt-snippet.txt` — системный сниппет;
- `docs/superpowers/reports/e2e/poster-summer-drive-document.md` — документ постера.

Сценарий: сниппет + документ → DeepSeek v4 Flash генерирует HTML/CSS-копию → рендер
playwright (viewport 2048×1152) → `sens_compare` с оригиналом → mismatch% в отчёт.
Порог калибруется первым прогоном (спека §10).

## Гейты

- Python: `pytest tests/sight -q` → **14 passed**.
- Rust: `cargo fmt --check`, `cargo clippy --workspace --all-targets -- -D warnings`,
  `cargo test --workspace` → **зелёное** (sight: 10 тестов валидации, из них 5 новых).

## Деплой

- `sidecars/sight/` + `sight-worker.py` + `models/` (lite-пак 636 МБ) скопированы в
  `AppData\Local\Sens`; воркер перезапущен; приёмка прогонена на **задеплоенной копии**.
- `target/release/sens-mcp.exe` и `sens-broker.exe` пересобраны (Z-Code-коннектор
  указывает на `D:\Sens\target\release\sens-mcp.exe` — новые инструменты активны после
  переподключения MCP-хоста). Копия бинарников установленной десктоп-приложения
  (`AppData\Local\Sens\sens-*.exe`) залочена запущенным процессом — обновится при
  следующем рестарте/переустановке приложения.

## Открытые вопросы / follow-ups

1. **Паки семантики (обновлено 2026-08-07 по результатам ресёрча квантов):**
   - `lite` (default): SmolVLM2-500M Q8, ~0.7 ГБ RAM;
   - `quality`: **SmolVLM2-2.2B-Instruct Q4_K_M** (ggml-org) + mmproj Q8_0, ~1.7 ГБ диск /
     ~2 ГБ RAM — новее и сильнее moondream2, в бюджете спеки. Скачан и проверен:
     vibe/caption заметно лучше lite;
   - `quality_large`: **Qwen2.5-VL-3B-Instruct Q4_K_M** (unsloth) + mmproj F16,
     ~3.3 ГБ диск / ~3.5 ГБ RAM — лучший по качеству (особенно OCR), скачивание
     по требованию: `scripts/download-vision-models.py --pack quality_large`,
     включение: `see(pack="quality_large")` (поле `pack` проброшено в MCP-инструменты).
   - Прежний вариант quality (moondream2 F16 ~3.75 ГБ) удалён из PACKS: F16 тяжёл,
     а Q4-кванты moondream2 (salivosa/moondream2-gguf) — модель 2024 года, слабее
     SmolVLM2-2.2B/Qwen2.5-VL.
2. **Круговой текст**: нужен детектор «мусорных OCR-кластеров» или региональная подсказка
   (см. критерий 1). Транскрипция вертикальной ленты Mono X7 не даётся даже 2.2B
   (повёрнутый текст) — кандидат на проверку `quality_large`.
3. **Исправлена гонка загрузок**: фоновый warm-тред (lite) и обработчик запроса
   (quality) могли строить два GGUF одновременно — процесс молча умирал. Загрузки
   сериализованы глобальным `_LOAD_LOCK` в `vlm.py`.
4. **Деплой бинарников десктоп-приложения**: при следующей сборке инсталлятора новые
   брокер/MCP попадут в AppData автоматически.
5. `eye-worker.mjs` в деплое отличается от репо (доведено до сведения; в этом изменении
   не трогался).
