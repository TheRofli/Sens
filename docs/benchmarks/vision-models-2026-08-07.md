# Sens local VLM benchmark

Measured: 2026-08-07T23:06:32+0800 on AMD Ryzen 7 5800XT, Windows, llama-cpp-python 0.3.34. CPU-only; one model process at a time.

| Pack | Model | Mean task score | Total time | Peak RSS |
|---|---|---:|---:|---:|
| `lite` | `Qwen/Qwen3-VL-2B-Instruct-GGUF` | 0.900 | 45.2s | 3395 MiB |
| `quality` | `ggml-org/SmolVLM2-2.2B-Instruct-GGUF` | 0.585 | 27.7s | 3237 MiB |

## Per-task evidence

### lite — Qwen/Qwen3-VL-2B-Instruct-GGUF

- `design_exact_ocr`: score 1.000, 8.48s — Settings / Profile / Actions / faint hint / this label runs off the edge of the panel
- `design_overflow_grounding`: score 1.000, 6.00s — YES / this label runs off the edge of the panel
- `negative_hallucination`: score 1.000, 5.67s — NONE
- `russian_ui_ocr`: score 1.000, 13.40s — Возможности / Чувства модели / Выберите чувство, чтобы настроить его точное. / Зрение / Анализ изображений, OCR, поиск деталей и визуальная самопроверка. / Слух / Локальное распознавание аудиофайлов без доступа модели к микрофону. / Будущие чувства / Новые модули появятся здесь после установки — отдельно от подключения моделей.
- `poster_ui_grounding`: score 0.500, 11.66s — - "made for art lovers and art creators" / - "Mono X7 is a canvas for the digital age." /  / Product: Mono X7

### quality — ggml-org/SmolVLM2-2.2B-Instruct-GGUF

- `design_exact_ocr`: score 0.923, 5.92s — Settings /  / Profile /  / Actions /  / this label runs off the edge of the panel
- `design_overflow_grounding`: score 1.000, 5.76s — Yes, the text extends outside the bordered panel. The text "this label runs off the edge of the panel" is visible in the image.
- `negative_hallucination`: score 1.000, 3.51s — None
- `russian_ui_ocr`: score 0.000, 7.84s — Main headline: "Уведение модели в устройстве, это всегда высоко!" /  / Capability names: "Eye: MIMO-v2.5" and "Spyglass: CnG-M1-v1.0"
- `poster_ui_grounding`: score 0.000, 4.72s — The product shown in the center is a "MonoX 2.0" which is a "made for art lovers and art creators" device.

## Decision

`lite` (`Qwen/Qwen3-VL-2B-Instruct-GGUF`) is the default semantic pack because it has the highest measured Sens-task score. Deterministic OCR, geometry, color, and comparison remain the primary truth; VLM output remains explicitly inferred.

Model absence is a supported degraded state. No model is downloaded or loaded implicitly.
