# 06 — Пайплайн диктовки и качество текста

## 1. Пайплайн (по шагам)

```
микрофон (16k mono float32)
   │  AudioRecorder (sounddevice InputStream)
   ▼
сырые сэмплы [0..N]
   │  trim_silence(samples, sr, vad_sensitivity)      [speech_app/vad.py]
   ▼
обрезанный буфер (или пусто → "No speech detected")
   │  engine.transcribe(trimmed, sr, settings)         [выбранная модель]
   ▼
raw text
   │  postprocess(raw) if postprocess_text else raw    [speech_app/textpost.py]
   ▼
чистый текст
   │  TranscriptPublisher.publish(text, settings)      [speech_app/output.py]
   ▼
история (history.jsonl) + клипборд + Ctrl+V в активное поле
```

## 2. VAD — обрезка тишины (`speech_app/vad.py`)

RMS-энергия по кадрам ~30 мс. Кадр «речь», если RMS ≥ `sensitivity`.
Возвращается диапазон от первого до последнего «речевого» кадра (внутренние
паузы сохраняются — для push-to-talk этого достаточно).

- `sensitivity=0.02` (default) — порог; меньше = более чувствительно.
- `min_duration_s=0.3` — если после обрезки короче 300 мс → пустой массив
  (значит, речи не было; щелчок клавиши не пойдёт в модель).
- Пустой результат → публикуется пустая строка → notice «No speech detected».

**Зачем:** главная защита от галлюцинаций Whisper на почти пустом аудио
(клики клавиш при нажатии/отпускании хоткея, дыхание).

## 3. textpost — детерминированная чистка (`speech_app/textpost.py`)

Порядок операций:

1. `_normalize_whitespace` — схлопывание переводов строк и множественных
   пробелов (включая NBSP/Unicode-пробелы).
2. `_strip_hallucinations` — удаление известных Whisper-фраз-галлюцинаций
   («thank you for watching», «please subscribe», «[music]», одиночные
   «so», «you» и т.д.) по границам фраз, case-insensitive.
3. Удаление ведущей пунктуации (`-–—.•,;:!?`).
4. Капитализация первого буквенного символа.

Язык-агностичный (RU/EN безопасно). Консервативный: короткие слова вроде
«you» удаляются только как самостоятельные фразы на границах, не внутри
предложения.

## 4. Настройки качества (из `AppSettings`)

| Параметр | Дефолт | Применяется к | Смысл |
|----------|--------|---------------|-------|
| `beam_size` | 5 | Parakeet (num_beams), Whisper (beam_size) | ширина beam-поиска |
| `temperature` | 0.0 | Parakeet (do_sample), Whisper | 0 = жадная декодировка |
| `repetition_penalty` | 1.0 | Parakeet | >1 штрафует повторы |
| `no_repeat_ngram_size` | 0 | Parakeet | запрет повтора n-грамм |
| `compression_ratio_threshold` | 2.4 | Whisper | отбрасывает «мусорные» сегменты (анти-галлюц.) |
| `log_prob_threshold` | -1.0 | Whisper | порог лог-вероятности (анти-галлюц.) |
| `vad_sensitivity` | 0.02 | все (до модели) | порог VAD |
| `postprocess_text` | true | все (после модели) | вкл/выкл textpost |

Whisper дополнительно всегда транскрибирует с
`condition_on_previous_text=False` (не кормит модель своим прошлым выводом —
главный источник зацикленных фраз на тишине) и `vad_filter=True` (встроенный
Silero VAD как страховка).

## 5. Публикация (`speech_app/output.py`)

`publish(text, settings)`:

1. `text.strip()`; пусто → `None` (caller показывает «No speech detected»).
2. `history.add(text)` — запись в `data/history.jsonl` (UUID, cap 100).
3. если `copy_to_clipboard` → `system.copy_to_clipboard(text)`.
4. если `paste_to_active_input` → `system.paste_into_active_input(text)`.

## 6. Вставка в активное окно (`speech_app/system.py`)

`SystemActions` — Win32-механика, из-за которой диктовка ощущается «магией»:

1. `remember_active_window()` вызывается **до** начала записи? — нет: текущая
   реализация запоминает окно в момент старта диктовки неявно — на самом
   деле `_begin_recording` вызывает `release_hotkey_modifiers()`, а
   remember/restore происходят в `paste_into_active_input` через
   `_remember_cursor_position()` + `_click_target_point()`. См. код `system.py`:
   - `copy_to_clipboard` — pyperclip, fallback tkinter clipboard.
   - `paste_into_active_input` — Ctrl+V через `keybd_event` (или pynput).
   - фокус: `AttachThreadInput` → `SetForegroundWindow(hwnd)` → клик в точку,
     где был курсор.
   - `release_hotkey_modifiers` — отпускает Win/Alt/Ctrl перед вставкой,
     чтобы Ctrl+V не сочетался с зажатым хоткеем.

## 7. Как отлаживать качество

- Проверить VAD: `vad_sensitivity` вверх/вниз; пустой результат = слишком
  агрессивный порог.
- Whisper-галлюцинации → `condition_on_previous_text=False` (уже), пороги.
- Сравнение моделей: переключать в трее Model → диктовать одну и ту же фразу.
- GigaAM: лучший русский + пунктуация из коробки; Whisper: лучший RU+EN
  смешанный; Parakeet: самый быстрый.
