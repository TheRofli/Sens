# 04 — Потоки и потокобезопасность

## 1. Карта потоков

| Поток | Кто | Что делает |
|-------|-----|-----------|
| **UI-поток** (main) | tkinter `root.mainloop()` | ВСЁ состояние приложения, overlay, tray-колбэки, post_ui-очередь |
| tray-поток | pystray (внутри `run_detached`) | системный трей; колбэки меню выполняются здесь |
| hotkey-поток | pynput Listener | глобальные клавиши; on_start/on_stop здесь |
| audio-поток | sounddevice callback | захват микрофона (реального потока нет — callback из PortAudio) |
| worker (транскрипция) | `threading.Thread` daemon | VAD + engine.transcribe + textpost (тяжёлая работа) |
| worker (загрузка модели) | `threading.Thread` daemon | `engine.load(settings)` (скачивание/инициализация весов) |
| API-сервер | `ThreadingHTTPServer.serve_forever` daemon | HTTP (по потоку на запрос) |
| install | `threading.Thread` daemon | установка модели |

**Главное правило: tkinter-объекты (root, overlay, Canvas) трогать можно
только из UI-потока.** Всё остальное — через очередь.

## 2. post_ui — асинхронный переход на UI-поток

```python
def post_ui(self, callback: Callable[[], None]) -> None:
    self.ui_queue.put(callback)
```

`_pump_ui_queue` — `root.after(30, ...)`, разбирает очередь каждые 30 мс и
выполняет колбэки на UI-потоке. Любой другой поток вызывает
`app.post_ui(lambda: ...)` и не ждёт.

## 3. post_ui_sync — синхронный вызов на UI-потоке

```python
def post_ui_sync(self, callback, timeout: float = 5.0) -> object:
    done = threading.Event()
    box: dict[str, object] = {}
    def runner():
        try:
            box["result"] = callback()
        except BaseException as exc:
            box["error"] = exc
        finally:
            done.set()
    self.ui_queue.put(runner)
    if not done.wait(timeout):
        raise TimeoutError("UI thread did not service the callback in time")
    if "error" in box:
        raise box["error"]
    return box.get("result")
```

Используется **HTTP API-сервером** для мутаций (`POST /api/settings`,
`/api/model`, ...). Гарантии:
- callback выполняется на UI-потоке (tkinter-safe),
- ошибки из callback пробрасываются вызывающему потоку,
- таймаут 5 с — защита от зависшего UI.

**Не вызывать `post_ui_sync` из самого UI-потока** — deadlock (UI ждёт
события, которое сам же должен поставить). В UI-потоке просто вызывай метод
напрямую.

## 4. Загрузка модели — паттерн

```
[любой поток] load_model_background()
  ├─ если engine.is_loaded → no-op (уведомление)
  ├─ если model_loading → no-op
  ├─ model_loading = True; post_ui("loading" notice)
  └─ Thread(_load_model_worker).start()
        └─ engine.load(settings)          [фоновый поток, может быть долго]
        └─ post_ui(_model_load_succeeded) [всегда, через post_ui]
        └─ (или post_ui(_model_load_failed))
```

`_model_load_succeeded`/`_model_load_failed` сбрасывают `model_loading=False`
на UI-потоке. Флаги (`model_loading`, `transcribing`) — обычные атрибуты
объекта; **мутировать их можно только с UI-потока** (все внешние пути идут
через post_ui).

## 5. Транскрипция — паттерн с гарантией сброса флага

```python
def _transcribe_worker(self, samples, sample_rate, settings_snapshot):
    text = ""
    failed: tuple[str, Exception] | None = None
    try:
        trimmed = trim_silence(...)
        if trimmed.size == 0:
            self.post_ui(lambda: setattr(self, "transcribing", False))
            self.post_ui(lambda: self._publish_transcript("", settings_snapshot))
            return
        raw_text = self.engine.transcribe(trimmed, sample_rate, settings_snapshot)
        text = postprocess(raw_text) if settings_snapshot.postprocess_text else ...
    except EngineUnavailable as exc:
        failed = ("Engine unavailable", exc)
    except Exception as exc:
        failed = ("Transcription failed", exc)

    # ВСЕГДА сбрасываем флаг — overlay не должен залипнуть
    self.post_ui(lambda: setattr(self, "transcribing", False))

    if failed is not None:
        title, exc = failed
        # eager binding — иначе late-binding lambda словит NameError
        self.post_ui(lambda t=title, e=exc: self._show_error(t, e))
        return
    self.post_ui(lambda: self._publish_transcript(text, settings_snapshot))
```

**Грабли, которые уже были (коммит 90755ff):**
- lambda `lambda: self._show_error(title, exc)` внутри except замыкает
  переменные **поздно** — если `title`/`exc` перезаписаны (или не
  определены в момент выполнения на UI-потоке), ловится NameError, и
  реальная ошибка теряется. Правильно: `lambda t=title, e=exc: ...`.
- Сброс `transcribing=False` должен быть **вне** try/except, чтобы ни одна
  ветка не оставила overlay в состоянии «Transcribing…» навсегда.

## 6. Горячие клавиши и Win32-специфика

pynput Listener работает в своём потоке. `on_start`/`on_stop` вызываются
оттуда, поэтому всегда: `on_start=lambda: self.post_ui(self._begin_recording)`.

Особенности Win32-фильтра (`hotkeys.py`):
- Ctrl+Win: системный Win перехватывается раньше pynput; фильтр
  `win32_event_filter` сам генерирует press/release для VK 0x5B/0x5C.
- Инъецированные события (флаг `LLKHF_INJECTED`) игнорируются — иначе
  собственная вставка Ctrl+V вызвала бы рекурсивный старт диктовки.
- `suppress=True` — гасит меню Пуск при зажатом Win.

## 7. API-сервер и потоки

`ThreadingHTTPServer` — по потоку на запрос. GET-ы читают снапшоты
(атрибуты + `history.list()` — безопасно). POST-ы — `post_ui_sync`
(см. §3). `stop()` вызывает `shutdown()` + `join(timeout=2.0)`.

## 8. Правила для разработчика (cheat-sheet)

1. Мутация UI-состояния из не-UI потока → `post_ui` или `post_ui_sync`.
2. `post_ui_sync` — только из не-UI потоков.
3. Тяжёлые операции (загрузка, транскрипция, установка) — в daemon-потоках.
4. Флаги, которые читают другие потоки (`model_loading`, `transcribing`) —
   пиши только с UI-потока.
5. В worker-потоках не обращайся к tkinter напрямую и не читай `settings`
   «живьём» — передавай снапшот (`settings_snapshot`), чтобы не поймать
   изменение настроек посреди транскрипции.
6. Lambda-замыкания в колбэках очереди — всегда eager binding
   (`lambda x=x: ...`), особенно в except-ветках.
