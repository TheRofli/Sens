# 08 — История решений (Decision Log)

Почему проект устроен именно так. Хронологически, с граблями.

## Решения на уровне продукта

1. **Локально и офлайн.** Никаких облачных ASR. Всё распознавание на машине,
   CPU-only. GPU не грузим (пользователь явно просил). Это сформировало выбор
   моделей и форматов (CT2 INT8, INT8 ONNX, small-models).
2. **Push-to-talk.** Удерживаешь хоткей — говоришь — отпускаешь. Не
   always-on: меньше шума, проще UX, дешевле по CPU.
3. **Русский + английский вперемешку (код-свитчинг).** Язык НЕ фиксируется —
   авто-детект у всех моделей. Под это выбран
   `coriollon/whisper-large-v3-turbo-russian-codeswitch` — файн-тюн
   специально под RU+EN в одном предложении.
4. **Три модели, переключаемые из трея и окна:**
   - Parakeet TDT 0.6B v3 — быстрая, мультиязычная, default;
   - Whisper RU-codeswitch — точная для смешанного языка (через
     faster-whisper CT2 INT8);
   - GigaAM v3 e2e_rnnt — лучший русский на CPU, пунктуация из коробки
     (добавлена последней, см. 5e6dd3d).
5. **Tauri — единственный GUI.** tkinter-окно удалено. Остаются скрытый
   tkinter root (для overlay и UI-очереди) + pystray-трей. Tauri общается с
   ядром через локальный HTTP API — это позволило вообще не тащить Python в
   GUI-процесс.

## Ключевые технические решения

6. **HTTP API вместо IPC/файлов** (для Tauri). Причины: не зависит от
   платформы, легко отлаживать (`curl`), любой локальный клиент может
   управлять Speech (пригодится при объединении с Eye). Порт — эфемерный,
   discovery через `data/api.port`.
7. **Реестр моделей** (`speech_app/models.py`) как единый источник правды.
   Трей, API, установка, статус, Tauri — всё читает из реестра. Новая модель
   = новый пресет + движок + несколько веток диспетчера; UI подхватывает
   сам.
8. **Многопоточность через очередь.** tkinter не потокобезопасен; все
   переходы между потоками — через `post_ui` (async queue) и `post_ui_sync`
   (sync c Event + timeout). См. `04-THREADING.md`.
9. **VAD-обрезка по RMS перед моделью.** Главная защита от галлюцинаций
   Whisper на почти пустом аудио (клики хоткея, дыхание). Плюс для Whisper:
   `condition_on_previous_text=False` + пороги компрессии/лог-проба.
10. **textpost — детерминированная чистка.** Без LLM/эвристик с «умным»
    переписыванием: только схлопывание пробелов, удаление фраз-галлюцинаций,
    капитализация. (AI-очистка SAGE/OpenAI была предложена и **отклонена**
    пользователем — он хочет точный текст, а не «улучшенный».)

## Грабли (в хронологическом порядке)

11. **pystray arity (0bb4c1d).** Динамические label-колбэки обязаны принимать
    **1 аргумент** (`item`), action — **2** (`icon, item`), checked — **1**.
    Один неверный lambda — и tray-поток молча умирает, иконки нет, а
    приложение «работает, но трей пуст».
12. **Whisper: preprocessor_config.json (90755ff).** faster-whisper не
    получал `preprocessor_config.json` после CT2-конверсии и использовал 80
    mel-bins вместо 128 для large-v3 → `Invalid input features shape:
    expected (1, 128, 3000), got (1, 80, 3000)`. Фикс: сохранять
    preprocessor при конверсии (`_save_preprocessor_config`).
13. **Whisper: NameError в worker (90755ff).** Lambda в except-ветке
    замыкала переменные поздно → NameError маскировал настоящую ошибку, и
    пользователь видел бесконечное «анализирует…». Фикс: eager binding
    (`lambda t=title, e=exc: ...`) + сброс `transcribing=False` вне try.
14. **CT2-конвертер из чужого venv (90755ff).** `shutil.which("ct2-transformers-converter")`
    находил CLI из другого проекта (hermes-agent venv) → криптические
    NameError. Фикс: конверсия in-process через `TransformersConverter`,
    без subprocess.
15. **Tauri: discovery пути (41c5bc2).** `speech_root()` на compile-time
    `env!("CARGO_MANIFEST_DIR")` не работал на другой машине/установке →
    Rust не находил `data/api.port` → UI «всё остановлено», модели не
    видны, история пуста. Фикс: `std::env::current_exe()` + parent×4 +
    проверка существования `data/api.port`/`speech_app`.
16. **Tauri: serde case (41c5bc2).** `#[serde(rename_all = "camelCase")]` на
    структурах, читающих Python (snake_case) → «missing field engineEnabled».
    Фикс: `rename_all(serialize="camelCase", deserialize="snake_case")`.
17. **Tauri: сборка (критично!).** `cargo build --release` напрямую НЕ
    встраивает `dist/` → exe грузит devUrl 1420 → «ERR_CONNECTION_REFUSED».
    Только `npm run tauri:build` (vite build → cargo → bundle).
18. **GigaAM vs transformers 5.x (5e6dd3d).** Remote-code модель написана под
    4.x; в 5.x ломается: (а) meta-device init против torchaudio `.item()`;
    (б) ffmpeg-зависимость в load_audio; (в) отсутствие
    `all_tied_weights_keys`. Все три закрыты патчем, применяемым при
    установке (см. `02-ENGINES.md` §6).
19. **Двойной запуск ядра.** При отладке обнаружили, что запуск «через
    speech.ps1 + ещё раз вручную» даёт два процесса (`pythonw` из venv и
    системный Python311) с конфликтом за порт/lock. SingleInstanceLock
    (`data/speech.lock`) защищает один экземпляр; launcher-обёртки должны
    вызываться один раз.

## Что НЕ входит (осознанно)

- **AI-постобработка** (переписывание текста нейросетью) — отклонено
  пользователем.
- **Streaming/real-time ASR** — push-to-talk батч-транскрипция достаточно
  быстра (GigaAM: ~0.35с на фразу, Parakeet быстрее).
- **Облачные провайдеры** — только локально.
- **Анализ транскриптов (вкладка Analysis)** — в roadmap, не приоритет.
