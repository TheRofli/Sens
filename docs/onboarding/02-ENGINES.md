# 02 — Система ASR-движков

## 1. Интерфейс

Все движки реализуют протокол `SpeechEngine` (`speech_app/engines/base.py`):

```python
@runtime_checkable
class SpeechEngine(Protocol):
    @property
    def is_loaded(self) -> bool: ...
    @property
    def model_id(self) -> str: ...
    def load(self, settings: "AppSettings") -> None: ...
    def unload(self) -> None: ...
    def transcribe(self, samples: np.ndarray, sample_rate: int,
                   settings: "AppSettings") -> str: ...
```

- `samples` — float32 mono в диапазоне [-1, 1], уже обрезанный VAD-ом.
- Возврат — строка текста (пустая строка допустима).
- Ошибки — `EngineUnavailable(RuntimeError)` с человеческим сообщением.

Дополнительно: `LoadedEngine` dataclass (backend, device, model_id, model,
processor) — legacy-обёртка для Parakeet, не обязательна для новых движков.

## 2. Три движка

| Движок | Файл | Модель | Рантайм | Особенности |
|--------|------|--------|---------|-------------|
| `parakeet` | `engines/parakeet.py` | `nvidia/parakeet-tdt-0.6b-v3` | transformers (AutoModelForTDT), fallback NeMo | 600M, мультиязычная, быстрая на CPU; генерация с beam/temperature/repetition_penalty/no_repeat_ngram |
| `whisper` | `engines/whisper.py` | `coriollon/whisper-large-v3-turbo-russian-codeswitch` | faster-whisper (CTranslate2 INT8) | 809M, файн-тюн под RU+EN код-свитчинг; анти-галлюцинация: condition_on_previous_text=False, compression_ratio/log_prob пороги, VAD-фильтр |
| `gigaam` | `engines/gigaam.py` | `ai-sage/GigaAM-v3` (revision e2e_rnnt) | transformers (trust_remote_code, патченая копия) | 230M, лучший русский на CPU; e2e: пунктуация + нормализация прямо на выходе; принимает путь к файлу → временный WAV |

## 3. Как движки получают аудио

Разные модели принимают данные по-разному:

- **Parakeet (transformers)**: `processor([audio], sampling_rate=...)` →
  тензоры → `model.generate(...)` → `processor.decode(...)`.
- **Whisper**: numpy float32 16k mono → `WhisperModel.transcribe(...)`.
- **GigaAM**: `model.transcribe(wav_file_path)` — принимает **путь к файлу**.
  Движок пишет временный WAV через `_write_temp_wav()` (общий хелпер из
  `engines/parakeet.py`) и удаляет его после.

## 4. EngineManager — выбор и переключение

`speech_app/engine_manager.py`:

```python
def make_engine(kind: str) -> SpeechEngine:
    if kind == "whisper":  return WhisperEngine()
    if kind == "gigaam":   return GigaAMEngine()
    return ParakeetEngine()
```

- `resolve_engine(settings)` (из `models.py`) возвращает kind по `settings.model`.
- `load(settings)`: если уже загружен нужный kind и healthy — no-op; если
  kind сменился — `unload()` старого.
- `transcribe(...)`: авто-reload, если модель сменилась или выгружена.

## 5. Реестр моделей — как добавить 4-ю модель

Точка расширения — `speech_app/models.py`. Шаги:

1. **Пресет** в `MODELS`:
   ```python
   "my-model": ModelPreset(
       key="my-model",
       label="My Model (метка для UI)",
       engine="myengine",               # ключ движка
       model_id="org/my-model",          # HF repo id
       family="myengine",                # family для кэша
       description="Короткое описание.",
   )
   ```
2. **Движок** `speech_app/engines/myengine.py` — класс с протоколом
   `SpeechEngine`.
3. **make_engine** в `engine_manager.py` — ветка `if kind == "myengine"`.
4. **Пути** в `engines/paths.py` — функция `my_model_dir(preset)`.
5. **Статус установки** — `find_myengine_model_status(preset)` в
   `model_status.py` + ветка в `find_model_status_for_preset` и в
   `SpeechApp.model_status_for` (app.py).
6. **Установка** — `install_myengine_model(preset)` в `engines/install.py` +
   ветка в `install_model(preset_key)`.
7. **Зависимости** — `requirements-myengine.txt` + строка в `speech.ps1` /
   `speech.sh`.
8. **Тесты** — registry/status/install-патч.

Всё остальное (трей, API, Tauri, CLI) подхватит автоматически, потому что
читает из реестра.

## 6. Установка моделей (`engines/install.py`)

- `install_parakeet_model(model_id)` — `snapshot_download` в HF-кэш.
- `install_whisper_model(preset)` — конвертация transformers → CTranslate2
  INT8 **in-process** (`TransformersConverter`), затем `_save_preprocessor_config`
  (иначе faster-whisper упадёт на 80 mel bins для large-v3!), затем маркер
  `INSTALLED.json` в `models/whisper/<key>/`.
- `install_gigaam_model(preset)` — snapshot_download + копия в
  `models/gigaam/<key>/` + `_patch_gigaam_module()` (патч под transformers 5.x,
  см. ниже).
- `install_model(preset_key)` — диспетчер по `preset.engine`.
- `list_models()` — печать статуса всех пресетов.

### GigaAM: патч под transformers 5.x

GigaAM — remote-code модель (`trust_remote_code`), написанная под
transformers 4.x. В transformers 5.x ломается в трёх местах — все закрыты в
`_patch_gigaam_module()` (`engines/install.py`), патч применяется при
установке и живёт в `models/gigaam/<key>/modeling_gigaam.py`:

1. **FeatureExtractor**: transformers 5.x создаёт модели внутри
   `torch.device("meta")`; `torchaudio.transforms.MelSpectrogram` вызывает
   `.item()` при `__init__` → краш. Патч: строить MelSpectrogram в контексте
   `with torch.device("cpu")`.
2. **load_audio**: оригинал вызывает ffmpeg через subprocess (не всегда есть
   на Windows). Патч: читать через `soundfile` (+ линейный ресемплинг).
3. **all_tied_weights_keys**: transformers 5.x требует этот атрибут в
   `_finalize_model_loading`; у модели его нет (нет tied weights). Патч:
   `self.all_tied_weights_keys = {}` в `GigaAMModel.__init__`.
   **Нельзя** вызывать `post_init()` — он в конце делает `init_weights()`,
   что переинициализировало бы загруженные веса.

## 7. Настройки, влияющие на движки

Все из `AppSettings` (см. `settings.py`), пробрасываются через `settings`
в `transcribe()`:

- `beam_size` (5), `temperature` (0.0) — Parakeet generate / Whisper beam.
- `repetition_penalty` (1.0), `no_repeat_ngram_size` (0) — Parakeet.
- `compression_ratio_threshold` (2.4), `log_prob_threshold` (-1.0) — Whisper
  анти-галлюцинация.
- `vad_sensitivity` (0.02) — VAD-обрезка до движка.
- `postprocess_text` (True) — textpost после движка.
- `device` ("cpu"), `backend` ("auto") — Parakeet; GigaAM всегда CPU.

## 8. Известные грабли

- **faster-whisper без preprocessor_config.json** → `Invalid input features
  shape: expected (1, 128, 3000), got (1, 80, 3000)`. Решение:
  `_save_preprocessor_config()` при конверсии (коммит 90755ff).
- **ct2-transformers-converter из чужого venv** (PATH) → cryptic NameError.
  Решение: конверсия in-process, без CLI (коммит в 90755ff).
- **GigaAM meta-device / tied-keys** — см. выше, `_patch_gigaam_module`.
- **Whisper галлюцинации на тишине** — VAD trim + condition_on_previous_text=False.
