# Speech

Speech is a local push-to-talk dictation app. Hold a hotkey, speak, release,
and the transcript is pasted into the active input, copied to the clipboard,
and saved in searchable local history.

Speech recognition runs locally on your machine. Audio and transcripts are not
sent to an online speech service.

## Quick Install

### Windows 11

Open PowerShell and run:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -Command "irm https://raw.githubusercontent.com/TheRofli/speech/main/bootstrap.ps1 | iex"
```

Then download the local speech model:

```powershell
speech parakeet install
```

Start Speech:

```powershell
speech
```

The Windows bootstrap downloads the source from GitHub, creates a local virtual
environment, installs dependencies, and adds the `speech` command to your user
PATH. If Python 3.11 is missing, it tries to install it with `winget`.

Default install location:

```text
%LOCALAPPDATA%\Programs\Speech
```

To choose another folder:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -Command "& ([scriptblock]::Create((irm https://raw.githubusercontent.com/TheRofli/speech/main/bootstrap.ps1))) -InstallDir 'E:\Apps\Speech'"
```

### macOS

Open Terminal and run:

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/TheRofli/speech/main/bootstrap.sh)"
```

Then download the local speech model:

```bash
speech parakeet install
```

Start Speech:

```bash
speech
```

The macOS bootstrap downloads the source from GitHub, creates a local virtual
environment, installs dependencies, and links `speech` into `~/.local/bin`.
If Python 3.11 is missing, it uses `uv` to install a local Python runtime.

Default install location:

```text
~/.speech
```

To choose another folder:

```bash
SPEECH_INSTALL_DIR="$HOME/Applications/Speech" /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/TheRofli/speech/main/bootstrap.sh)"
```

macOS support is source-install support. The Python tray/runtime is the stable
path; packaged signed `.app` builds are still planned.

## Models

Speech ships with two selectable ASR models. Switch between them from the tray
(`Model` submenu) or the Controls tab in the window. Both run on CPU.

### Parakeet (default, fast)

```text
nvidia/parakeet-tdt-0.6b-v3
```

Parakeet TDT 0.6B v3 is a 600M-parameter multilingual automatic speech
recognition model from NVIDIA. The model card lists 25 supported languages,
including English, Russian, Ukrainian, German, French, Spanish, Portuguese,
Italian, Polish, Dutch, Turkish, Arabic, Chinese, Japanese, and Korean.

Useful capabilities:

- automatic language detection
- punctuation and capitalization
- timestamps
- long audio support
- CPU mode by default, CUDA optional on Windows/Linux systems with a compatible
  NVIDIA setup

Sources:

- [NVIDIA Parakeet TDT 0.6B v3 model card](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3)
- [NVIDIA NeMo ASR collection](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/asr/intro.html)

### Whisper RU codeswitch (accurate)

```text
coriollon/whisper-large-v3-turbo-russian-codeswitch
```

A fine-tune of Whisper large-v3-turbo (809M) trained specifically for
Russian+English code-switching. Recommended when you mix both languages in the
same phrase. Runs through `faster-whisper` (CTranslate2, INT8) for fast CPU
inference.

Install it once (this downloads the source checkpoint and converts it to the
fast CTranslate2 format — needs several GB of RAM and a few minutes):

```powershell
speech model install whisper-ru
```

Sources:

- [coriollon/whisper-large-v3-turbo-russian-codeswitch](https://huggingface.co/coriollon/whisper-large-v3-turbo-russian-codeswitch)
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper)

## Quality Pipeline

Every transcript goes through a deterministic cleanup pipeline on the way to
your input:

1. **VAD trim** — leading/trailing silence and keyboard clicks are removed
   before the model sees the audio. This is the main defence against Whisper
   hallucinations on near-empty input. Tune it with `vad_sensitivity` in the
   Controls tab (smaller value = more permissive).
2. **ASR** — the selected model (Parakeet or Whisper) transcribes the trimmed
   audio. Automatic language detection is always on, so RU/EN code-switching
   works without fixing a language.
3. **Post-processing** — whitespace is collapsed, the first character is
   capitalised, and a short list of known Whisper silence-hallucination phrases
   is stripped. Toggle with `Post-process text` in the Controls tab.

Generation knobs exposed in the Controls tab (Quality section):

- `beam_size`, `temperature`, `repetition_penalty` — forwarded to the model
- `compression_ratio_threshold`, `log_prob_threshold` — Whisper anti-hallucination

## Requirements

Minimum practical setup:

- Windows 11 or macOS 13+
- Python 3.11, installed automatically by the bootstrap when possible
- working microphone
- 8 GB RAM
- 10 GB free disk space for the app, virtual environment, caches, and model

Recommended:

- 16 GB RAM or more
- modern 4-core CPU or better
- 15-20 GB free disk space
- NVIDIA GPU only if you specifically want CUDA

Notes:

- CPU mode is the stable default.
- CUDA requires a compatible NVIDIA driver and PyTorch CUDA install.
- The model is downloaded only when you run `speech parakeet install` or pass
  the bootstrap download flag.
- Model files, virtual environments, caches, transcripts, and local settings are
  intentionally excluded from Git.

## Controls

Default hotkeys:

- Windows: hold `Ctrl + Win`
- macOS: hold `Control + Command`

Workflow:

1. Hold the hotkey.
2. Speak.
3. Release.
4. Speech transcribes locally, then sends text to the active input, clipboard,
   and local history.

When a transcript arrives it is trimmed of silence, transcribed by the active
model, post-processed, then sent to the active input, clipboard, and history.

The tray menu lets you open the window, copy the last transcript, load/unload
the active model, switch model (Parakeet / Whisper), switch CPU/CUDA mode, and
quit.

## Commands

Windows PowerShell:

```powershell
speech
speech status
speech stop
speech restart
speech open
speech diagnose
speech parakeet install
speech model install parakeet
speech model install whisper-ru
speech model list
speech foreground
```

macOS Terminal:

```bash
speech
speech status
speech stop
speech restart
speech diagnose
speech parakeet install
speech model install parakeet
speech model install whisper-ru
speech model list
speech foreground
```

`speech` starts the background tray/runtime. `speech foreground` is mainly for
debugging because it keeps logs attached to the terminal. `speech parakeet
install` is kept as a backwards-compatible alias for `speech model install
parakeet`.

## Local Data

Speech stores local runtime data inside the install folder:

```text
data/       transcripts, settings, runtime state
models/     Hugging Face and Torch model caches
cache/      package and runtime cache
tmp/        temporary audio files
.venv/      local Python virtual environment
```

Audio is never uploaded by Speech and transcripts stay on the device. Hugging
Face is contacted only when downloading models.

## Development

Clone and install locally:

```powershell
git clone https://github.com/TheRofli/speech.git
cd speech
.\install.ps1
```

macOS/Linux shell:

```bash
git clone https://github.com/TheRofli/speech.git
cd speech
./install.sh
```

Python tests:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s .\tests
```

Frontend checks:

```powershell
cd tauri
npm install
npm run build
cd src-tauri
cargo check
```

Keep these out of commits:

- `.venv/`
- `models/`
- `data/`
- `cache/`
- `tmp/`
- `tauri/node_modules/`
- `tauri/dist/`
- `tauri/src-tauri/target/`
- `tauri/src-tauri/gen/`

## Roadmap

- signed Windows installer release
- signed macOS `.app` release
- optional CUDA install helper
- optional NeMo backend
- optional transcript analysis tab through DeepSeek, OpenAI, or a local model
