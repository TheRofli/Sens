#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
venv_python="$root/.venv/bin/python"
pid_file="$root/data/speech.pid"
log_file="$root/data/speech.log"

export SPEECH_HOME="$root"
export PYTHONPATH="$root"
export SPEECH_DATA_DIR="$root/data"
export HF_HOME="$root/models/huggingface"
export HF_HUB_CACHE="$root/models/huggingface/hub"
export TRANSFORMERS_CACHE="$root/models/huggingface/transformers"
export TORCH_HOME="$root/models/torch"
export XDG_CACHE_HOME="$root/cache"
export PIP_CACHE_DIR="$root/cache/pip"
export TMPDIR="$root/tmp"

mkdir -p \
  "$SPEECH_DATA_DIR" \
  "$HF_HOME" \
  "$HF_HUB_CACHE" \
  "$TRANSFORMERS_CACHE" \
  "$TORCH_HOME" \
  "$XDG_CACHE_HOME" \
  "$PIP_CACHE_DIR" \
  "$TMPDIR"

find_host_python() {
  for candidate in python3.11 python3 python; do
    if command -v "$candidate" >/dev/null 2>&1; then
      if "$candidate" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)' >/dev/null 2>&1; then
        printf '%s\n' "$candidate"
        return 0
      fi
    fi
  done
  return 1
}

ensure_uv() {
  if command -v uv >/dev/null 2>&1; then
    return 0
  fi
  echo "Installing uv to fetch Python 3.11..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
  command -v uv >/dev/null 2>&1
}

new_speech_venv() {
  if [ -x "$venv_python" ]; then
    return 0
  fi

  if host_python="$(find_host_python)"; then
    "$host_python" -m venv "$root/.venv"
    return 0
  fi

  ensure_uv
  uv python install 3.11
  uv venv --python 3.11 "$root/.venv"
}

app_python() {
  if [ -x "$venv_python" ]; then
    printf '%s\n' "$venv_python"
  else
    find_host_python
  fi
}

install_speech() {
  echo "Creating Speech environment..."
  new_speech_venv

  echo "Installing Speech runtime dependencies..."
  "$venv_python" -m pip install --upgrade pip
  "$venv_python" -m pip install -r "$root/requirements.txt"

  if [ -s "$root/requirements-parakeet.txt" ]; then
    echo "Installing Parakeet dependencies..."
    "$venv_python" -m pip install -r "$root/requirements-parakeet.txt"
  fi

  if [ -s "$root/requirements-whisper.txt" ]; then
    echo "Installing Whisper (faster-whisper) dependencies..."
    "$venv_python" -m pip install -r "$root/requirements-whisper.txt"
  fi

  if [ -s "$root/requirements-gigaam.txt" ]; then
    echo "Installing GigaAM dependencies..."
    "$venv_python" -m pip install -r "$root/requirements-gigaam.txt"
  fi

  chmod +x "$root/speech.sh" "$root/bin/speech" 2>/dev/null || true
  mkdir -p "$HOME/.local/bin"
  ln -sf "$root/bin/speech" "$HOME/.local/bin/speech"

  case ":$PATH:" in
    *":$HOME/.local/bin:"*) ;;
    *)
      shell_profile="$HOME/.zshrc"
      [ -n "${BASH_VERSION:-}" ] && shell_profile="$HOME/.bashrc"
      if [ -f "$shell_profile" ] && ! grep -q 'HOME/.local/bin' "$shell_profile"; then
        printf '\nexport PATH="$HOME/.local/bin:$PATH"\n' >> "$shell_profile"
      fi
      ;;
  esac

  echo
  echo "Speech install complete."
  echo "Run: speech"
  echo "Download Parakeet: speech parakeet install"
}

is_running() {
  [ -f "$pid_file" ] && kill -0 "$(cat "$pid_file")" >/dev/null 2>&1
}

start_detached() {
  local python
  python="$(app_python)"
  if is_running; then
    echo "Speech is already running."
    return 0
  fi
  (
    cd "$root"
    nohup "$python" -m speech_app run "$@" >>"$log_file" 2>&1 &
    echo $! > "$pid_file"
  )
  echo "Speech started."
}

stop_speech() {
  if ! is_running; then
    echo "Speech is not running."
    rm -f "$pid_file"
    return 0
  fi
  kill "$(cat "$pid_file")" >/dev/null 2>&1 || true
  rm -f "$pid_file"
  echo "Speech stopped."
}

show_status() {
  if is_running; then
    echo "Speech is running with PID $(cat "$pid_file")."
  else
    echo "Speech is not running."
  fi
}

invoke_python() {
  local python
  python="$(app_python)"
  cd "$root"
  "$python" -m speech_app "$@"
}

command_name="${1:-start}"
if [ "$#" -gt 0 ]; then
  shift
fi

case "$command_name" in
  install)
    install_speech
    ;;
  run|start)
    start_detached "$@"
    ;;
  stop)
    stop_speech
    ;;
  restart)
    stop_speech
    start_detached "$@"
    ;;
  open)
    start_detached --show-window
    ;;
  status)
    show_status
    ;;
  foreground)
    invoke_python run "$@"
    ;;
  diagnose)
    invoke_python diagnose
    ;;
  parakeet)
    invoke_python parakeet "$@"
    ;;
  model)
    if [ $# -eq 0 ]; then
      echo "Usage: speech model install [parakeet|whisper-ru]" >&2
      echo "       speech model list" >&2
      exit 1
    fi
    # Legacy "model install/download/preload" (no preset arg) installs the
    # active model; otherwise pass through to the speech_app model parser.
    if [ $# -eq 1 ] && { [ "$1" = "install" ] || [ "$1" = "download" ] || [ "$1" = "preload" ]; }; then
      invoke_python model install
    else
      invoke_python model "$@"
    fi
    ;;
  *)
    invoke_python "$command_name" "$@"
    ;;
esac
