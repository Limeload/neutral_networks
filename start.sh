#!/usr/bin/env bash
set -e
ROOT="$(cd "$(dirname "$0")" && pwd)"

PYTHON="$ROOT/.venv/bin/python3.12"
export DYLD_LIBRARY_PATH="/opt/homebrew/opt/expat/lib:$DYLD_LIBRARY_PATH"

echo "Starting Streamlit on :8501…"
(cd "$ROOT" && "$PYTHON" -m streamlit run app/app.py --server.port 8501)
