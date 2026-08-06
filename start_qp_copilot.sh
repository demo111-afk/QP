#!/usr/bin/env bash

set -u

BASE_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON_PATH=""

for candidate in "$BASE_DIR/.venv/bin/python" "$BASE_DIR/venv/bin/python"; do
    if [ -x "$candidate" ]; then
        PYTHON_PATH="$candidate"
        break
    fi
done

if [ -z "$PYTHON_PATH" ]; then
    MESSAGE="Virtual Environment not found.

Initialize it once in the project folder:

python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m playwright install chromium"
    if command -v zenity >/dev/null 2>&1; then
        zenity --error --title="QP Copilot" --text="$MESSAGE"
    elif command -v xmessage >/dev/null 2>&1; then
        xmessage -center "$MESSAGE"
    fi
    exit 1
fi

cd "$BASE_DIR"
exec "$PYTHON_PATH" "$BASE_DIR/ui_app.py"
