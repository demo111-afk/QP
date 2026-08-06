#!/usr/bin/env bash

set -eu

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
TEMPLATE_PATH="$PROJECT_DIR/QP_Copilot.desktop.in"

if command -v xdg-user-dir >/dev/null 2>&1; then
    DESKTOP_DIR="$(xdg-user-dir DESKTOP)"
else
    DESKTOP_DIR="$HOME/Desktop"
fi

if [ -z "$DESKTOP_DIR" ]; then
    DESKTOP_DIR="$HOME/Desktop"
fi

LAUNCHER_PATH="$DESKTOP_DIR/QP Copilot.desktop"
APPLICATIONS_DIR="$HOME/.local/share/applications"
APPLICATION_PATH="$APPLICATIONS_DIR/qp-copilot.desktop"
ESCAPED_PROJECT_DIR="$(printf '%s' "$PROJECT_DIR" | sed 's/[&|\\]/\\&/g')"

mkdir -p "$DESKTOP_DIR" "$APPLICATIONS_DIR"
sed "s|@PROJECT_DIR@|$ESCAPED_PROJECT_DIR|g" "$TEMPLATE_PATH" > "$LAUNCHER_PATH"
cp "$LAUNCHER_PATH" "$APPLICATION_PATH"
chmod +x "$LAUNCHER_PATH" "$APPLICATION_PATH"

if command -v gio >/dev/null 2>&1; then
    gio set "$LAUNCHER_PATH" metadata::trusted true >/dev/null 2>&1 || true
fi

printf 'QP Copilot launcher installed: %s\n' "$LAUNCHER_PATH"
