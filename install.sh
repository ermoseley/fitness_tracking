#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="$PROJECT_DIR/.venv"

echo "Installing BodyMetrics dependencies..."

if ! command -v python3 >/dev/null 2>&1; then
  echo "Error: python3 not found. Please install Python 3.9+ and re-run." >&2
  exit 1
fi

PY=python3

if [ ! -d "$VENV_DIR" ]; then
  echo "Creating virtual environment at $VENV_DIR"
  "$PY" -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"

echo "Upgrading pip..."
python -m pip install --upgrade pip

echo "Installing requirements..."
pip install -r "$PROJECT_DIR/requirements.txt"

echo "Checking tkinter availability..."
python - <<'PY'
try:
    import tkinter  # noqa: F401
    print("tkinter: OK")
except Exception as e:
    print("tkinter not available: " + str(e))
PY

echo "Installing optional macOS tools (sips, iconutil) check..."
if command -v sips >/dev/null 2>&1 && command -v iconutil >/dev/null 2>&1; then
  echo "macOS icon tools: OK"
else
  echo "Note: sips/iconutil not found. Icon packaging steps will be skipped."
fi

LAUNCHER="$HOME/Desktop/BodyMetrics.command"
echo "Creating Desktop launcher: $LAUNCHER"
cat > "$LAUNCHER" <<EOF
#!/usr/bin/env bash
cd "$PROJECT_DIR"
source "$VENV_DIR/bin/activate"
exec python gui.py
EOF
chmod +x "$LAUNCHER"

echo "Install complete. Use the Desktop launcher or run:"
echo "  source '$VENV_DIR/bin/activate' && python gui.py"

