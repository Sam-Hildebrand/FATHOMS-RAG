#!/usr/bin/env bash
set -e  # exit on any error

# Load .env if present
if [ -f ".env" ]; then
  echo "Loading environment from .env"
  # shellcheck disable=SC2046
  export $(grep -v '^#' .env | xargs)
fi

# Ensure OPENWEBUI_API_KEY is set (and persist to .env)
if [ -z "${OPENWEBUI_API_KEY:-}" ]; then
  read -r -p "Enter your OpenWebUI API key: " OPENWEBUI_API_KEY
  export OPENWEBUI_API_KEY

  # Only append if .env exists or create it
  touch .env
  # Avoid duplicating the key
  if ! grep -q '^OPENWEBUI_API_KEY=' .env; then
    echo "OPENWEBUI_API_KEY=${OPENWEBUI_API_KEY}" >> .env
    echo "Saved OPENWEBUI_API_KEY to .env"
  fi
fi

# Ensure OPENWEBUI_URL is set (and persist to .env)
if [ -z "${OPENWEBUI_URL:-}" ]; then
  read -r -p "Enter your OpenWebUI URL: " OPENWEBUI_URL
  export OPENWEBUI_API_KEY

  # Only append if .env exists or create it
  touch .env
  # Avoid duplicating the key
  if ! grep -q '^OPENWEBUI_URL=' .env; then
    echo "OPENWEBUI_URL=${OPENWEBUI_URL}" >> .env
    echo "Saved OPENWEBUI_URL to .env"
  fi
fi

echo "Using OPENWEBUI_URL: ${OPENWEBUI_URL}"
echo "Using OPENWEBUI_API_KEY: ${OPENWEBUI_API_KEY:0:4}*** (hidden)"

# Make sure uv is installed. If not, install via pip:
# curl -LsSf https://astral.sh/uv/install.sh | sh
# or: pip install uv

# Pin the Python version you want
PYTHON_VERSION="3.12.0"

# Install that Python version (if you don’t already have it)
echo "Installing Python ${PYTHON_VERSION} via uv..."
uv python install "${PYTHON_VERSION}"

# Create a .venv using that interpreter
echo "Creating virtual environment (.venv) with Python ${PYTHON_VERSION}..."
uv venv --python "${PYTHON_VERSION}"            # creates .venv

# Activate it
source .venv/bin/activate

# Use uv’s pip-compile to generate locked deps
uv pip compile local-requirements.in \
    --universal \
    --output-file uv-requirements.txt

echo "Installing dependencies (sync)..."
uv pip sync uv-requirements.txt

rm uv-requirements.txt

# Show what’s in your env
echo
echo "# Environment variables"
echo "OPENWEBUI_URL: ${OPENWEBUI_URL}"
echo "OPENWEBUI_API_KEY=${OPENWEBUI_API_KEY:0:4}***"
echo

echo "Using Python: $(which python)"
echo "Using Pip:   $(which pip)"
echo
echo "# Now Run:"
echo "source .venv/bin/activate"
echo
