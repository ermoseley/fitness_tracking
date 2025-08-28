#!/bin/bash
set -euo pipefail

# Clean build artifacts and caches
printf "Cleaning build artifacts and caches...\n"

rm -rf build dist installer_temp WeightTracker-Installer.dmg WeightTracker.spec || true

# Bytecode and caches
find . -name "__pycache__" -type d -prune -exec rm -rf {} +
find . -name "*.pyc" -delete
find . -name "*.pyo" -delete

# Optional: remove virtual envs when called with --all
if [[ "${1:-}" == "--all" ]]; then
  printf "Removing virtual environments...\n"
  rm -rf venv venv_minimal .venv || true
fi

printf "Done.\n"
