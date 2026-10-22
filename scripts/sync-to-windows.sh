#!/usr/bin/env bash
# Mirror the WSL checkout to E:\projects\nil-dj for Windows runs.
# Excluded paths are never deleted on the Windows side, so its .venv survives.
set -euo pipefail

rsync -a --delete \
  --exclude .git \
  --exclude .venv \
  --exclude __pycache__ \
  --exclude .pytest_cache \
  --exclude .ruff_cache \
  ~/projects/nil-dj/ /mnt/e/projects/nil-dj/
