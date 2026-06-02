#!/usr/bin/env bash
set -euo pipefail

export OPENAI_API_KEY=""
export OPENAI_MODEL="gpt-4.1-mini"

#python scripts/generate_test_data.py

existing_pid=""
if command -v lsof >/dev/null 2>&1; then
  existing_pid="$(lsof -ti tcp:8000 -sTCP:LISTEN 2>/dev/null | head -n 1 || true)"
elif command -v fuser >/dev/null 2>&1; then
  existing_pid="$(fuser 8000/tcp 2>/dev/null | awk '{print $1}' || true)"
fi

if [ -n "${existing_pid}" ]; then
  kill "${existing_pid}" 2>/dev/null || true
fi

exec python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
