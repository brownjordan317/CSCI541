#!/usr/bin/env bash
set -e
EXP_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
printf '%s\n' "$$" > "$EXP_ROOT/queue.pid"
exec bash "$EXP_ROOT/run_queue.sh"
