#!/usr/bin/env bash
set -u
EXP_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
printf '%s\n' "$$" > "$EXP_ROOT/refresh_comparison.pid"
for variant in pool8 width96 kaiming; do
    while [ ! -f "$EXP_ROOT/sub5_$variant/results.json" ]; do
        if grep -q "^$variant[[:space:]]" "$EXP_ROOT/queue_status.tsv"; then
            break
        fi
        queue_pid=$(cat "$EXP_ROOT/queue.pid")
        if ! kill -0 "$queue_pid" 2>/dev/null; then
            "$EXP_ROOT/../.venv/bin/python" "$EXP_ROOT/summarize.py"
            exit 0
        fi
        sleep 60
    done
    "$EXP_ROOT/../.venv/bin/python" "$EXP_ROOT/summarize.py"
done
