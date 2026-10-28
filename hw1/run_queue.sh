#!/usr/bin/env bash
set -u
EXP_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$EXP_ROOT/../.." && pwd)"
TASK_PY="$REPO_ROOT/hw1/.venv/bin/python"
TASK_SCRIPT="$REPO_ROOT/hw1/run_mlp_experiment.py"
TASK_DATA="$REPO_ROOT/hw1/augmented_dataset"
STATUS_FILE="$EXP_ROOT/queue_status.tsv"
printf 'variant\tstatus\texit_code\tfinished_utc\n' > "$STATUS_FILE"
for variant in baseline pool8 width96 kaiming; do
    output="$EXP_ROOT/sub5_$variant"
    mkdir -p "$output"
    printf '%s starting %s\n' "$(date -u +%FT%TZ)" "$variant"
    "$TASK_PY" -u "$TASK_SCRIPT" --dataset "$TASK_DATA" \
        --variant "$variant" --output "$output" > "$output/training.log" 2>&1
    code=$?
    if [ "$code" -eq 0 ]; then
        status=complete
    else
        status=failed
    fi
    printf '%s\t%s\t%s\t%s\n' "$variant" "$status" "$code" "$(date -u +%FT%TZ)" >> "$STATUS_FILE"
    "$TASK_PY" "$EXP_ROOT/summarize.py"
    printf '%s %s %s\n' "$(date -u +%FT%TZ)" "$variant" "$status"
done
"$TASK_PY" "$EXP_ROOT/summarize.py"
