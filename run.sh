#!/usr/bin/env bash
# Single entry point for the RESGA / SAEGA codebase.
#
#   ./run.sh main     --persona sycophancy --method residual --fluency medium \
#                     --layer_idx 14 --device cuda:0
#   ./run.sh sweep    [--config configs/sweep.yaml] [--dry_run]
#   ./run.sh baseline gcg            --device cuda:0
#   ./run.sh baseline protegi        --device cuda:0      # needs OPENROUTER_API_KEY
#   ./run.sh baseline prefix_tuning  --device cuda:0
#   ./run.sh baseline prompt_steering --device cuda:0 --layer_idx 14
#
# Any extra flags are passed straight through to the underlying script.
set -e

usage() {
    grep '^#' "$0" | sed 's/^# \{0,1\}//'
    echo
    echo "Available baselines: $(ls baselines/*.py | grep -v common | xargs -n1 basename | sed 's/.py//' | tr '\n' ' ')"
}

MODE="${1:-}"; shift || true
case "$MODE" in
    main)
        python main.py "$@" ;;
    sweep)
        python sweep.py "$@" ;;
    baseline)
        NAME="${1:-}"; shift || true
        if [ -z "$NAME" ] || [ ! -f "baselines/${NAME}.py" ]; then
            echo "Unknown baseline: '${NAME}'"; usage; exit 1
        fi
        python "baselines/${NAME}.py" "$@" ;;
    -h|--help|"")
        usage ;;
    *)
        echo "Unknown mode: '${MODE}'"; usage; exit 1 ;;
esac
