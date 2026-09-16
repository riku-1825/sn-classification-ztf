source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

DATA="${1:-data/synthetic_objects.json}"
OUTDIR="${2:-figures}"
RUN_NAME="${3:-run}"

if [ ! -f "$DATA" ]; then
    echo "!!! $DATA not found. Generate/fetch data first (see RUN_COMMANDS.md)." >&2
    exit 1
fi

run_logged "error_analysis_${RUN_NAME}" python error_analysis.py \
    --data "$DATA" --outdir "$OUTDIR" --run_name "$RUN_NAME"
