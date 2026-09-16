source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

DATA="${1:-data/synthetic_objects.json}"
OUTDIR="${2:-figures}"
RUN_NAME="${3:-run}"
N_SEEDS="${4:-5}"

if [ ! -f "$DATA" ]; then
    echo "!!! $DATA not found. Generate/fetch data first (see RUN_COMMANDS.md)." >&2
    exit 1
fi

run_logged "multiseed_eval_${RUN_NAME}" python multiseed_eval.py \
    --data "$DATA" --outdir "$OUTDIR" --run_name "$RUN_NAME" --n_seeds "$N_SEEDS"
