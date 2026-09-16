source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

NEW_N_PER_CLASS="${1:-350}"
EPOCHS="${2:-50}"
OUT="data/ztf_real_objects.json"
RUN_NAME="real_expanded_${NEW_N_PER_CLASS}"

if [ ! -f "$OUT" ]; then
    echo "!!! $OUT not found. Run scripts/07_run_all_real.sh first to fetch an initial real dataset." >&2
    exit 1
fi

echo "########## STEP 1/2: expand $OUT to $NEW_N_PER_CLASS objects/class (resuming) ##########"
run_logged "fetch_real_data_expand_${NEW_N_PER_CLASS}" python fetch_real_data.py \
    --n_per_class "$NEW_N_PER_CLASS" --out "$OUT"

echo "########## STEP 2/2: retrain + evaluate on the expanded dataset ##########"
run_logged "train_${RUN_NAME}" python train.py \
    --data "$OUT" --outdir "figures_${RUN_NAME}" --run_name "$RUN_NAME" --epochs "$EPOCHS"

echo ""
echo "Compare figures_${RUN_NAME}/${RUN_NAME}_summary.json (expanded, ~${NEW_N_PER_CLASS}/class)"
echo "against figures_real/real_summary.json (original, 150/class) -- specifically compare"
echo "the LightGBM-vs-GRU gap in each. If the GRU's macro-F1 gains more than LightGBM's as"
echo "data grows, that supports the 'GRU needs more real data' hypothesis from the report."
