source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

DATA="${1:-data/synthetic_objects.json}"
OUTDIR="${2:-figures}"
RUN_NAME="${3:-run}"

if [ ! -f "$DATA" ]; then
    echo "!!! $DATA not found. Generate/fetch data first (see RUN_COMMANDS.md)." >&2
    exit 1
fi

echo "########## 1/4: early-epoch ablation ##########"
run_logged "ablation_early_epoch_${RUN_NAME}" python ablation_early_epoch.py \
    --data "$DATA" --outdir "$OUTDIR" --run_name "$RUN_NAME"

echo "########## 2/4: class-weight ablation ##########"
run_logged "ablation_class_weight_${RUN_NAME}" python ablation_class_weight.py \
    --data "$DATA" --outdir "$OUTDIR" --run_name "$RUN_NAME"

echo "########## 3/4: interpolation ablation ##########"
run_logged "ablation_interpolation_${RUN_NAME}" python ablation_interpolation.py \
    --data "$DATA" --outdir "$OUTDIR" --run_name "$RUN_NAME"

echo "########## 4/4: error analysis (misclassified examples) ##########"
run_logged "error_analysis_${RUN_NAME}" python error_analysis.py \
    --data "$DATA" --outdir "$OUTDIR" --run_name "$RUN_NAME"

echo ""
echo "All analysis steps completed. Check $OUTDIR/ for:"
echo "  ${RUN_NAME}_ablation_early_epoch.png / .json"
echo "  ${RUN_NAME}_ablation_class_weight.json"
echo "  ${RUN_NAME}_ablation_interpolation.json"
echo "  ${RUN_NAME}_test_predictions.csv"
echo "  ${RUN_NAME}_misclassified_examples.png"
