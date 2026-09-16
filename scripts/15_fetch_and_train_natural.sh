source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

TOTAL_BUDGET="${1:-450}"
EPOCHS="${2:-50}"
OUT="data/ztf_real_objects_natural.json"

echo "########## STEP 1/3: fetch NATURAL (non-curated) real ZTF sample ##########"
run_logged "fetch_real_data_natural" python fetch_real_data.py \
    --balance_mode natural --total_budget "$TOTAL_BUDGET" --out "$OUT"

echo "########## STEP 2/3: train + evaluate on the natural sample ##########"
run_logged "train_natural" python train.py \
    --data "$OUT" --outdir figures_natural --run_name natural --epochs "$EPOCHS"

echo "########## STEP 3/3: class-weighting ablation (meaningful now that classes are imbalanced) ##########"
run_logged "ablation_class_weight_natural" python ablation_class_weight.py \
    --data "$OUT" --outdir figures_natural --run_name natural

echo ""
echo "All steps completed. Compare figures_natural/natural_ablation_class_weight.json"
echo "against figures_real/real_ablation_class_weight.json (curated/balanced) -- the"
echo "effect of class weighting should be more visible here, if it matters at all."
