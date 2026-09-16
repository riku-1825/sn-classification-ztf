source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
N_PER_CLASS="${1:-150}"
EPOCHS="${2:-50}"

echo "########## STEP 1/2: fetch real ZTF data via ALeRCE ##########"
run_logged "fetch_real_data" python fetch_real_data.py \
    --n_per_class "$N_PER_CLASS" --out data/ztf_real_objects.json

echo "########## STEP 2/2: train + evaluate on real data ##########"
run_logged "train_real" python train.py \
    --data data/ztf_real_objects.json --outdir figures_real --run_name real --epochs "$EPOCHS"

echo ""
echo "All steps completed. Logs are in logs/, figures in figures_real/."
