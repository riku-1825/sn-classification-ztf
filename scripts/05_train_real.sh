source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
EPOCHS="${1:-50}"
if [ ! -f data/ztf_real_objects.json ]; then
    echo "data/ztf_real_objects.json not found." >&2
    echo "Run ./scripts/03_fetch_real_data.sh first (requires internet access)." >&2
    exit 1
fi
run_logged "train_real" python train.py \
    --data data/ztf_real_objects.json --outdir figures_real --run_name real --epochs "$EPOCHS"
