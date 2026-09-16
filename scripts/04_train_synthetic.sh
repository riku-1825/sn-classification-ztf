source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
EPOCHS="${1:-30}"
if [ ! -f data/synthetic_objects.json ]; then
    echo "data/synthetic_objects.json not found — generating it first."
    run_logged "generate_synthetic" python data/generate_synthetic_data.py \
        --n_per_class 200 --out data/synthetic_objects.json
fi
run_logged "train_synthetic" python train.py \
    --data data/synthetic_objects.json --outdir figures --run_name synthetic --epochs "$EPOCHS"
