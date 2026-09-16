source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
N_PER_CLASS="${1:-200}"
run_logged "generate_synthetic" python data/generate_synthetic_data.py \
    --n_per_class "$N_PER_CLASS" --out data/synthetic_objects.json
