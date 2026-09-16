source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
N_PER_CLASS="${1:-150}"
run_logged "fetch_real_data" python fetch_real_data.py \
    --n_per_class "$N_PER_CLASS" --out data/ztf_real_objects.json
