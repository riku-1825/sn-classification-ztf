source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

if conda env list | grep -q "sn-classification-ztf"; then
    run_logged "setup_env_update" conda env update -f environment.yaml --prune
else
    run_logged "setup_env_create" conda env create -f environment.yaml
fi

echo ""
echo "Environment ready. Activate it with:"
echo "    conda activate sn-classification-ztf"
