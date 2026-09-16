source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
echo "########## STEP 1/3: unit tests ##########"
run_logged "tests" pytest tests/ -q

echo "########## STEP 2/3: generate synthetic data ##########"
run_logged "generate_synthetic" python data/generate_synthetic_data.py \
    --n_per_class 200 --out data/synthetic_objects.json

echo "########## STEP 3/3: train + evaluate ##########"
run_logged "train_synthetic" python train.py \
    --data data/synthetic_objects.json --outdir figures --run_name synthetic --epochs 30

echo ""
echo "All steps completed. Logs are in logs/, figures in figures/."
