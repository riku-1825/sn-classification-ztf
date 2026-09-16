from __future__ import annotations
import argparse
import json
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.logging_utils import get_logger, log_exception_and_exit
from src.preprocessing import (
    records_to_long_df, filter_objects, PreprocessConfig,
    to_padded_sequences, train_val_test_split_by_object,
)
from src.features import build_feature_table
from src.models.lightgbm_model import prepare_xy, train_lightgbm, evaluate_lightgbm
from src.models.gru_model import (
    LightCurveDataset, GRUClassifier, train_gru, predict_gru, get_device,
)
from src.evaluate import summarize


def run_one_seed(df, cfg, seed, epochs, batch_size, lr, device):
    """Runs the identical LightGBM + GRU pipeline train.py uses, but for
    a single given seed, returning just the two macro-F1 numbers. Split
    methodology is copy-identical to train.py (row-level stratified
    split for LightGBM, object-level split for GRU) so results are
    directly comparable to a train.py run with the same seed.
    """
    feat_df = build_feature_table(df).fillna(0)
    X, y, cols, le = prepare_xy(feat_df)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=seed, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=seed, stratify=y_temp)

    lgb_model = train_lightgbm(X_train, y_train, X_val, y_val, num_class=len(le.classes_))
    lgb_res = evaluate_lightgbm(lgb_model, X_test, y_test, le)
    lgb_f1 = summarize(y_test, lgb_res["y_pred"], le.classes_, title=f"LightGBM (seed={seed})")

    X_seq, mask, y_seq, oids = to_padded_sequences(df, cfg)
    y_enc = le.transform(y_seq)
    idx_train, idx_val, idx_test = train_val_test_split_by_object(oids, y_seq, seed=seed)

    ds_train = LightCurveDataset(X_seq[idx_train], mask[idx_train], y_enc[idx_train])
    ds_val = LightCurveDataset(X_seq[idx_val], mask[idx_val], y_enc[idx_val])
    ds_test = LightCurveDataset(X_seq[idx_test], mask[idx_test], y_enc[idx_test])
    dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True)
    dl_val = DataLoader(ds_val, batch_size=batch_size)
    dl_test = DataLoader(ds_test, batch_size=batch_size)

    gru_model = GRUClassifier(input_dim=X_seq.shape[2] * X_seq.shape[3], num_classes=len(le.classes_))
    gru_model = train_gru(gru_model, dl_train, dl_val, epochs=epochs, lr=lr, device=device)
    gru_proba, gru_y_true = predict_gru(gru_model, dl_test, device=device)
    gru_y_pred = np.argmax(gru_proba, axis=1)
    gru_f1 = summarize(gru_y_true, gru_y_pred, le.classes_, title=f"GRU (seed={seed})")

    return float(lgb_f1), float(gru_f1)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--outdir", type=str, default="figures")
    parser.add_argument("--run_name", type=str, default="run")
    parser.add_argument("--min_detections", type=int, default=3)
    parser.add_argument("--n_seeds", type=int, default=5)
    parser.add_argument("--base_seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-3)
    args = parser.parse_args()

    logger, log_path = get_logger(f"multiseed_eval_{args.run_name}")
    logger.info(f"Arguments: {vars(args)}")

    try:
        os.makedirs(args.outdir, exist_ok=True)
        objects = json.load(open(args.data))
        df = records_to_long_df(objects)
        cfg = PreprocessConfig(min_detections=args.min_detections)
        df = filter_objects(df, cfg)
        logger.info(f"{df['object_id'].nunique()} objects available after filtering")

        device = get_device()
        logger.info(f"Training on device: {device}")

        seeds = [args.base_seed + i for i in range(args.n_seeds)]
        lgb_f1s, gru_f1s = [], []
        for seed in seeds:
            logger.info(f"=== Seed {seed} ({seeds.index(seed) + 1}/{len(seeds)}) ===")
            lgb_f1, gru_f1 = run_one_seed(df, cfg, seed, args.epochs, args.batch_size, args.lr, device)
            lgb_f1s.append(lgb_f1)
            gru_f1s.append(gru_f1)
            logger.info(f"Seed {seed}: LightGBM={lgb_f1:.4f}, GRU={gru_f1:.4f}, "
                        f"delta(LGB-GRU)={lgb_f1 - gru_f1:+.4f}")

        lgb_f1s, gru_f1s = np.array(lgb_f1s), np.array(gru_f1s)
        deltas = lgb_f1s - gru_f1s

        summary = {
            "n_seeds": args.n_seeds,
            "seeds": seeds,
            "lightgbm": {"per_seed": lgb_f1s.tolist(), "mean": float(lgb_f1s.mean()), "std": float(lgb_f1s.std())},
            "gru": {"per_seed": gru_f1s.tolist(), "mean": float(gru_f1s.mean()), "std": float(gru_f1s.std())},
            "delta_lightgbm_minus_gru": {
                "per_seed": deltas.tolist(), "mean": float(deltas.mean()), "std": float(deltas.std()),
            },
        }

        mean_delta, std_delta = deltas.mean(), deltas.std()
        if std_delta == 0 or abs(mean_delta) > 2 * std_delta:
            verdict = ("LightGBM" if mean_delta > 0 else "GRU") + \
                      f" is likely a robust winner (mean delta {mean_delta:+.4f}, std {std_delta:.4f})"
        else:
            verdict = (f"Ranking is NOT clearly robust given seed variance "
                       f"(mean delta {mean_delta:+.4f}, std {std_delta:.4f} -- "
                       f"same order of magnitude, don't over-claim a winner)")
        summary["verdict"] = verdict
        logger.info(f"VERDICT: {verdict}")

        out_json = os.path.join(args.outdir, f"{args.run_name}_multiseed_eval.json")
        with open(out_json, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Saved: {out_json}")

        fig, ax = plt.subplots(figsize=(6, 4.5))
        x_positions = [0, 1]
        means = [lgb_f1s.mean(), gru_f1s.mean()]
        stds = [lgb_f1s.std(), gru_f1s.std()]
        ax.bar(x_positions, means, yerr=stds, capsize=6, color=["#1f77b4", "#ff7f0e"], width=0.5)
        rng = np.random.default_rng(0)
        for x, vals in zip(x_positions, [lgb_f1s, gru_f1s]):
            jitter = rng.uniform(-0.08, 0.08, size=len(vals))
            ax.scatter(np.full(len(vals), x) + jitter, vals, color="black", s=20, zorder=5, alpha=0.7)
        ax.set_xticks(x_positions)
        ax.set_xticklabels([f"LightGBM\n(n={args.n_seeds} seeds)", f"GRU\n(n={args.n_seeds} seeds)"])
        ax.set_ylabel("Macro-F1")
        ax.set_ylim(0, 1.05)
        ax.set_title(f"Macro-F1 across {args.n_seeds} random splits (mean +/- std)", fontsize=11)
        plot_path = os.path.join(args.outdir, f"{args.run_name}_multiseed_eval.png")
        fig.savefig(plot_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved: {plot_path}")
        logger.info(f"DONE. Full log written to: {log_path}")

    except Exception as e:
        log_exception_and_exit(logger, "multiseed_eval.py failed", e)


if __name__ == "__main__":
    main()
