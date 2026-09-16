from __future__ import annotations
import argparse
import json
import os
import sys

import numpy as np
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.logging_utils import get_logger, log_exception_and_exit
from src.preprocessing import (
    records_to_long_df, filter_objects, PreprocessConfig,
    truncate_to_days_since_first_detection,
)
from src.features import build_feature_table
from src.models.lightgbm_model import prepare_xy, train_lightgbm, evaluate_lightgbm
from src.evaluate import summarize

import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--outdir", type=str, default="figures")
    parser.add_argument("--run_name", type=str, default="run")
    parser.add_argument("--min_detections", type=int, default=3)
    parser.add_argument("--cutoffs_days", type=float, nargs="+",
                         default=[3, 7, 14, 21, 30, 60, 9999],
                         help="Day cutoffs since each object's own first detection. "
                              "9999 effectively means 'full light curve'.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logger, log_path = get_logger(f"ablation_early_epoch_{args.run_name}")
    logger.info(f"Arguments: {vars(args)}")

    try:
        os.makedirs(args.outdir, exist_ok=True)
        np.random.seed(args.seed)

        logger.info(f"Loading data from {args.data}")
        objects = json.load(open(args.data))
        df_full = records_to_long_df(objects)
        cfg = PreprocessConfig(min_detections=args.min_detections)
        df_full = filter_objects(df_full, cfg)
        n_obj = df_full["object_id"].nunique()
        logger.info(f"{n_obj} objects available after filtering")

        # Fix ONE object-level split, reused at every cutoff, so results
        # across cutoffs are directly comparable (same test objects
        # throughout -- only the amount of their data visible changes).
        all_object_ids = sorted(df_full["object_id"].unique())
        labels_by_object = df_full.groupby("object_id")["label"].first()
        y_all = labels_by_object.loc[all_object_ids].values

        idx = np.arange(len(all_object_ids))
        idx_train, idx_temp, y_train_ids, y_temp_ids = train_test_split(
            idx, y_all, test_size=0.3, random_state=args.seed, stratify=y_all
        )
        idx_val, idx_test, _, _ = train_test_split(
            idx_temp, y_temp_ids, test_size=0.5, random_state=args.seed, stratify=y_temp_ids
        )
        train_ids = set(np.array(all_object_ids)[idx_train])
        val_ids = set(np.array(all_object_ids)[idx_val])
        test_ids = set(np.array(all_object_ids)[idx_test])
        logger.info(f"Fixed split: train={len(train_ids)}, val={len(val_ids)}, test={len(test_ids)} "
                    f"objects (identical across all cutoffs below)")

        results = []
        for cutoff in args.cutoffs_days:
            df_cut = truncate_to_days_since_first_detection(df_full, cutoff)
            df_cut = filter_objects(df_cut, cfg)  # re-check min_detections after truncation
            surviving_ids = set(df_cut["object_id"].unique())

            feat_df = build_feature_table(df_cut).fillna(0)
            feat_df = feat_df[feat_df["object_id"].isin(surviving_ids)]

            train_mask = feat_df["object_id"].isin(train_ids & surviving_ids)
            val_mask = feat_df["object_id"].isin(val_ids & surviving_ids)
            test_mask = feat_df["object_id"].isin(test_ids & surviving_ids)

            n_train, n_val, n_test = train_mask.sum(), val_mask.sum(), test_mask.sum()
            if n_train < 10 or n_val < 5 or n_test < 5:
                logger.warning(f"cutoff={cutoff}d: too few objects survive truncation "
                                f"(train={n_train}, val={n_val}, test={n_test}) -- skipping")
                continue

            X, y, cols, le = prepare_xy(feat_df)
            X_train, y_train = X[train_mask.values], y[train_mask.values]
            X_val, y_val = X[val_mask.values], y[val_mask.values]
            X_test, y_test = X[test_mask.values], y[test_mask.values]

            model = train_lightgbm(X_train, y_train, X_val, y_val, num_class=len(le.classes_))
            res = evaluate_lightgbm(model, X_test, y_test, le)
            f1 = summarize(y_test, res["y_pred"], le.classes_, title=f"cutoff={cutoff}d")

            results.append({
                "cutoff_days": cutoff,
                "n_train": int(n_train), "n_val": int(n_val), "n_test": int(n_test),
                "macro_f1": float(f1),
            })
            logger.info(f"cutoff={cutoff}d: n_test={n_test}, macro-F1={f1:.4f}")

        if len(results) < 2:
            raise RuntimeError("Fewer than 2 cutoffs produced usable results -- "
                                "cannot plot a trend. Check --cutoffs_days and --min_detections.")

        # ---------------- Plot ----------------
        fig, ax = plt.subplots(figsize=(7, 4.5))
        xs = [r["cutoff_days"] for r in results]
        ys = [r["macro_f1"] for r in results]
        ns = [r["n_test"] for r in results]
        ax.plot(xs, ys, marker="o", color="#1f77b4")
        for x, y_, n in zip(xs, ys, ns):
            ax.annotate(f"n={n}", (x, y_), textcoords="offset points", xytext=(0, 10),
                        ha="center", fontsize=8)
        ax.set_xscale("log")
        ax.set_xlabel("Days since first detection used (log scale)")
        ax.set_ylabel("LightGBM macro-F1")
        ax.set_title("Early-epoch ablation: same objects, same split, varying light-curve length")
        ax.set_ylim(0, 1)
        plot_path = os.path.join(args.outdir, f"{args.run_name}_ablation_early_epoch.png")
        fig.savefig(plot_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved: {plot_path}")

        out_json = os.path.join(args.outdir, f"{args.run_name}_ablation_early_epoch.json")
        with open(out_json, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Saved: {out_json}")
        logger.info(f"DONE. Full log written to: {log_path}")

    except Exception as e:
        log_exception_and_exit(logger, "ablation_early_epoch.py failed", e)


if __name__ == "__main__":
    main()
