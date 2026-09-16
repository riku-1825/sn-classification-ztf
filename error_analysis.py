from __future__ import annotations
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.logging_utils import get_logger, log_exception_and_exit
from src.preprocessing import records_to_long_df, filter_objects, PreprocessConfig, BANDS
from src.features import build_feature_table
from src.models.lightgbm_model import prepare_xy, train_lightgbm, evaluate_lightgbm


BAND_COLORS = {"g": "#2ca02c", "r": "#d62728", "i": "#7f7f7f"}


def plot_object_light_curve(ax, obj_df: pd.DataFrame, title: str):
    for band in BANDS:
        band_df = obj_df[obj_df["band"] == band].sort_values("mjd")
        if len(band_df) == 0:
            continue
        t0 = obj_df["mjd"].min()
        ax.plot(band_df["mjd"] - t0, band_df["flux"], "o-", color=BAND_COLORS.get(band, "black"),
                label=band, markersize=4, linewidth=1)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("Days since first detection", fontsize=8)
    ax.set_ylabel("Flux", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=7, loc="best")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--outdir", type=str, default="figures")
    parser.add_argument("--run_name", type=str, default="run")
    parser.add_argument("--min_detections", type=int, default=3)
    parser.add_argument("--n_plot", type=int, default=9,
                         help="Max number of misclassified light curves to plot.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logger, log_path = get_logger(f"error_analysis_{args.run_name}")
    logger.info(f"Arguments: {vars(args)}")

    try:
        os.makedirs(args.outdir, exist_ok=True)
        objects = json.load(open(args.data))
        df = records_to_long_df(objects)
        cfg = PreprocessConfig(min_detections=args.min_detections)
        df = filter_objects(df, cfg)
        logger.info(f"{df['object_id'].nunique()} objects available after filtering")

        # Reproduce train.py's exact LightGBM split (same seed, same
        # test_size/stratify calls) so "test set" here means the same
        # thing it means in your train.py run.
        feat_df = build_feature_table(df).fillna(0)
        X, y, cols, le = prepare_xy(feat_df)
        idx_all = np.arange(len(feat_df))
        idx_train, idx_temp, y_train, y_temp = train_test_split(
            idx_all, y, test_size=0.3, random_state=args.seed, stratify=y
        )
        idx_val, idx_test, y_val, _ = train_test_split(
            idx_temp, y_temp, test_size=0.5, random_state=args.seed, stratify=y_temp
        )
        X_train, X_val, X_test = X[idx_train], X[idx_val], X[idx_test]
        y_test = y[idx_test]
        test_object_ids = feat_df.iloc[idx_test]["object_id"].values
        test_n_det = feat_df.iloc[idx_test]["total_n_det"].values

        model = train_lightgbm(X_train, y_train, X[idx_val], y_val, num_class=len(le.classes_))
        res = evaluate_lightgbm(model, X_test, y_test, le)
        y_pred = res["y_pred"]
        proba = res["proba"]
        confidence = proba[np.arange(len(y_pred)), y_pred]

        rows = []
        for i in range(len(y_test)):
            rows.append({
                "object_id": test_object_ids[i],
                "true_label": le.classes_[y_test[i]],
                "predicted_label": le.classes_[y_pred[i]],
                "confidence": float(confidence[i]),
                "n_detections": int(test_n_det[i]),
                "correct": bool(y_test[i] == y_pred[i]),
            })
        results_df = pd.DataFrame(rows)
        csv_path = os.path.join(args.outdir, f"{args.run_name}_test_predictions.csv")
        results_df.to_csv(csv_path, index=False)
        logger.info(f"Saved all {len(results_df)} test predictions to: {csv_path}")

        mis_df = results_df[~results_df["correct"]].sort_values("confidence", ascending=False)
        logger.info(f"{len(mis_df)}/{len(results_df)} test objects misclassified "
                    f"({100 * len(mis_df) / len(results_df):.1f}%)")

        if len(mis_df) == 0:
            logger.info("No misclassified objects to plot -- nothing further to do.")
            return

        # Confusion breakdown of misclassifications, logged explicitly
        # (not just visible in the confusion matrix PNG) so it's greppable
        confusion_pairs = mis_df.groupby(["true_label", "predicted_label"]).size().sort_values(ascending=False)
        logger.info("Misclassification breakdown (true -> predicted, count):")
        for (true_lbl, pred_lbl), count in confusion_pairs.items():
            logger.info(f"  {true_lbl} -> {pred_lbl}: {count}")

        # Plot up to n_plot misclassified light curves, highest-confidence
        # (most "confidently wrong") first -- these are the most
        # interesting/concerning cases to inspect.
        n_plot = min(args.n_plot, len(mis_df))
        ncols = 3
        nrows = int(np.ceil(n_plot / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 3.2 * nrows))
        axes = np.atleast_1d(axes).flatten()

        for i in range(n_plot):
            row = mis_df.iloc[i]
            obj_df = df[df["object_id"] == row["object_id"]]
            title = (f"{row['object_id']}\ntrue={row['true_label']} pred={row['predicted_label']} "
                     f"(conf={row['confidence']:.2f}, n={row['n_detections']})")
            plot_object_light_curve(axes[i], obj_df, title)
        for j in range(n_plot, len(axes)):
            axes[j].axis("off")

        fig.suptitle(f"Misclassified test objects, most-confident-wrong first ({args.run_name})", fontsize=11)
        fig.tight_layout()
        plot_path = os.path.join(args.outdir, f"{args.run_name}_misclassified_examples.png")
        fig.savefig(plot_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved: {plot_path}")
        logger.info(f"DONE. Full log written to: {log_path}")

    except Exception as e:
        log_exception_and_exit(logger, "error_analysis.py failed", e)


if __name__ == "__main__":
    main()
