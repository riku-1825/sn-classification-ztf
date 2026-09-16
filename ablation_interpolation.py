from __future__ import annotations
import argparse
import json
import os
import sys

from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.logging_utils import get_logger, log_exception_and_exit
from src.preprocessing import records_to_long_df, filter_objects, PreprocessConfig
from src.features import build_feature_table
from src.models.lightgbm_model import prepare_xy, train_lightgbm, evaluate_lightgbm
from src.evaluate import summarize


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--outdir", type=str, default="figures")
    parser.add_argument("--run_name", type=str, default="run")
    parser.add_argument("--min_detections", type=int, default=3)
    parser.add_argument("--grid_step_days", type=float, default=1.0,
                         help="Regular grid spacing (days) for the interpolated variant.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logger, log_path = get_logger(f"ablation_interpolation_{args.run_name}")
    logger.info(f"Arguments: {vars(args)}")

    try:
        os.makedirs(args.outdir, exist_ok=True)
        objects = json.load(open(args.data))
        df = records_to_long_df(objects)
        cfg = PreprocessConfig(min_detections=args.min_detections)
        df = filter_objects(df, cfg)
        logger.info(f"{df['object_id'].nunique()} objects available after filtering")

        # ONE fixed object-level split, reused for both feature variants,
        # so the comparison isolates the feature-extraction method itself.
        all_ids = sorted(df["object_id"].unique())
        labels_by_object = df.groupby("object_id")["label"].first()
        y_ids = labels_by_object.loc[all_ids].values
        import numpy as np
        idx = np.arange(len(all_ids))
        idx_train, idx_temp, y_tr, y_tmp = train_test_split(idx, y_ids, test_size=0.3, random_state=args.seed, stratify=y_ids)
        idx_val, idx_test, _, _ = train_test_split(idx_temp, y_tmp, test_size=0.5, random_state=args.seed, stratify=y_tmp)
        train_ids = set(np.array(all_ids)[idx_train])
        val_ids = set(np.array(all_ids)[idx_val])
        test_ids = set(np.array(all_ids)[idx_test])
        logger.info(f"Fixed split: train={len(train_ids)}, val={len(val_ids)}, test={len(test_ids)}")

        results = {}
        for mode in ["raw", "interpolated"]:
            interpolate = (mode == "interpolated")
            feat_df = build_feature_table(df, interpolate=interpolate,
                                           grid_step_days=args.grid_step_days).fillna(0)
            train_mask = feat_df["object_id"].isin(train_ids)
            val_mask = feat_df["object_id"].isin(val_ids)
            test_mask = feat_df["object_id"].isin(test_ids)

            X, y, cols, le = prepare_xy(feat_df)
            X_train, y_train = X[train_mask.values], y[train_mask.values]
            X_val, y_val = X[val_mask.values], y[val_mask.values]
            X_test, y_test = X[test_mask.values], y[test_mask.values]

            model = train_lightgbm(X_train, y_train, X_val, y_val, num_class=len(le.classes_))
            res = evaluate_lightgbm(model, X_test, y_test, le)
            f1 = summarize(y_test, res["y_pred"], le.classes_, title=f"LightGBM ({mode})")
            results[mode] = {"macro_f1": float(f1), "n_test": int(len(y_test))}
            logger.info(f"[{mode}] macro-F1={f1:.4f} (n_test={len(y_test)})")

        delta = results["interpolated"]["macro_f1"] - results["raw"]["macro_f1"]
        verdict = ("interpolation helped" if delta > 0.01
                    else "interpolation hurt" if delta < -0.01
                    else "negligible difference")
        logger.info(f"Effect of interpolation on macro-F1: {delta:+.4f} ({verdict})")

        out_path = os.path.join(args.outdir, f"{args.run_name}_ablation_interpolation.json")
        with open(out_path, "w") as f:
            json.dump({
                "grid_step_days": args.grid_step_days,
                "results": results,
                "macro_f1_delta_interpolated_minus_raw": delta,
                "verdict": verdict,
            }, f, indent=2)
        logger.info(f"Saved: {out_path}")
        logger.info(f"DONE. Full log written to: {log_path}")

    except Exception as e:
        log_exception_and_exit(logger, "ablation_interpolation.py failed", e)


if __name__ == "__main__":
    main()
