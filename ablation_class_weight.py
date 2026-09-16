from __future__ import annotations
import argparse
import json
import os
import sys

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

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
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logger, log_path = get_logger(f"ablation_class_weight_{args.run_name}")
    logger.info(f"Arguments: {vars(args)}")

    try:
        os.makedirs(args.outdir, exist_ok=True)
        objects = json.load(open(args.data))
        df = records_to_long_df(objects)
        cfg = PreprocessConfig(min_detections=args.min_detections)
        df = filter_objects(df, cfg)

        class_counts = df.groupby("object_id")["label"].first().value_counts().to_dict()
        logger.info(f"Class distribution: {class_counts}")
        imbalance_ratio = max(class_counts.values()) / min(class_counts.values())
        logger.info(f"Class imbalance ratio (max/min count): {imbalance_ratio:.2f}"
                    + ("  (roughly balanced, so weighting is unlikely to matter much)"
                       if imbalance_ratio < 1.3 else ""))

        feat_df = build_feature_table(df).fillna(0)
        X, y, cols, le = prepare_xy(feat_df)
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.3, random_state=args.seed, stratify=y
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=args.seed, stratify=y_temp
        )

        results = {}
        for weight_mode in [None, "balanced"]:
            label = weight_mode or "unweighted"
            model = train_lightgbm(X_train, y_train, X_val, y_val,
                                    num_class=len(le.classes_), class_weight=weight_mode)
            res = evaluate_lightgbm(model, X_test, y_test, le)
            f1 = summarize(y_test, res["y_pred"], le.classes_, title=f"LightGBM ({label})")
            per_class_recall = {
                cls: res["report"][cls]["recall"] for cls in le.classes_
            }
            results[label] = {"macro_f1": float(f1), "per_class_recall": per_class_recall}
            logger.info(f"[{label}] macro-F1={f1:.4f}, per-class recall={per_class_recall}")

        delta = results["balanced"]["macro_f1"] - results["unweighted"]["macro_f1"]
        logger.info(f"Effect of class weighting on macro-F1: {delta:+.4f} "
                    f"({'balanced weighting helped' if delta > 0.01 else 'negligible/no effect' if abs(delta) <= 0.01 else 'balanced weighting hurt'})")

        out_path = os.path.join(args.outdir, f"{args.run_name}_ablation_class_weight.json")
        with open(out_path, "w") as f:
            json.dump({
                "class_distribution": class_counts,
                "imbalance_ratio": imbalance_ratio,
                "results": results,
                "macro_f1_delta_balanced_minus_unweighted": delta,
            }, f, indent=2)
        logger.info(f"Saved: {out_path}")
        logger.info(f"DONE. Full log written to: {log_path}")

    except Exception as e:
        log_exception_and_exit(logger, "ablation_class_weight.py failed", e)


if __name__ == "__main__":
    main()
