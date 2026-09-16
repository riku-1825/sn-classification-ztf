from __future__ import annotations
import argparse
import json
import os
import sys

import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.logging_utils import get_logger, log_exception_and_exit
from src.preprocessing import (
    records_to_long_df, filter_objects, to_padded_sequences,
    PreprocessConfig, train_val_test_split_by_object,
)
from src.features import build_feature_table
from src.models.lightgbm_model import prepare_xy, train_lightgbm, evaluate_lightgbm
from src.models.gru_model import (
    LightCurveDataset, GRUClassifier, train_gru, predict_gru, get_device,
)
from src.evaluate import summarize, plot_confusion_matrix, plot_roc_ovr, accuracy_vs_length
from src.anomaly import prediction_entropy, flag_low_confidence


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=str, required=True,
                         help="Path to a JSON file of objects (see data/generate_synthetic_data.py or "
                              "data/fetch_real_data.py for the schema)")
    parser.add_argument("--outdir", type=str, default="figures")
    parser.add_argument("--run_name", type=str, default="run",
                         help="Prefix used for log file name and saved figure names "
                              "(e.g. 'synthetic' or 'real') so results don't overwrite each other")
    parser.add_argument("--max_len", type=int, default=100)
    parser.add_argument("--min_detections", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logger, log_path = get_logger(f"train_{args.run_name}")
    logger.info(f"Arguments: {vars(args)}")

    try:
        os.makedirs(args.outdir, exist_ok=True)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

        # ---------------- Load & preprocess ----------------
        logger.info(f"Loading data from {args.data}")
        if not os.path.exists(args.data):
            raise FileNotFoundError(
                f"{args.data} does not exist. Run data/generate_synthetic_data.py "
                f"or data/fetch_real_data.py first (see RUN_COMMANDS.md)."
            )
        objects = json.load(open(args.data))
        if len(objects) == 0:
            raise ValueError(f"{args.data} contains zero objects — nothing to train on.")
        logger.info(f"Loaded {len(objects)} raw objects")

        df = records_to_long_df(objects)
        cfg = PreprocessConfig(max_len=args.max_len, min_detections=args.min_detections)
        df = filter_objects(df, cfg)
        n_obj = df["object_id"].nunique()
        logger.info(f"{n_obj} objects remain after filtering (min_detections={args.min_detections}), "
                    f"{len(df)} total detections")
        if n_obj < 20:
            logger.warning(
                f"Only {n_obj} objects survived filtering — results will be noisy/unreliable. "
                f"Consider lowering --min_detections or fetching more data."
            )
        class_counts = df.groupby("object_id")["label"].first().value_counts().to_dict()
        logger.info(f"Class distribution (objects): {class_counts}")
        if len(class_counts) < 2:
            raise ValueError(f"Only {len(class_counts)} class(es) present after filtering — "
                              f"need at least 2 to train a classifier.")

        # ---------------- LightGBM branch ----------------
        logger.info("--- Feature engineering + LightGBM ---")
        feat_df = build_feature_table(df).fillna(0)
        X, y, cols, le = prepare_xy(feat_df)
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.3, random_state=args.seed, stratify=y
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=args.seed, stratify=y_temp
        )
        logger.info(f"LightGBM split sizes: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")

        lgb_model = train_lightgbm(X_train, y_train, X_val, y_val, num_class=len(le.classes_))
        lgb_res = evaluate_lightgbm(lgb_model, X_test, y_test, le)
        lgb_f1 = summarize(y_test, lgb_res["y_pred"], le.classes_, title="LightGBM")
        logger.info(f"LightGBM macro-F1: {lgb_f1:.4f}")

        # Majority-class baseline: the floor any model must clear to be
        # worth reporting. Computed from the ACTUAL training class
        # distribution, not assumed to be balanced.
        from collections import Counter
        from sklearn.metrics import f1_score
        majority_class = Counter(y_train).most_common(1)[0][0]
        baseline_pred = np.full_like(y_test, fill_value=majority_class)
        baseline_f1 = f1_score(y_test, baseline_pred, average="macro", zero_division=0)
        baseline_acc = float((baseline_pred == y_test).mean())
        logger.info(f"Majority-class baseline ('{le.classes_[majority_class]}'): "
                    f"accuracy={baseline_acc:.4f}, macro-F1={baseline_f1:.4f}")

        cm_path = os.path.join(args.outdir, f"{args.run_name}_lightgbm_confusion_matrix.png")
        roc_path = os.path.join(args.outdir, f"{args.run_name}_lightgbm_roc.png")
        plot_confusion_matrix(y_test, lgb_res["y_pred"], le.classes_,
                               title="LightGBM Confusion Matrix", save_path=cm_path)
        plot_roc_ovr(y_test, lgb_res["proba"], le.classes_,
                     title="LightGBM ROC (one-vs-rest)", save_path=roc_path)
        logger.info(f"Saved: {cm_path}, {roc_path}")

        # ---------------- GRU branch ----------------
        logger.info("--- Sequence prep + GRU ---")
        X_seq, mask, y_seq, oids = to_padded_sequences(df, cfg)
        y_enc = le.transform(y_seq)
        idx_train, idx_val, idx_test = train_val_test_split_by_object(oids, y_seq, seed=args.seed)
        logger.info(f"GRU split sizes (objects): train={len(idx_train)}, val={len(idx_val)}, test={len(idx_test)}")

        ds_train = LightCurveDataset(X_seq[idx_train], mask[idx_train], y_enc[idx_train])
        ds_val = LightCurveDataset(X_seq[idx_val], mask[idx_val], y_enc[idx_val])
        ds_test = LightCurveDataset(X_seq[idx_test], mask[idx_test], y_enc[idx_test])

        dl_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True)
        dl_val = DataLoader(ds_val, batch_size=args.batch_size)
        dl_test = DataLoader(ds_test, batch_size=args.batch_size)

        device = get_device()
        logger.info(f"Training GRU on device: {device}")
        gru_model = GRUClassifier(input_dim=X_seq.shape[2] * X_seq.shape[3], num_classes=len(le.classes_))
        gru_model = train_gru(gru_model, dl_train, dl_val, epochs=args.epochs, lr=args.lr, device=device)

        gru_proba, gru_y_true = predict_gru(gru_model, dl_test, device=device)
        gru_y_pred = np.argmax(gru_proba, axis=1)
        gru_f1 = summarize(gru_y_true, gru_y_pred, le.classes_, title="GRU")
        logger.info(f"GRU macro-F1: {gru_f1:.4f}")

        cm_path = os.path.join(args.outdir, f"{args.run_name}_gru_confusion_matrix.png")
        roc_path = os.path.join(args.outdir, f"{args.run_name}_gru_roc.png")
        acc_len_path = os.path.join(args.outdir, f"{args.run_name}_gru_accuracy_vs_length.png")
        plot_confusion_matrix(gru_y_true, gru_y_pred, le.classes_,
                               title="GRU Confusion Matrix", save_path=cm_path)
        plot_roc_ovr(gru_y_true, gru_proba, le.classes_,
                     title="GRU ROC (one-vs-rest)", save_path=roc_path)
        accuracy_fig, acc_len_result = accuracy_vs_length(
            gru_y_true, gru_y_pred, ds_test.lengths, save_path=acc_len_path, logger=logger
        )
        logger.info(f"Saved: {cm_path}, {roc_path}, {acc_len_path}")

        # ---------------- Confidence / anomaly layer ----------------
        logger.info("--- Confidence / anomaly flagging (prototype) ---")
        # IMPORTANT: the threshold must come from VALIDATION data, not from
        # the test set being flagged. Thresholding on the test set's own
        # median entropy guarantees ~50% of it gets flagged by construction,
        # regardless of how confident the model actually is -- that was a
        # bug in the original version of this script (see RUN_COMMANDS.md /
        # chat history for the writeup). Using a fixed percentile of the
        # validation set's entropy distribution instead means the flagged
        # fraction on test reflects genuine confidence, not a tautology.
        val_proba, _ = predict_gru(gru_model, dl_val, device=device)
        val_entropy = prediction_entropy(val_proba)
        anomaly_percentile = 90  # flag the least-confident ~10% seen on val
        threshold = float(np.percentile(val_entropy, anomaly_percentile))
        logger.info(f"Anomaly threshold set from VALIDATION entropy "
                    f"({anomaly_percentile}th percentile) = {threshold:.4f}")

        entropy = prediction_entropy(gru_proba)
        flags, scores = flag_low_confidence(gru_proba, threshold=threshold, method="entropy")
        logger.info(f"Flagged {int(flags.sum())}/{len(flags)} test objects as low-confidence "
                    f"(threshold tuned on val, {anomaly_percentile}th percentile of val entropy; "
                    f"illustrative prototype, not a validated anomaly detector)")

        # ---------------- Summary ----------------
        summary_path = os.path.join(args.outdir, f"{args.run_name}_summary.json")
        with open(summary_path, "w") as f:
            json.dump({
                "run_name": args.run_name,
                "data_path": args.data,
                "n_objects_used": int(n_obj),
                "class_distribution": class_counts,
                "majority_class_baseline": {
                    "class": str(le.classes_[majority_class]),
                    "accuracy": baseline_acc,
                    "macro_f1": float(baseline_f1),
                },
                "lightgbm_macro_f1": float(lgb_f1),
                "gru_macro_f1": float(gru_f1),
                "device": str(device),
                "anomaly_threshold_val_p90_entropy": threshold,
                "anomaly_flagged_n": int(flags.sum()),
                "anomaly_flagged_total": int(len(flags)),
                "accuracy_vs_length": acc_len_result,
            }, f, indent=2)
        logger.info(f"Saved run summary: {summary_path}")
        logger.info(f"DONE. Majority-baseline macro-F1={baseline_f1:.4f} | "
                    f"LightGBM macro-F1={lgb_f1:.4f} | GRU macro-F1={gru_f1:.4f}")
        logger.info(f"Full log written to: {log_path}")

    except Exception as e:
        log_exception_and_exit(logger, "train.py failed", e)


if __name__ == "__main__":
    main()
