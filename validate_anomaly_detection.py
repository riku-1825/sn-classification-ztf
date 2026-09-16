from __future__ import annotations
import argparse
import json
import os
import sys

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.logging_utils import get_logger, log_exception_and_exit
from src.preprocessing import (
    records_to_long_df, filter_objects, PreprocessConfig,
    to_padded_sequences, train_val_test_split_by_object, BANDS,
)
from src.models.lightgbm_model import prepare_xy
from src.models.gru_model import (
    LightCurveDataset, GRUClassifier, train_gru, predict_gru, get_device,
)
from src.features import build_feature_table
from src.anomaly import prediction_entropy, flag_low_confidence


def generate_ood_objects(n_objects: int, max_len: int, n_bands: int, seed: int = 0):
    """Generates light curves that look nothing like any trained SN
    class: pure random noise with no rise/decline structure at all.
    This is a deliberately extreme, easy-to-flag-if-working-correctly
    case -- it tests basic mechanism sanity, not subtle real-world
    anomaly discrimination (see module docstring).
    """
    rng = np.random.default_rng(seed)
    X = np.zeros((n_objects, max_len, n_bands, 2), dtype=np.float32)
    mask = np.zeros((n_objects, max_len, n_bands), dtype=np.float32)
    for i in range(n_objects):
        n_det = rng.integers(10, max_len)
        for b in range(n_bands):
            t = np.sort(rng.uniform(0, 1, size=n_det))  # already in the ~[0,1] scaled range
            flux = rng.uniform(-1, 1, size=n_det)  # pure noise, no template shape at all
            X[i, :n_det, b, 0] = t
            X[i, :n_det, b, 1] = flux
            mask[i, :n_det, b] = 1.0
    return X, mask


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--outdir", type=str, default="figures")
    parser.add_argument("--run_name", type=str, default="run")
    parser.add_argument("--min_detections", type=int, default=3)
    parser.add_argument("--n_ood", type=int, default=50,
                         help="Number of synthetic out-of-distribution objects to inject.")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--anomaly_percentile", type=float, default=90,
                         help="Same convention as train.py: flag threshold = this percentile of "
                              "VALIDATION-set entropy.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logger, log_path = get_logger(f"validate_anomaly_{args.run_name}")
    logger.info(f"Arguments: {vars(args)}")

    try:
        os.makedirs(args.outdir, exist_ok=True)
        objects = json.load(open(args.data))
        df = records_to_long_df(objects)
        cfg = PreprocessConfig(min_detections=args.min_detections)
        df = filter_objects(df, cfg)
        logger.info(f"{df['object_id'].nunique()} objects available after filtering")

        # Train a GRU exactly as train.py does (same seed/split) so this
        # validation is checking the SAME model train.py would report on.
        feat_df = build_feature_table(df).fillna(0)
        _, y_all, _, le = prepare_xy(feat_df)

        X_seq, mask, y_seq, oids = to_padded_sequences(df, cfg)
        y_enc = le.transform(y_seq)
        idx_train, idx_val, idx_test = train_val_test_split_by_object(oids, y_seq, seed=args.seed)

        ds_train = LightCurveDataset(X_seq[idx_train], mask[idx_train], y_enc[idx_train])
        ds_val = LightCurveDataset(X_seq[idx_val], mask[idx_val], y_enc[idx_val])
        ds_test = LightCurveDataset(X_seq[idx_test], mask[idx_test], y_enc[idx_test])
        dl_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True)
        dl_val = DataLoader(ds_val, batch_size=args.batch_size)
        dl_test = DataLoader(ds_test, batch_size=args.batch_size)

        device = get_device()
        logger.info(f"Training GRU on device: {device}")
        model = GRUClassifier(input_dim=X_seq.shape[2] * X_seq.shape[3], num_classes=len(le.classes_))
        model = train_gru(model, dl_train, dl_val, epochs=args.epochs, lr=args.lr, device=device)

        val_proba, _ = predict_gru(model, dl_val, device=device)
        val_entropy = prediction_entropy(val_proba)
        threshold = float(np.percentile(val_entropy, args.anomaly_percentile))
        logger.info(f"Anomaly threshold (p{args.anomaly_percentile} of val entropy) = {threshold:.4f}")

        test_proba, test_y_true = predict_gru(model, dl_test, device=device)
        test_y_pred = np.argmax(test_proba, axis=1)
        test_entropy = prediction_entropy(test_proba)
        is_error = (test_y_pred != test_y_true).astype(int)

        # ---------------- Check 1: correctness-confidence correlation ----------------
        if is_error.sum() == 0 or is_error.sum() == len(is_error):
            logger.warning("Test set has 0 errors or 100% errors -- ROC-AUC of entropy vs. "
                            "correctness is undefined (need both classes present). Skipping check 1.")
            auc_entropy_vs_error = None
        else:
            auc_entropy_vs_error = float(roc_auc_score(is_error, test_entropy))
            logger.info(f"CHECK 1 -- ROC-AUC(entropy predicts misclassification) = "
                        f"{auc_entropy_vs_error:.4f} on real held-out test set "
                        f"({int(is_error.sum())}/{len(is_error)} test objects were errors). "
                        + ("Meaningfully above chance (0.5) -- confidence carries real signal "
                           "about correctness." if auc_entropy_vs_error > 0.6
                           else "Close to chance (0.5) -- confidence signal may not usefully "
                                "predict errors on this dataset; interpret flagged objects with "
                                "caution."))

        test_flags, _ = flag_low_confidence(test_proba, threshold=threshold, method="entropy")
        flagged_error_rate = float(is_error[test_flags].mean()) if test_flags.sum() > 0 else None
        unflagged_error_rate = float(is_error[~test_flags].mean()) if (~test_flags).sum() > 0 else None
        logger.info(f"Error rate among FLAGGED test objects: {flagged_error_rate}, "
                    f"among UNFLAGGED: {unflagged_error_rate} "
                    "(flagged should be higher if the mechanism is working)")

        # ---------------- Check 2: synthetic OOD injection ----------------
        max_len = X_seq.shape[1]
        n_bands = X_seq.shape[2]
        ood_X, ood_mask = generate_ood_objects(args.n_ood, max_len, n_bands, seed=args.seed)
        ood_labels = np.zeros(args.n_ood, dtype=np.int64)  # dummy, unused for scoring
        ood_ds = LightCurveDataset(ood_X, ood_mask, ood_labels)
        ood_dl = DataLoader(ood_ds, batch_size=args.batch_size)
        ood_proba, _ = predict_gru(model, ood_dl, device=device)
        ood_entropy = prediction_entropy(ood_proba)
        ood_flags, _ = flag_low_confidence(ood_proba, threshold=threshold, method="entropy")
        ood_recall = float(ood_flags.mean())
        normal_fpr = float(test_flags.mean())
        logger.info(f"CHECK 2 -- of {args.n_ood} synthetic OOD (pure-noise) objects, "
                    f"{ood_flags.sum()}/{args.n_ood} ({100 * ood_recall:.1f}%) were flagged "
                    f"as low-confidence, vs. {100 * normal_fpr:.1f}% of normal real test objects "
                    "(the false-positive-like rate). "
                    + ("Mechanism correctly distinguishes obvious OOD inputs from normal ones."
                       if ood_recall > normal_fpr + 0.15
                       else "Mechanism does NOT clearly distinguish obvious OOD inputs from "
                            "normal test objects -- this is a real finding, not a formality, "
                            "and should be reported as a limitation if it holds."))

        summary = {
            "threshold": threshold,
            "anomaly_percentile": args.anomaly_percentile,
            "check1_correctness_confidence_correlation": {
                "auc_entropy_vs_is_error": auc_entropy_vs_error,
                "n_test": int(len(is_error)),
                "n_errors": int(is_error.sum()),
                "error_rate_among_flagged": flagged_error_rate,
                "error_rate_among_unflagged": unflagged_error_rate,
            },
            "check2_synthetic_ood_injection": {
                "n_ood_objects": args.n_ood,
                "ood_flag_rate": ood_recall,
                "normal_test_flag_rate": normal_fpr,
                "note": "OOD objects are synthetic pure-noise light curves, a proxy for "
                        "'genuinely anomalous', NOT real astrophysical anomalies. See module "
                        "docstring for scope.",
            },
        }
        out_json = os.path.join(args.outdir, f"{args.run_name}_anomaly_validation.json")
        with open(out_json, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Saved: {out_json}")

        # ---------------- Plot: entropy distributions ----------------
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.hist(test_entropy[is_error == 0], bins=15, alpha=0.6, label="Correct predictions (real test)", color="#1f77b4")
        ax.hist(test_entropy[is_error == 1], bins=15, alpha=0.6, label="Misclassified (real test)", color="#d62728")
        ax.hist(ood_entropy, bins=15, alpha=0.6, label="Synthetic OOD (pure noise)", color="#7f7f7f")
        ax.axvline(threshold, color="black", linestyle="--", label=f"Flag threshold (p{args.anomaly_percentile} val)")
        ax.set_xlabel("Prediction entropy")
        ax.set_ylabel("Count")
        ax.set_title("Entropy distribution: correct vs. misclassified vs. synthetic OOD", fontsize=11)
        ax.legend(fontsize=8)
        plot_path = os.path.join(args.outdir, f"{args.run_name}_anomaly_validation.png")
        fig.savefig(plot_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved: {plot_path}")
        logger.info(f"DONE. Full log written to: {log_path}")

    except Exception as e:
        log_exception_and_exit(logger, "validate_anomaly_detection.py failed", e)


if __name__ == "__main__":
    main()
