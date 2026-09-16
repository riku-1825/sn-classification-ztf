from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    f1_score, confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, auc, classification_report,
)


def summarize(y_true, y_pred, class_names, title="Model"):
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    print(f"[{title}] macro-F1: {macro_f1:.4f}")
    print(classification_report(y_true, y_pred, target_names=class_names))
    return macro_f1


def plot_confusion_matrix(y_true, y_pred, class_names, title="Confusion Matrix", save_path=None):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(5, 5))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(title)
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_roc_ovr(y_true, y_proba, class_names, title="ROC (one-vs-rest)", save_path=None):
    fig, ax = plt.subplots(figsize=(6, 6))
    for i, cls in enumerate(class_names):
        y_bin = (y_true == i).astype(int)
        fpr, tpr, _ = roc_curve(y_bin, y_proba[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"{cls} (AUC={roc_auc:.2f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def accuracy_vs_length(y_true, y_pred, lengths, bins=(0, 5, 10, 20, 40, 100),
                        save_path=None, min_bin_n=5, logger=None):
    """Plot accuracy as a function of light-curve length (number of
    detections used). Central to the error-analysis / ablation plan.

    Each point is annotated with its sample count (n=...) directly on
    the plot, bins with fewer than `min_bin_n` objects are marked as
    low-confidence (dashed/lighter), and each point gets a 95% Wilson
    score confidence interval error bar -- with n in the range of ~10-30
    typical of a two-week project's held-out set, the point estimate
    alone can look like a strong trend even when adjacent bins'
    intervals overlap heavily. NOTE: this plot is CORRELATIONAL --
    it bins test objects by however many detections they happened to
    have, which can be confounded with other properties of those
    objects (see ablation_early_epoch.py for a controlled version that
    truncates the SAME objects to test this directly).

    Returns a dict of {bin_range: (accuracy, n, ci_lo, ci_hi)} so the
    counts/intervals can also be logged/inspected programmatically, not
    just eyeballed on the plot.
    """
    lengths = np.asarray(lengths)
    correct = (np.asarray(y_true) == np.asarray(y_pred)).astype(int)
    bin_centers, accs, counts, low_n_flags, ci_los, ci_his = [], [], [], [], [], []

    z = 1.96  # 95% CI

    def wilson_ci(k, n, z=1.96):
        if n == 0:
            return (np.nan, np.nan)
        p = k / n
        denom = 1 + z**2 / n
        center = (p + z**2 / (2 * n)) / denom
        halfwidth = (z * np.sqrt((p * (1 - p) + z**2 / (4 * n)) / n)) / denom
        return max(0.0, center - halfwidth), min(1.0, center + halfwidth)

    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (lengths >= lo) & (lengths < hi)
        n = int(mask.sum())
        if n == 0:
            continue
        k = int(correct[mask].sum())
        acc = k / n
        ci_lo, ci_hi = wilson_ci(k, n, z)
        bin_centers.append((lo + hi) / 2)
        accs.append(acc)
        counts.append(n)
        low_n_flags.append(n < min_bin_n)
        ci_los.append(ci_lo)
        ci_his.append(ci_hi)

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for i in range(len(bin_centers)):
        color = "#cccccc" if low_n_flags[i] else "#1f77b4"
        yerr_lo = accs[i] - ci_los[i]
        yerr_hi = ci_his[i] - accs[i]
        ax.errorbar(bin_centers[i], accs[i], yerr=[[yerr_lo], [yerr_hi]],
                     fmt="o", color=color, markersize=9, capsize=4, elinewidth=1.5)
        if i > 0:
            style = "--" if (low_n_flags[i] or low_n_flags[i - 1]) else "-"
            seg_color = "#cccccc" if (low_n_flags[i] or low_n_flags[i - 1]) else "#1f77b4"
            ax.plot(bin_centers[i - 1:i + 1], accs[i - 1:i + 1], linestyle=style, color=seg_color)
        ax.annotate(f"n={counts[i]}", (bin_centers[i], min(accs[i] + yerr_hi + 0.03, 0.97)),
                     ha="center", fontsize=8)

    ax.set_xlabel("Number of detections used")
    ax.set_ylabel("Accuracy")
    title = "Accuracy vs. light-curve length (95% Wilson CI)"
    if any(low_n_flags):
        title += "\n(grey points/dashed segments = fewer than "
        title += f"{min_bin_n} objects in that bin -- do not over-interpret)"
    ax.set_title(title, fontsize=11)
    ax.set_ylim(0, 1.05)

    result = {f"[{lo},{hi})": {"accuracy": acc, "n": n, "ci_95_lo": ci_lo, "ci_95_hi": ci_hi}
              for lo, hi, acc, n, ci_lo, ci_hi in zip(bins[:-1], bins[1:], accs, counts, ci_los, ci_his)}

    if logger is not None:
        for (rng, d) in result.items():
            flag = " <-- LOW SAMPLE COUNT, treat as noise" if d["n"] < min_bin_n else ""
            logger.info(f"accuracy_vs_length bin {rng}: accuracy={d['accuracy']:.3f} "
                        f"(95% CI [{d['ci_95_lo']:.3f}, {d['ci_95_hi']:.3f}]), n={d['n']}{flag}")

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig, result
