from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass

BANDS = ["g", "r", "i"]  # drop "i" if your dataset only has g, r


@dataclass
class PreprocessConfig:
    max_len: int = 100          # max detections per band kept per object
    min_detections: int = 3     # objects with fewer detections are dropped
    normalize: str = "peak"     # "peak" | "zscore"
    fill_value: float = 0.0     # padding value for masked slots
    time_scale_days: float = 100.0  # t_rel is divided by this so it's ~O(1),
                                     # comparable in scale to normalized flux.
                                     # IMPORTANT for GRU training: without this,
                                     # raw day-counts (tens) dominate flux
                                     # values (~[-1, 1]) and gradients collapse.


def records_to_long_df(objects: list[dict]) -> pd.DataFrame:
    """Flatten raw per-object records into a long-format DataFrame:
    one row per (object_id, band, mjd) detection.
    """
    rows = []
    for obj in objects:
        for det in obj["detections"]:
            rows.append({
                "object_id": obj["object_id"],
                "label": obj["label"],
                "mjd": det["mjd"],
                "band": det["band"],
                "flux": det["flux"],
                "flux_err": det.get("flux_err", np.nan),
                "detected": det.get("detected", True),
            })
    df = pd.DataFrame(rows)
    df = df.sort_values(["object_id", "band", "mjd"]).reset_index(drop=True)
    return df


def filter_objects(df: pd.DataFrame, cfg: PreprocessConfig) -> pd.DataFrame:
    """Drop objects with too few detections; drop rows with non-finite flux."""
    df = df[np.isfinite(df["flux"])].copy()
    counts = df.groupby("object_id")["mjd"].count()
    keep_ids = counts[counts >= cfg.min_detections].index
    return df[df["object_id"].isin(keep_ids)].copy()


def normalize_object(sub_df: pd.DataFrame, cfg: PreprocessConfig) -> pd.DataFrame:
    """Normalize flux for a single object's detections (all bands together)."""
    sub_df = sub_df.copy()
    if cfg.normalize == "peak":
        peak = sub_df["flux"].abs().max()
        peak = peak if peak > 0 else 1.0
        sub_df["flux_norm"] = sub_df["flux"] / peak
    elif cfg.normalize == "zscore":
        mu, sigma = sub_df["flux"].mean(), sub_df["flux"].std()
        sigma = sigma if sigma > 0 else 1.0
        sub_df["flux_norm"] = (sub_df["flux"] - mu) / sigma
    else:
        raise ValueError(f"Unknown normalize mode: {cfg.normalize}")
    # time relative to first detection, in days, then scaled to be roughly
    # O(1) so it doesn't dominate the flux feature when fed to the GRU
    sub_df["t_rel"] = (sub_df["mjd"] - sub_df["mjd"].min()) / cfg.time_scale_days
    return sub_df


def to_padded_sequences(df: pd.DataFrame, cfg: PreprocessConfig):
    """Convert long-format df into padded arrays for a sequence model.

    Returns
    -------
    X : np.ndarray, shape [N, max_len, n_bands, 2]   (2 = [t_rel, flux_norm])
    mask : np.ndarray, shape [N, max_len, n_bands]    (1 = real, 0 = padded)
    y : np.ndarray, shape [N]                         (string labels)
    object_ids : list[str]
    """
    object_ids = sorted(df["object_id"].unique())
    n_bands = len(BANDS)
    X = np.full((len(object_ids), cfg.max_len, n_bands, 2), cfg.fill_value, dtype=np.float32)
    mask = np.zeros((len(object_ids), cfg.max_len, n_bands), dtype=np.float32)
    y = []

    for i, oid in enumerate(object_ids):
        sub = df[df["object_id"] == oid]
        sub = normalize_object(sub, cfg)
        y.append(sub["label"].iloc[0])
        for b_idx, band in enumerate(BANDS):
            band_df = sub[sub["band"] == band].sort_values("t_rel")
            n = min(len(band_df), cfg.max_len)
            if n == 0:
                continue
            X[i, :n, b_idx, 0] = band_df["t_rel"].values[:n]
            X[i, :n, b_idx, 1] = band_df["flux_norm"].values[:n]
            mask[i, :n, b_idx] = 1.0

    return X, mask, np.array(y), object_ids


def truncate_to_days_since_first_detection(df: pd.DataFrame, max_days: float) -> pd.DataFrame:
    """Keep only detections within `max_days` of each object's own first
    detection. This simulates "early-epoch-only" classification: same
    objects, same labels, but as if the alert stream had only just
    started for each of them.

    Used for the early-epoch ablation (see ablation_early_epoch.py) to
    test, in a CONTROLLED way, whether classification genuinely gets
    harder with less light-curve coverage -- as opposed to the
    correlational accuracy_vs_length plot in evaluate.py, which measures
    accuracy against however much data each object happened to have and
    so cannot separate "shorter light curve" from "different kind of
    object that happens to have a shorter light curve".
    """
    df = df.copy()
    first_mjd = df.groupby("object_id")["mjd"].transform("min")
    keep = (df["mjd"] - first_mjd) <= max_days
    return df[keep].copy()


def train_val_test_split_by_object(object_ids, y, val_frac=0.15, test_frac=0.15, seed=42):
    """Stratified split BY OBJECT (never by individual alert/detection) to
    avoid data leakage across splits.
    """
    from sklearn.model_selection import train_test_split
    idx = np.arange(len(object_ids))
    idx_train, idx_temp, y_train, y_temp = train_test_split(
        idx, y, test_size=(val_frac + test_frac), random_state=seed, stratify=y
    )
    rel_test = test_frac / (val_frac + test_frac)
    idx_val, idx_test, _, _ = train_test_split(
        idx_temp, y_temp, test_size=rel_test, random_state=seed, stratify=y_temp
    )
    return idx_train, idx_val, idx_test


if __name__ == "__main__":
    print("This module is meant to be imported. See notebooks/01_eda.md "
          "and HOW_TO.md for a worked example.")
